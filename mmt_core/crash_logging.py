"""Early crash logging helpers for persistent on-disk diagnostics."""

from __future__ import annotations

import atexit
from datetime import datetime, timezone
import faulthandler
import json
import os
from pathlib import Path
import signal
import sys
import threading
import traceback
from typing import Any, TextIO


_LOCK = threading.RLock()
_INSTALLED = False
_ATEXIT_REGISTERED = False
_QT_HANDLER_INSTALLED = False
_TRACEBACK_LATER_INSTALLED = False

_WORKSPACE_ROOT: Path | None = None
_WORKSPACE_LOG_DIR: Path | None = None
_PROJECT_ROOT: Path | None = None
_PROJECT_LOG_DIR: Path | None = None

_CRASH_HANDLE: TextIO | None = None
_FAULT_HANDLE: TextIO | None = None
_QT_HANDLE: TextIO | None = None
_PREVIOUS_QT_HANDLER: Any | None = None


def install_crash_logging(workspace_root: Path | str | None = None) -> None:
    """Install persistent crash logging hooks as early as possible."""

    global _INSTALLED
    try:
        with _LOCK:
            _ensure_workspace_handles_locked(workspace_root)
            if not _INSTALLED:
                _install_faulthandler_locked()
                _install_python_hooks_locked()
                _install_qt_message_handler_locked()
                _register_atexit_locked()
                _configure_traceback_interval_locked()
                _INSTALLED = True
            _write_line_locked(
                _CRASH_HANDLE,
                _format_line("info", "crash logging installed", workspace_root=str(_WORKSPACE_ROOT or "")),
            )
    except Exception:
        pass


def write_crash_breadcrumb(message: str, **details: Any) -> None:
    """Write a lightweight crash breadcrumb to the persistent crash log."""

    level = str(details.pop("level", "info") or "info")
    try:
        with _LOCK:
            _ensure_workspace_handles_locked(None)
            _write_line_locked(_CRASH_HANDLE, _format_line(level, message, **details))
    except Exception:
        pass


def flush_crash_logs() -> None:
    """Flush crash-related log files to disk."""

    try:
        with _LOCK:
            for handle in (_CRASH_HANDLE, _FAULT_HANDLE, _QT_HANDLE):
                _flush_handle_locked(handle)
    except Exception:
        pass


def set_project_log_dir(project_root: Path | str | None) -> None:
    """Remember the active project log directory for cross-log breadcrumbs."""

    global _PROJECT_ROOT, _PROJECT_LOG_DIR
    try:
        with _LOCK:
            if project_root is None or not str(project_root).strip():
                _PROJECT_ROOT = None
                _PROJECT_LOG_DIR = None
                _write_line_locked(_CRASH_HANDLE, _format_line("info", "project log directory cleared"))
                return
            _PROJECT_ROOT = Path(project_root).expanduser().resolve()
            _PROJECT_LOG_DIR = _PROJECT_ROOT / "cache" / "logs"
            _PROJECT_LOG_DIR.mkdir(parents=True, exist_ok=True)
            _write_line_locked(
                _CRASH_HANDLE,
                _format_line(
                    "info",
                    "project log directory set",
                    project_root=str(_PROJECT_ROOT),
                    project_log_dir=str(_PROJECT_LOG_DIR),
                ),
            )
    except Exception:
        pass


def dump_all_thread_tracebacks(reason: str = "") -> None:
    """Force a full faulthandler traceback dump for all threads."""

    try:
        with _LOCK:
            _ensure_workspace_handles_locked(None)
            if reason:
                _write_line_locked(
                    _CRASH_HANDLE,
                    _format_line("warning", "dumping all thread tracebacks", reason=reason),
                )
            if _FAULT_HANDLE is not None:
                faulthandler.dump_traceback(file=_FAULT_HANDLE, all_threads=True)
                _flush_handle_locked(_FAULT_HANDLE)
    except Exception:
        pass


def _ensure_workspace_handles_locked(workspace_root: Path | str | None) -> None:
    global _WORKSPACE_ROOT, _WORKSPACE_LOG_DIR, _CRASH_HANDLE, _FAULT_HANDLE, _QT_HANDLE

    if _WORKSPACE_LOG_DIR is None:
        _WORKSPACE_ROOT = _resolve_root(workspace_root)
        _WORKSPACE_LOG_DIR = _WORKSPACE_ROOT / "logs"
        _WORKSPACE_LOG_DIR.mkdir(parents=True, exist_ok=True)
        try:
            (_WORKSPACE_LOG_DIR / "runtime_diagnostics.log").touch(exist_ok=True)
        except Exception:
            pass

    if _CRASH_HANDLE is None:
        _CRASH_HANDLE = _open_log_handle(_WORKSPACE_LOG_DIR / "crash.log")
    if _FAULT_HANDLE is None:
        _FAULT_HANDLE = _open_log_handle(_WORKSPACE_LOG_DIR / "faulthandler.log")
    if _QT_HANDLE is None:
        _QT_HANDLE = _open_log_handle(_WORKSPACE_LOG_DIR / "qt_messages.log")


def _resolve_root(workspace_root: Path | str | None) -> Path:
    if workspace_root is None or not str(workspace_root).strip():
        return Path.cwd().resolve()
    return Path(workspace_root).expanduser().resolve()


def _open_log_handle(path: Path) -> TextIO | None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open("a", encoding="utf-8", buffering=1)
    except Exception:
        return None


def _write_line_locked(handle: TextIO | None, line: str) -> None:
    if handle is None:
        return
    try:
        handle.write(line)
        _flush_handle_locked(handle)
    except Exception:
        pass


def _flush_handle_locked(handle: TextIO | None) -> None:
    if handle is None:
        return
    try:
        handle.flush()
    except Exception:
        return
    try:
        os.fsync(handle.fileno())
    except Exception:
        pass


def _format_line(level: str, message: str, **details: Any) -> str:
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    thread = threading.current_thread()
    parts = [
        f"timestamp={timestamp}",
        f"pid={os.getpid()}",
        f"thread={_sanitize_text(thread.name or 'unknown')}",
        f"tid={threading.get_ident()}",
        f"level={_sanitize_text(level.upper() or 'INFO')}",
        f"message={_sanitize_text(message)}",
    ]
    clean_details = _clean_details(details)
    if clean_details:
        parts.append(f"details={json.dumps(clean_details, ensure_ascii=True, sort_keys=True)}")
    return " ".join(parts).rstrip() + "\n"


def _sanitize_text(value: Any) -> str:
    text = str(value or "")
    return text.replace("\r", "\\r").replace("\n", "\\n")


def _clean_details(details: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in details.items():
        name = str(key or "").strip()
        if not name:
            continue
        clean[name] = _stringify_detail(value)
    return clean


def _stringify_detail(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return [_stringify_detail(item) for item in list(value)]
    if isinstance(value, dict):
        return {str(key): _stringify_detail(item) for key, item in value.items()}
    return _sanitize_text(value)


def _install_faulthandler_locked() -> None:
    try:
        if _FAULT_HANDLE is not None:
            faulthandler.enable(file=_FAULT_HANDLE, all_threads=True)
            _flush_handle_locked(_FAULT_HANDLE)
    except Exception:
        pass
    try:
        sigbreak = getattr(signal, "SIGBREAK", None)
        if sigbreak is not None and _FAULT_HANDLE is not None:
            faulthandler.register(sigbreak, file=_FAULT_HANDLE, all_threads=True, chain=True)
    except Exception:
        pass


def _install_python_hooks_locked() -> None:
    def _sys_hook(exc_type: type[BaseException], exc_value: BaseException, exc_tb: Any) -> None:
        try:
            write_crash_breadcrumb(
                "unhandled sys exception",
                level="critical",
                exception_type=getattr(exc_type, "__name__", str(exc_type)),
                exception=str(exc_value),
                traceback="".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
            )
            dump_all_thread_tracebacks("sys.excepthook")
            flush_crash_logs()
        except Exception:
            pass

    sys.excepthook = _sys_hook

    if hasattr(threading, "excepthook"):
        def _threading_hook(args: Any) -> None:
            try:
                write_crash_breadcrumb(
                    "unhandled thread exception",
                    level="critical",
                    exception_type=getattr(getattr(args, "exc_type", None), "__name__", ""),
                    exception=str(getattr(args, "exc_value", "")),
                    thread_name=str(getattr(getattr(args, "thread", None), "name", "") or ""),
                    traceback="".join(
                        traceback.format_exception(
                            getattr(args, "exc_type", Exception),
                            getattr(args, "exc_value", Exception("unknown thread exception")),
                            getattr(args, "exc_traceback", None),
                        )
                    ),
                )
                dump_all_thread_tracebacks("threading.excepthook")
                flush_crash_logs()
            except Exception:
                pass

        threading.excepthook = _threading_hook  # type: ignore[assignment]


def _install_qt_message_handler_locked() -> None:
    global _QT_HANDLER_INSTALLED, _PREVIOUS_QT_HANDLER
    if _QT_HANDLER_INSTALLED:
        return
    try:
        from PyQt6.QtCore import QtMsgType, qInstallMessageHandler
    except Exception:
        return

    level_map = {
        QtMsgType.QtDebugMsg: "debug",
        QtMsgType.QtInfoMsg: "info",
        QtMsgType.QtWarningMsg: "warning",
        QtMsgType.QtCriticalMsg: "critical",
        QtMsgType.QtFatalMsg: "fatal",
    }

    def _qt_handler(msg_type: Any, context: Any, message: str) -> None:
        try:
            level = level_map.get(msg_type, "info")
            details = {
                "qt_category": _qt_context_value(context, "category"),
                "qt_file": _qt_context_value(context, "file"),
                "qt_function": _qt_context_value(context, "function"),
                "qt_line": _qt_context_value(context, "line"),
            }
            line = _format_line(level, f"Qt message: {message}", **details)
            with _LOCK:
                _write_line_locked(_QT_HANDLE, line)
                _write_line_locked(_CRASH_HANDLE, line)
        except Exception:
            pass
        try:
            if _PREVIOUS_QT_HANDLER is not None:
                _PREVIOUS_QT_HANDLER(msg_type, context, message)
        except Exception:
            pass

    try:
        _PREVIOUS_QT_HANDLER = qInstallMessageHandler(_qt_handler)
        _QT_HANDLER_INSTALLED = True
    except Exception:
        _PREVIOUS_QT_HANDLER = None


def _qt_context_value(context: Any, attribute: str) -> Any:
    try:
        return getattr(context, attribute, "")
    except Exception:
        return ""


def _register_atexit_locked() -> None:
    global _ATEXIT_REGISTERED
    if _ATEXIT_REGISTERED:
        return
    atexit.register(flush_crash_logs)
    atexit.register(_close_handles_safely)
    _ATEXIT_REGISTERED = True


def _configure_traceback_interval_locked() -> None:
    global _TRACEBACK_LATER_INSTALLED
    if _TRACEBACK_LATER_INSTALLED:
        return
    interval_text = str(os.environ.get("MMT_DUMP_TRACEBACK_INTERVAL_SECONDS", "") or "").strip()
    if not interval_text:
        return
    try:
        interval = int(interval_text)
    except Exception:
        interval = 0
    if interval <= 0 or _FAULT_HANDLE is None:
        return
    try:
        faulthandler.dump_traceback_later(interval, repeat=True, file=_FAULT_HANDLE)
        _TRACEBACK_LATER_INSTALLED = True
        _write_line_locked(
            _CRASH_HANDLE,
            _format_line(
                "warning",
                "scheduled repeated traceback dump",
                interval_seconds=interval,
            ),
        )
    except Exception:
        pass


def _close_handles_safely() -> None:
    try:
        with _LOCK:
            flush_crash_logs()
            _close_handle_locked(_QT_HANDLE)
            _close_handle_locked(_FAULT_HANDLE)
            _close_handle_locked(_CRASH_HANDLE)
    except Exception:
        pass


def _close_handle_locked(handle: TextIO | None) -> None:
    if handle is None:
        return
    try:
        handle.close()
    except Exception:
        pass


__all__ = [
    "dump_all_thread_tracebacks",
    "flush_crash_logs",
    "install_crash_logging",
    "set_project_log_dir",
    "write_crash_breadcrumb",
]
