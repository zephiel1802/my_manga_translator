"""Base classes for long-lived resident stage services."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import traceback
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from mmt_core.crash_logging import write_crash_breadcrumb
from .models import (
    CancelToken,
    ServiceCommand,
    ServiceCommandResult,
    ServiceEvent,
    ServiceStatusSnapshot,
    utc_timestamp,
)


class ServiceCanceledError(RuntimeError):
    """Raised when a resident service command is canceled cooperatively."""


@dataclass(slots=True)
class _LogPayload:
    service_name: str
    level: str
    message: str
    command_id: str = ""
    action: str = ""
    lane: str = ""
    thread_name: str = ""
    created_at: str = field(default_factory=utc_timestamp)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "service_name": self.service_name,
            "level": self.level,
            "message": self.message,
            "created_at": self.created_at,
        }
        if self.command_id:
            payload["command_id"] = self.command_id
        if self.action:
            payload["action"] = self.action
        if self.lane:
            payload["lane"] = self.lane
        if self.thread_name:
            payload["thread_name"] = self.thread_name
        return payload


class _Emitter:
    def __init__(self, callback: Callable[..., None]) -> None:
        self.emit = callback


class WorkerSignalsBridge:
    """Compatibility shim so legacy worker callbacks can run inside resident services."""

    def __init__(
        self,
        *,
        progress_callback: Callable[[int], None],
        message_callback: Callable[[str], None],
        event_callback: Callable[[object], None],
    ) -> None:
        self.started = _Emitter(lambda *_args, **_kwargs: None)
        self.progress = _Emitter(progress_callback)
        self.message = _Emitter(message_callback)
        self.event = _Emitter(event_callback)
        self.finished = _Emitter(lambda *_args, **_kwargs: None)
        self.failed = _Emitter(lambda *_args, **_kwargs: None)


class BaseService(QObject):
    """Base QObject that lives in a dedicated long-lived QThread."""

    status_changed = pyqtSignal(object)
    command_started = pyqtSignal(object)
    command_progress = pyqtSignal(object)
    command_event = pyqtSignal(object)
    command_finished = pyqtSignal(object)
    command_failed = pyqtSignal(object)
    command_canceled = pyqtSignal(object)
    log_message = pyqtSignal(object)

    _submit_requested = pyqtSignal(object)
    _cancel_requested = pyqtSignal(str)
    _restart_requested = pyqtSignal()
    _shutdown_requested = pyqtSignal()

    def __init__(
        self,
        service_name: str,
        *,
        scheduler: Any | None = None,
        startup_options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.service_name = str(service_name or "").strip().lower()
        self.scheduler = scheduler
        self.startup_options = dict(startup_options or {})
        self._active_command: ServiceCommand | None = None
        self._stopping = False
        self._ready = False
        self._status = ServiceStatusSnapshot(service_name=self.service_name, state="starting", message="Starting...")
        self._submit_requested.connect(self._on_submit_requested)
        self._cancel_requested.connect(self._on_cancel_requested)
        self._restart_requested.connect(self._on_restart_requested)
        self._shutdown_requested.connect(self._on_shutdown_requested)

    def submit_command(self, command: ServiceCommand) -> None:
        self._submit_requested.emit(command)

    def cancel_command(self, command_id: str) -> None:
        self._cancel_requested.emit(str(command_id or ""))

    def restart_service(self) -> None:
        self._restart_requested.emit()

    def shutdown_service(self) -> None:
        self._shutdown_requested.emit()

    @pyqtSlot()
    def initialize(self) -> None:
        write_crash_breadcrumb("service initialized", service=self.service_name)
        self._emit_status("starting", "Starting resident service...")
        try:
            self.on_initialize()
        except Exception as exc:
            self._ready = False
            write_crash_breadcrumb(
                "service error",
                level="critical",
                service=self.service_name,
                exception=str(exc),
            )
            self._emit_log("error", f"Service startup failed: {exc}")
            self._emit_status("error", str(exc))
            return
        if self._status.state == "error":
            self._ready = False
            return
        if self._status.state == "stopped":
            self._ready = False
            return

        self._ready = True
        write_crash_breadcrumb("service ready", service=self.service_name)
        self._emit_status("ready", "Ready")
        self._emit_status("idle", "Idle")

    def on_initialize(self) -> None:
        """Optional subclass hook for startup preload."""

    def on_shutdown(self) -> None:
        """Optional subclass hook for cleanup."""

    def on_restart(self) -> None:
        """Optional subclass hook for soft service restart."""
        self.on_shutdown()
        self.on_initialize()

    def lane_for_command(self, command: ServiceCommand) -> str | None:
        return None

    def execute_command(self, command: ServiceCommand, bridge: WorkerSignalsBridge) -> Any:
        raise NotImplementedError

    def _emit_status(self, state: str, message: str, **details: Any) -> None:
        self._status = ServiceStatusSnapshot(
            service_name=self.service_name,
            state=str(state or "").strip().lower() or "idle",
            message=str(message or ""),
            active_command_id=self._active_command.command_id if self._active_command is not None else "",
            queued_count=0,
            updated_at=utc_timestamp(),
            details=dict(details),
        )
        self.status_changed.emit(self._status)

    def _emit_log(
        self,
        level: str,
        message: str,
        *,
        command: ServiceCommand | None = None,
        lane: str = "",
    ) -> None:
        thread_name = ""
        try:
            thread = self.thread()
            thread_name = thread.objectName() if thread is not None else ""
        except Exception:
            thread_name = ""
        payload = _LogPayload(
            service_name=self.service_name,
            level=str(level or "info"),
            message=str(message or ""),
            command_id="" if command is None else command.command_id,
            action="" if command is None else command.action,
            lane=lane,
            thread_name=thread_name,
        )
        self.log_message.emit(payload.to_dict())

    def _build_bridge(self, command: ServiceCommand) -> WorkerSignalsBridge:
        return WorkerSignalsBridge(
            progress_callback=lambda value, active_command=command: self._emit_command_progress(
                active_command,
                int(value or 0),
            ),
            message_callback=lambda message, active_command=command: self._emit_log(
                "info",
                str(message or ""),
                command=active_command,
            ),
            event_callback=lambda payload, active_command=command: self._emit_command_event(
                active_command,
                payload,
            ),
        )

    def _emit_command_started(self, command: ServiceCommand, *, message: str = "") -> None:
        event = ServiceEvent(
            command_id=command.command_id,
            service_name=self.service_name,
            action=command.action,
            stage=command.stage,
            event="command_started",
            image_relative_path=command.current_page,
            page_total=max(0, len(command.image_relative_paths)),
            message=message or f"{command.action} started.",
        )
        self.command_started.emit(event.to_payload())

    def _emit_command_progress(self, command: ServiceCommand, progress: int) -> None:
        event = ServiceEvent(
            command_id=command.command_id,
            service_name=self.service_name,
            action=command.action,
            stage=command.stage,
            event="command_progress",
            image_relative_path=command.current_page,
            progress=max(0, min(100, int(progress))),
            page_total=max(0, len(command.image_relative_paths)),
        )
        self.command_progress.emit(event.to_payload())

    def _emit_command_event(self, command: ServiceCommand, payload: object) -> None:
        event_payload = dict(payload) if isinstance(payload, dict) else {"payload": payload}
        event_payload.setdefault("command_id", command.command_id)
        event_payload.setdefault("service_name", self.service_name)
        event_payload.setdefault("action", command.action)
        event_payload.setdefault("stage", command.stage)
        self.command_event.emit(event_payload)

    def _emit_command_finished(self, command: ServiceCommand, *, result: Any, summary: dict[str, Any] | None = None) -> None:
        payload = ServiceCommandResult(
            command_id=command.command_id,
            service_name=self.service_name,
            action=command.action,
            stage=command.stage,
            state="completed",
            result=result,
            summary=dict(summary or {}),
        )
        self.command_finished.emit(payload)

    def _emit_command_failed(self, command: ServiceCommand, *, message: str) -> None:
        payload = ServiceCommandResult(
            command_id=command.command_id,
            service_name=self.service_name,
            action=command.action,
            stage=command.stage,
            state="failed",
            error=str(message or ""),
        )
        self.command_failed.emit(payload)

    def _emit_command_canceled(self, command: ServiceCommand, *, message: str) -> None:
        payload = ServiceCommandResult(
            command_id=command.command_id,
            service_name=self.service_name,
            action=command.action,
            stage=command.stage,
            state="canceled",
            error=str(message or ""),
            canceled=True,
        )
        self.command_canceled.emit(payload)

    def _emit_command_busy(self, command: ServiceCommand, *, message: str) -> None:
        payload = ServiceCommandResult(
            command_id=command.command_id,
            service_name=self.service_name,
            action=command.action,
            stage=command.stage,
            state="busy",
            error=str(message or ""),
        )
        self.command_failed.emit(payload)

    def _check_canceled(self, command: ServiceCommand, *, message: str | None = None) -> None:
        if command.cancel_token.is_cancel_requested():
            raise ServiceCanceledError(message or "Command canceled.")

    def _run_command(self, command: ServiceCommand) -> None:
        write_crash_breadcrumb(
            "_run_command entered",
            service=self.service_name,
            action=command.action,
            command_id=command.command_id,
        )
        self._active_command = command
        self._emit_status("busy", f"Running {command.action}...", action=command.action)
        self._emit_command_started(command, message=f"{command.action} started.")
        try:
            self._check_canceled(command)
            bridge = self._build_bridge(command)
            write_crash_breadcrumb(
                "before execute_command",
                service=self.service_name,
                action=command.action,
                command_id=command.command_id,
            )
            result = self.execute_command(command, bridge)
            write_crash_breadcrumb(
                "after execute_command",
                service=self.service_name,
                action=command.action,
                command_id=command.command_id,
            )
        except ServiceCanceledError as exc:
            write_crash_breadcrumb(
                "command canceled",
                level="warning",
                service=self.service_name,
                action=command.action,
                command_id=command.command_id,
                error=str(exc),
            )
            self._emit_log("warning", str(exc), command=command)
            self._emit_command_canceled(command, message=str(exc))
        except Exception as exc:
            write_crash_breadcrumb(
                "command failed",
                level="critical",
                service=self.service_name,
                action=command.action,
                command_id=command.command_id,
                error=str(exc),
            )
            self._emit_log("error", f"{exc}\n{traceback.format_exc()}", command=command)
            self._emit_command_failed(command, message=str(exc))
        else:
            write_crash_breadcrumb(
                "command finished",
                service=self.service_name,
                action=command.action,
                command_id=command.command_id,
            )
            self._emit_command_finished(command, result=result)
        finally:
            self._active_command = None
            if self._stopping:
                try:
                    self.on_shutdown()
                finally:
                    self._ready = False
                    self._emit_status("stopped", "Stopped")
            else:
                self._emit_status("idle", "Idle")

    @pyqtSlot(object)
    def _on_submit_requested(self, command: object) -> None:
        if not isinstance(command, ServiceCommand):
            return
        write_crash_breadcrumb(
            "_on_submit_requested entered",
            service=self.service_name,
            action=command.action,
            command_id=command.command_id,
        )
        if not self._ready:
            self._emit_command_failed(command, message=f"{self.service_name.title()} service is not ready.")
            return
        if self._stopping:
            self._emit_command_failed(command, message=f"{self.service_name.title()} service is stopping.")
            return
        if self._active_command is not None:
            self._emit_command_busy(command, message=f"{self.service_name.title()} worker is busy.")
            return
        write_crash_breadcrumb(
            "before _run_command",
            service=self.service_name,
            action=command.action,
            command_id=command.command_id,
        )
        self._run_command(command)

    @pyqtSlot(str)
    def _on_cancel_requested(self, command_id: str) -> None:
        normalized_command_id = str(command_id or "").strip()
        if not normalized_command_id:
            return
        if self._active_command is not None and self._active_command.command_id == normalized_command_id:
            self._active_command.cancel_token.request_cancel()
            self._emit_log("warning", "Cancel requested for active command.", command=self._active_command)

    @pyqtSlot()
    def _on_restart_requested(self) -> None:
        if self._active_command is not None:
            self._emit_log("warning", "Restart requested while service is busy; ignoring request.")
            return
        try:
            self._emit_status("loading", "Restarting service...")
            self.on_restart()
        except Exception as exc:
            self._ready = False
            write_crash_breadcrumb(
                "service error",
                level="critical",
                service=self.service_name,
                exception=str(exc),
            )
            self._emit_log("error", f"Service restart failed: {exc}")
            self._emit_status("error", str(exc))
            return
        self._ready = True
        self._emit_status("ready", "Ready")
        self._emit_status("idle", "Idle")

    @pyqtSlot()
    def _on_shutdown_requested(self) -> None:
        write_crash_breadcrumb("service shutdown requested", service=self.service_name)
        self._stopping = True
        if self._active_command is not None:
            self._active_command.cancel_token.request_cancel()
            self._emit_status("stopping", "Stopping...")
            return
        try:
            self.on_shutdown()
        finally:
            self._ready = False
            self._emit_status("stopped", "Stopped")


class WorkerTaskService(BaseService):
    """Compatibility wrapper for services still using legacy stage functions."""

    def __init__(
        self,
        service_name: str,
        *,
        scheduler: Any | None = None,
        action_callbacks: dict[str, Callable[[Any, WorkerSignalsBridge], Any]],
        startup_options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(service_name, scheduler=scheduler, startup_options=startup_options)
        self._action_callbacks = dict(action_callbacks)

    def execute_command(self, command: ServiceCommand, bridge: WorkerSignalsBridge) -> Any:
        callback = self._action_callbacks.get(command.action)
        if callback is None:
            raise RuntimeError(f"{self.service_name} does not support action '{command.action}'.")
        return callback(command.task, bridge)


__all__ = [
    "BaseService",
    "ServiceCanceledError",
    "WorkerSignalsBridge",
    "WorkerTaskService",
]
