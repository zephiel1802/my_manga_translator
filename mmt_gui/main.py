"""Entry point for the PyQt6 desktop shell."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import traceback


def main() -> int:
    from mmt_core.crash_logging import (
        dump_all_thread_tracebacks,
        flush_crash_logs,
        install_crash_logging,
        write_crash_breadcrumb,
    )

    workspace_root = Path.cwd()
    install_crash_logging(workspace_root)
    write_crash_breadcrumb("app main entered", workspace_root=str(workspace_root))

    try:
        from PyQt6.QtWidgets import QApplication

        from . import APP_NAME
        from .app_settings import APP_SETTINGS_APPLICATION, APP_SETTINGS_ORGANIZATION
        from .main_window import MainWindow

        app = QApplication(sys.argv)
        write_crash_breadcrumb("QApplication created")
        app.setApplicationName(APP_NAME)
        app.setOrganizationName(APP_SETTINGS_ORGANIZATION)
        app.setOrganizationDomain("local.mmt")
        app.setApplicationDisplayName(APP_SETTINGS_APPLICATION)

        window = MainWindow()
        write_crash_breadcrumb("MainWindow created")
        window.show()
        write_crash_breadcrumb("MainWindow shown")

        if _env_flag("MMT_DUMP_TRACEBACK_ON_START"):
            dump_all_thread_tracebacks("MMT_DUMP_TRACEBACK_ON_START")

        exit_code = int(app.exec())
        write_crash_breadcrumb("application event loop exited", exit_code=exit_code)
        return exit_code
    except SystemExit:
        flush_crash_logs()
        raise
    except Exception as exc:
        write_crash_breadcrumb(
            "unhandled exception in mmt_gui.main",
            level="critical",
            exception_type=type(exc).__name__,
            exception=str(exc),
            traceback=traceback.format_exc(),
        )
        flush_crash_logs()
        return 1
    finally:
        flush_crash_logs()


def _env_flag(name: str) -> bool:
    value = str(os.environ.get(name, "") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    raise SystemExit(main())
