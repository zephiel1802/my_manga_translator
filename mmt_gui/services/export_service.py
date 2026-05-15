"""Resident export service."""

from __future__ import annotations

from typing import Any

from mmt_core import ExportConfig
from mmt_core.export_stage import run_export
from mmt_gui.workers import ExportTask, ExportWorkerResult

from .base_service import BaseService, WorkerSignalsBridge
from .models import ServiceCommand


class ExportService(BaseService):
    def __init__(self, *, scheduler: Any | None = None, startup_options: dict | None = None) -> None:
        super().__init__("export", scheduler=scheduler, startup_options=startup_options)

    def on_initialize(self) -> None:
        self._emit_status("loading", "Starting export worker...")
        self._emit_log("info", "Export service is ready.")

    def execute_command(self, command: ServiceCommand, bridge: WorkerSignalsBridge) -> Any:
        task = command.task
        if isinstance(task, ExportTask):
            return self._run_export_task(command, task, bridge)
        if command.action == "status":
            return {"ready": True}
        raise RuntimeError(f"Export service does not support action '{command.action}'.")

    def _run_export_task(
        self,
        command: ServiceCommand,
        task: ExportTask,
        bridge: WorkerSignalsBridge,
    ) -> ExportWorkerResult:
        if task.project is None:
            raise ValueError("No project was provided for export.")

        config = ExportConfig.from_value(task.config)
        bridge.message.emit(
            f"Starting export: source={config.export_source}, scope={config.page_scope}, format={config.output_format}"
        )
        bridge.progress.emit(0)

        def on_progress(event: dict[str, Any]) -> None:
            payload = dict(event)
            payload.setdefault("stage", "export")
            bridge.event.emit(payload)
            message = str(event.get("message", "") or "").strip()
            if message:
                bridge.message.emit(message)
            try:
                progress_value = int(event.get("progress", 0) or 0)
            except Exception:
                progress_value = 0
            bridge.progress.emit(max(0, min(100, progress_value)))

        manifest = run_export(
            task.project,
            current_page=task.current_page,
            selected_pages=task.selected_pages,
            config=config,
            logger=bridge.message.emit,
            progress_callback=on_progress,
        )
        bridge.progress.emit(100)
        bridge.message.emit(
            f"Export completed. Exported: {int(manifest.get('exported_count', 0) or 0)}, "
            f"skipped: {int(manifest.get('skipped_count', 0) or 0)}, "
            f"errors: {int(manifest.get('error_count', 0) or 0)}"
        )
        return ExportWorkerResult(manifest=manifest)


__all__ = ["ExportService"]
