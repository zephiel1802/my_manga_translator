"""Qt-facing inpaint service backed by a resident Python runtime thread."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mmt_core.crash_logging import write_crash_breadcrumb
from mmt_core.inpaint_io import inpaint_json_path, load_inpaint_json, summarize_inpaint_json
from mmt_core.inpaint_stage import (
    DEFAULT_CROP_MARGIN,
    DEFAULT_CROP_TRIGGER_SIZE,
    DEFAULT_PAD_MOD,
    DEFAULT_RESIZE_LIMIT,
    prepare_inpaint_mask_for_page,
)
from mmt_gui.workers import (
    InpaintMaskPageResult,
    InpaintMaskTask,
    InpaintMaskWorkerResult,
    InpaintTask,
    LamaModelTask,
)

from .base_service import BaseService, ServiceCanceledError, WorkerSignalsBridge
from .inpaint_runtime import (
    InpaintRuntimeBusyError,
    InpaintRuntimeCallbacks,
    InpaintRuntimeRequest,
    InpaintRuntimeThread,
)
from .models import ServiceCommand
from .resource_scheduler import ResourceScheduler


class InpaintService(BaseService):
    def __init__(self, *, scheduler: ResourceScheduler | None = None, startup_options: dict | None = None) -> None:
        super().__init__("inpaint", scheduler=scheduler, startup_options=startup_options)
        self._runtime: InpaintRuntimeThread | None = None

    def on_initialize(self) -> None:
        if not bool(self.startup_options.get("preload_inpaint", True)):
            raise RuntimeError("Inpaint preload is disabled. Enable startup preload or reload the service.")

        device = str(self.startup_options.get("device", "") or "")
        crop_trigger_size = int(self.startup_options.get("crop_trigger_size", DEFAULT_CROP_TRIGGER_SIZE))
        crop_margin = int(self.startup_options.get("crop_margin", DEFAULT_CROP_MARGIN))
        resize_limit = int(self.startup_options.get("resize_limit", DEFAULT_RESIZE_LIMIT))
        pad_mod = int(self.startup_options.get("pad_mod", DEFAULT_PAD_MOD))

        write_crash_breadcrumb(
            "InpaintService runtime thread starting",
            device=device,
            crop_trigger_size=crop_trigger_size,
            crop_margin=crop_margin,
            resize_limit=resize_limit,
            pad_mod=pad_mod,
        )
        self._runtime = InpaintRuntimeThread(
            device=device,
            crop_trigger_size=crop_trigger_size,
            crop_margin=crop_margin,
            resize_limit=resize_limit,
            pad_mod=pad_mod,
            logger=lambda message: self._emit_log("info", message),
            status_callback=lambda message: self._emit_status("loading", message),
        )
        self._runtime.start()
        if not self._runtime.wait_until_ready(timeout=300.0):
            self._runtime.stop()
            self._runtime = None
            write_crash_breadcrumb(
                "InpaintService runtime error",
                level="critical",
                error="Inpaint runtime did not become ready.",
            )
            raise RuntimeError("Inpaint runtime did not become ready. Check crash logs for the last breadcrumb.")
        if not self._runtime.is_ready():
            error_message = self._runtime.error_message() or "Inpaint runtime failed to initialize."
            self._runtime.stop()
            self._runtime = None
            write_crash_breadcrumb(
                "InpaintService runtime error",
                level="critical",
                error=error_message,
            )
            raise RuntimeError(error_message)
        write_crash_breadcrumb("InpaintService runtime ready")
        self._emit_log("info", "Inpaint runtime thread is ready.")

    def on_shutdown(self) -> None:
        runtime = self._runtime
        self._runtime = None
        if runtime is not None:
            runtime.stop()

    def on_restart(self) -> None:
        runtime = self._runtime
        self._runtime = None
        if runtime is not None:
            runtime.stop()
        self.on_initialize()

    def execute_command(self, command: ServiceCommand, bridge: WorkerSignalsBridge) -> Any:
        task = command.task
        if isinstance(task, InpaintMaskTask):
            return self._run_mask_task(command, task, bridge)
        if isinstance(task, InpaintTask):
            return self._run_inpaint_task(command, task, bridge)
        if isinstance(task, LamaModelTask):
            return self._run_model_task(command, task, bridge)
        if command.action == "status":
            return self._runtime_status()
        raise RuntimeError(f"Inpaint service does not support action '{command.action}'.")

    def _run_mask_task(
        self,
        command: ServiceCommand,
        task: InpaintMaskTask,
        bridge: WorkerSignalsBridge,
    ) -> InpaintMaskWorkerResult:
        if task.project is None:
            raise ValueError("No project was provided for inpaint mask preparation.")
        if not task.image_relative_paths:
            raise ValueError("No pages were provided for inpaint mask preparation.")

        total_pages = len(task.image_relative_paths)
        page_results: list[InpaintMaskPageResult] = []
        bridge.message.emit(f"Preparing inpaint masks for {total_pages} page(s).")
        bridge.progress.emit(0)

        for index, image_relative_path in enumerate(task.image_relative_paths, start=1):
            self._check_canceled(command, message="Inpaint mask preparation canceled before the next page.")
            page_name = Path(image_relative_path).name
            bridge.event.emit(
                {
                    "event": "page_start",
                    "image_relative_path": str(image_relative_path),
                    "page_index": index,
                    "page_total": total_pages,
                }
            )
            bridge.message.emit(f"[{index}/{total_pages}] Preparing mask for {page_name}")
            try:
                mask_path = prepare_inpaint_mask_for_page(
                    task.project,
                    image_relative_path,
                    mask_padding=task.mask_padding,
                    use_bubble_mask=task.use_bubble_mask,
                    force=task.force,
                    logger=bridge.message.emit,
                )
                metadata = load_inpaint_json(inpaint_json_path(task.project, image_relative_path))
                summary = summarize_inpaint_json(metadata)
            except Exception as exc:
                readable_error = f"{page_name}: {exc}"
                bridge.message.emit(f"Inpaint mask preparation failed: {readable_error}")
                page_results.append(
                    InpaintMaskPageResult(
                        image_relative_path=str(image_relative_path),
                        mask_path=None,
                        error=str(exc),
                        summary={},
                    )
                )
                bridge.event.emit(
                    {
                        "event": "page_error",
                        "image_relative_path": str(image_relative_path),
                        "page_index": index,
                        "page_total": total_pages,
                        "message": str(exc),
                        "error": str(exc),
                    }
                )
            else:
                page_results.append(
                    InpaintMaskPageResult(
                        image_relative_path=str(image_relative_path),
                        mask_path=mask_path,
                        error=None,
                        summary=summary,
                    )
                )
                bridge.message.emit(f"Inpaint mask ready: {mask_path}")
                bridge.event.emit(
                    {
                        "event": "mask_ready",
                        "image_relative_path": str(image_relative_path),
                        "page_index": index,
                        "page_total": total_pages,
                        "output_path": str(mask_path),
                        "summary": summary,
                        "message": f"Inpaint mask ready for {page_name}",
                    }
                )
            bridge.progress.emit(int((index / total_pages) * 100))

        successful_pages = [result for result in page_results if result.mask_path is not None]
        if not successful_pages:
            first_error = page_results[0].error if page_results else "Unknown inpaint mask failure."
            raise RuntimeError(f"Inpaint mask preparation failed for all pages. {first_error}")

        failed_count = len([result for result in page_results if result.error is not None])
        if failed_count:
            bridge.message.emit(
                f"Inpaint mask preparation finished with {failed_count} failed page(s) and {len(successful_pages)} successful page(s)."
            )
        else:
            bridge.message.emit(
                f"Inpaint mask preparation finished successfully for {len(successful_pages)} page(s)."
            )
        return InpaintMaskWorkerResult(page_results=page_results)

    def _run_inpaint_task(
        self,
        command: ServiceCommand,
        task: InpaintTask,
        bridge: WorkerSignalsBridge,
    ) -> Any:
        if task.project is None:
            raise ValueError("No project was provided for inpainting.")
        if not task.image_relative_paths:
            raise ValueError("No pages were provided for inpainting.")

        runtime = self._runtime
        if runtime is None or not runtime.is_ready():
            raise RuntimeError("Inpaint runtime is not ready. Restart the Inpaint service.")
        if runtime.is_busy():
            write_crash_breadcrumb(
                "InpaintService runtime busy",
                level="warning",
                command_id=command.command_id,
                action=command.action,
            )
            raise RuntimeError("Inpaint runtime is busy.")

        request = InpaintRuntimeRequest(
            command_id=command.command_id,
            action=command.action,
            project=task.project,
            image_relative_paths=[str(path) for path in task.image_relative_paths],
            force=bool(task.force),
            mask_padding=int(task.mask_padding),
            use_bubble_mask=bool(task.use_bubble_mask),
            use_crop_windows=bool(task.use_crop_windows),
            device=task.device,
            callbacks=InpaintRuntimeCallbacks(
                progress=bridge.progress.emit,
                message=bridge.message.emit,
                event=bridge.event.emit,
            ),
            cancel_token=command.cancel_token,
        )

        write_crash_breadcrumb(
            "InpaintService before submit to InpaintRuntimeThread",
            command_id=command.command_id,
            action=command.action,
            page_total=len(request.image_relative_paths),
        )
        try:
            outcome = runtime.submit_and_wait(
                request,
                accepted_callback=lambda: write_crash_breadcrumb(
                    "InpaintService after submit accepted",
                    command_id=command.command_id,
                    action=command.action,
                ),
            )
        except InpaintRuntimeBusyError as exc:
            write_crash_breadcrumb(
                "InpaintService runtime busy",
                level="warning",
                command_id=command.command_id,
                error=str(exc),
            )
            raise RuntimeError(str(exc)) from exc

        if outcome.canceled:
            write_crash_breadcrumb(
                "InpaintService runtime failed",
                level="warning",
                command_id=command.command_id,
                error=outcome.error or "Inpaint canceled.",
            )
            raise ServiceCanceledError(outcome.error or "Inpaint canceled.")
        if outcome.error:
            write_crash_breadcrumb(
                "InpaintService runtime failed",
                level="critical",
                command_id=command.command_id,
                error=outcome.error,
            )
            raise RuntimeError(outcome.error)

        write_crash_breadcrumb(
            "InpaintService runtime done",
            command_id=command.command_id,
            action=command.action,
        )
        return outcome.result

    def _run_model_task(
        self,
        command: ServiceCommand,
        task: LamaModelTask,
        bridge: WorkerSignalsBridge,
    ) -> Any:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("Inpaint runtime is not ready. Restart the Inpaint service.")
        if runtime.is_busy():
            write_crash_breadcrumb(
                "InpaintService runtime busy",
                level="warning",
                command_id=command.command_id,
                action=task.action,
            )
            raise RuntimeError("Inpaint runtime is busy.")

        request = InpaintRuntimeRequest(
            command_id=command.command_id,
            action=str(task.action or "status").strip().lower(),
            device=task.device,
            callbacks=InpaintRuntimeCallbacks(
                progress=bridge.progress.emit,
                message=bridge.message.emit,
                event=bridge.event.emit,
            ),
            cancel_token=command.cancel_token,
        )

        write_crash_breadcrumb(
            "InpaintService before submit to InpaintRuntimeThread",
            command_id=command.command_id,
            action=request.action,
        )
        try:
            outcome = runtime.submit_and_wait(
                request,
                accepted_callback=lambda: write_crash_breadcrumb(
                    "InpaintService after submit accepted",
                    command_id=command.command_id,
                    action=request.action,
                ),
            )
        except InpaintRuntimeBusyError as exc:
            write_crash_breadcrumb(
                "InpaintService runtime busy",
                level="warning",
                command_id=command.command_id,
                error=str(exc),
            )
            raise RuntimeError(str(exc)) from exc

        if outcome.error:
            write_crash_breadcrumb(
                "InpaintService runtime failed",
                level="critical",
                command_id=command.command_id,
                error=outcome.error,
            )
            raise RuntimeError(outcome.error)

        write_crash_breadcrumb(
            "InpaintService runtime done",
            command_id=command.command_id,
            action=request.action,
        )
        return outcome.result

    def _runtime_status(self) -> dict[str, Any]:
        runtime = self._runtime
        if runtime is None:
            return {
                "loaded": False,
                "device": "",
                "busy": False,
                "load_count": 0,
                "reload_count": 0,
                "signature": "unloaded",
                "message": "Inpaint runtime is not initialized.",
                "ready": False,
                "runtime_error": "Inpaint runtime is not initialized.",
            }
        return runtime.status()


__all__ = ["InpaintService"]
