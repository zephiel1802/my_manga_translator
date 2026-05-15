"""Resident translation service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mmt_core.translation_io import load_translation_json, summarize_translation_json, translation_json_path
from mmt_core.translation_stage import (
    initialize_translation_for_page,
    run_translation_for_page,
    run_translation_for_pages,
)
from mmt_gui.workers import (
    TranslationInitializationPageResult,
    TranslationInitializationTask,
    TranslationInitializationWorkerResult,
    TranslationPageResult,
    TranslationTask,
    TranslationWorkerResult,
)

from .base_service import BaseService, WorkerSignalsBridge
from .models import ServiceCommand


class TranslationService(BaseService):
    def __init__(self, *, scheduler: Any | None = None, startup_options: dict | None = None) -> None:
        super().__init__("translation", scheduler=scheduler, startup_options=startup_options)

    def on_initialize(self) -> None:
        self._emit_status("loading", "Starting translation worker...")
        self._emit_log("info", "Translation service is ready.")

    def execute_command(self, command: ServiceCommand, bridge: WorkerSignalsBridge) -> Any:
        task = command.task
        if isinstance(task, TranslationInitializationTask):
            return self._run_initialize_task(command, task, bridge)
        if isinstance(task, TranslationTask):
            return self._run_translation_task(command, task, bridge)
        if command.action == "status":
            return {"ready": True}
        raise RuntimeError(f"Translation service does not support action '{command.action}'.")

    def _run_initialize_task(
        self,
        command: ServiceCommand,
        task: TranslationInitializationTask,
        bridge: WorkerSignalsBridge,
    ) -> TranslationInitializationWorkerResult:
        if task.project is None:
            raise ValueError("No project was provided for translation initialization.")
        if not task.image_relative_paths:
            raise ValueError("No pages were provided for translation initialization.")

        total_pages = len(task.image_relative_paths)
        page_results: list[TranslationInitializationPageResult] = []
        bridge.message.emit(f"Preparing translation caches for {total_pages} page(s).")
        bridge.progress.emit(0)

        for index, image_relative_path in enumerate(task.image_relative_paths, start=1):
            self._check_canceled(command, message="Translation initialization canceled before the next page.")
            page_name = Path(image_relative_path).name
            bridge.event.emit(
                {
                    "stage": "translation",
                    "event": "page_start",
                    "image_relative_path": str(image_relative_path),
                    "page_index": index,
                    "page_total": total_pages,
                }
            )
            bridge.message.emit(f"[{index}/{total_pages}] Initializing translation for {page_name}")
            try:
                json_path = initialize_translation_for_page(
                    task.project,
                    image_relative_path,
                    task.config,
                    force=task.force,
                    logger=bridge.message.emit,
                )
                translation_data = load_translation_json(json_path)
                summary = summarize_translation_json(translation_data)
            except Exception as exc:
                page_results.append(
                    TranslationInitializationPageResult(
                        image_relative_path=str(image_relative_path),
                        json_path=None,
                        error=str(exc),
                        summary={},
                    )
                )
                bridge.event.emit(
                    {
                        "stage": "translation",
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
                    TranslationInitializationPageResult(
                        image_relative_path=str(image_relative_path),
                        json_path=json_path,
                        error=None,
                        summary=summary,
                    )
                )
                bridge.event.emit(
                    {
                        "stage": "translation",
                        "event": "page_done",
                        "image_relative_path": str(image_relative_path),
                        "page_index": index,
                        "page_total": total_pages,
                        "output_path": str(json_path),
                        "summary": summary,
                        "message": f"Translation initialized for {page_name}",
                    }
                )
            bridge.progress.emit(int((index / total_pages) * 100))

        successful_pages = [result for result in page_results if result.json_path is not None]
        if not successful_pages:
            first_error = page_results[0].error if page_results else "Unknown translation initialization failure."
            raise RuntimeError(f"Translation initialization failed for all pages. {first_error}")
        return TranslationInitializationWorkerResult(page_results=page_results)

    def _run_translation_task(
        self,
        command: ServiceCommand,
        task: TranslationTask,
        bridge: WorkerSignalsBridge,
    ) -> TranslationWorkerResult:
        if task.project is None:
            raise ValueError("No project was provided for translation.")
        if not task.image_relative_paths:
            raise ValueError("No pages were provided for translation.")

        total_pages = len(task.image_relative_paths)
        page_results: list[TranslationPageResult] = []
        bridge.message.emit(f"Starting translation for {total_pages} page(s).")
        bridge.progress.emit(0)

        def on_progress(event: dict[str, Any]) -> None:
            payload = dict(event)
            payload.setdefault("stage", "translation")
            bridge.event.emit(payload)
            message = str(payload.get("message", "") or "").strip()
            if message:
                bridge.message.emit(message)
            event_name = str(payload.get("event", "") or "").strip().lower()
            if event_name == "chunk_start":
                chunk_index = max(int(payload.get("chunk_index", 1) or 1), 1)
                chunk_total = max(int(payload.get("chunk_total", 1) or 1), 1)
                bridge.progress.emit(int(((chunk_index - 1) / chunk_total) * 100))

        if task.selected_item_ids_by_page:
            for index, image_relative_path in enumerate(task.image_relative_paths, start=1):
                self._check_canceled(command, message="Translation canceled before the next page.")
                try:
                    json_path = run_translation_for_page(
                        task.project,
                        image_relative_path,
                        task.config,
                        force=task.force,
                        selected_item_ids=task.selected_item_ids_by_page.get(str(image_relative_path)),
                        logger=bridge.message.emit,
                        progress_callback=on_progress,
                    )
                    translation_data = load_translation_json(json_path)
                    summary = summarize_translation_json(translation_data)
                except Exception as exc:
                    page_results.append(
                        TranslationPageResult(
                            image_relative_path=str(image_relative_path),
                            json_path=None,
                            error=str(exc),
                            summary={},
                        )
                    )
                    bridge.event.emit(
                        {
                            "stage": "translation",
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
                        TranslationPageResult(
                            image_relative_path=str(image_relative_path),
                            json_path=json_path,
                            error=None,
                            summary=summary,
                        )
                    )
                bridge.progress.emit(int((index / total_pages) * 100))
        else:
            run_translation_for_pages(
                task.project,
                task.image_relative_paths,
                task.config,
                force=task.force,
                logger=bridge.message.emit,
                progress_callback=on_progress,
            )
            for index, image_relative_path in enumerate(task.image_relative_paths, start=1):
                json_path = translation_json_path(task.project, image_relative_path)
                if json_path is None or not json_path.exists():
                    page_results.append(
                        TranslationPageResult(
                            image_relative_path=str(image_relative_path),
                            json_path=None,
                            error="Translation did not produce an output file for this page.",
                            summary={},
                        )
                    )
                    continue
                translation_data = load_translation_json(json_path)
                page_results.append(
                    TranslationPageResult(
                        image_relative_path=str(image_relative_path),
                        json_path=json_path,
                        error=None,
                        summary=summarize_translation_json(translation_data),
                    )
                )
                bridge.progress.emit(int((index / total_pages) * 100))

        successful_pages = [result for result in page_results if result.json_path is not None]
        if not successful_pages:
            first_error = page_results[0].error if page_results else "Unknown translation failure."
            raise RuntimeError(f"Translation failed for all pages. {first_error}")
        return TranslationWorkerResult(page_results=page_results)


__all__ = ["TranslationService"]
