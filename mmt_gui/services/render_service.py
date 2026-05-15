"""Resident render service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mmt_core.render_io import load_render_json, render_image_path, render_json_path, summarize_render_json
from mmt_core.render_stage import prepare_render_for_page, run_render_for_page, run_render_for_pages
from mmt_gui.workers import (
    RenderPageResult,
    RenderPreparationPageResult,
    RenderPreparationTask,
    RenderPreparationWorkerResult,
    RenderTask,
    RenderWorkerResult,
)

from .base_service import BaseService, WorkerSignalsBridge
from .models import ServiceCommand


class RenderService(BaseService):
    def __init__(self, *, scheduler: Any | None = None, startup_options: dict | None = None) -> None:
        super().__init__("render", scheduler=scheduler, startup_options=startup_options)

    def on_initialize(self) -> None:
        self._emit_status("loading", "Loading fonts/render backend...")
        if not bool(self.startup_options.get("preload_render", True)):
            self._emit_log("info", "Render backend preload disabled by settings.")
            return
        from text_rendering import find_fallback_font_for_text

        find_fallback_font_for_text("Render", None)
        self._emit_log("info", "Render service is ready.")

    def execute_command(self, command: ServiceCommand, bridge: WorkerSignalsBridge) -> Any:
        task = command.task
        if isinstance(task, RenderPreparationTask):
            return self._run_prepare_task(command, task, bridge)
        if isinstance(task, RenderTask):
            return self._run_render_task(command, task, bridge)
        if command.action == "status":
            return {"ready": True}
        raise RuntimeError(f"Render service does not support action '{command.action}'.")

    def _run_prepare_task(
        self,
        command: ServiceCommand,
        task: RenderPreparationTask,
        bridge: WorkerSignalsBridge,
    ) -> RenderPreparationWorkerResult:
        if task.project is None:
            raise ValueError("No project was provided for render preparation.")
        if not task.image_relative_paths:
            raise ValueError("No pages were provided for render preparation.")

        total_pages = len(task.image_relative_paths)
        page_results: list[RenderPreparationPageResult] = []
        bridge.message.emit(f"Preparing render metadata for {total_pages} page(s).")
        bridge.progress.emit(0)

        for index, image_relative_path in enumerate(task.image_relative_paths, start=1):
            self._check_canceled(command, message="Render preparation canceled before the next page.")
            page_name = Path(image_relative_path).name
            bridge.event.emit(
                {
                    "stage": "render",
                    "event": "page_start",
                    "image_relative_path": str(image_relative_path),
                    "page_index": index,
                    "page_total": total_pages,
                }
            )
            bridge.message.emit(f"[{index}/{total_pages}] Preparing render metadata for {page_name}")
            try:
                json_path = prepare_render_for_page(
                    task.project,
                    image_relative_path,
                    force=task.force,
                    logger=bridge.message.emit,
                )
                render_data = load_render_json(json_path)
                summary = summarize_render_json(render_data)
            except Exception as exc:
                page_results.append(
                    RenderPreparationPageResult(
                        image_relative_path=str(image_relative_path),
                        json_path=None,
                        error=str(exc),
                        summary={},
                    )
                )
                bridge.event.emit(
                    {
                        "stage": "render",
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
                    RenderPreparationPageResult(
                        image_relative_path=str(image_relative_path),
                        json_path=json_path,
                        error=None,
                        summary=summary,
                    )
                )
                bridge.event.emit(
                    {
                        "stage": "render",
                        "event": "page_done",
                        "image_relative_path": str(image_relative_path),
                        "page_index": index,
                        "page_total": total_pages,
                        "output_path": str(json_path),
                        "summary": summary,
                        "message": f"Render metadata ready for {page_name}",
                    }
                )
            bridge.progress.emit(int((index / total_pages) * 100))

        successful_pages = [result for result in page_results if result.json_path is not None]
        if not successful_pages:
            first_error = page_results[0].error if page_results else "Unknown render preparation failure."
            raise RuntimeError(f"Render preparation failed for all pages. {first_error}")
        return RenderPreparationWorkerResult(page_results=page_results)

    def _run_render_task(
        self,
        command: ServiceCommand,
        task: RenderTask,
        bridge: WorkerSignalsBridge,
    ) -> RenderWorkerResult:
        if task.project is None:
            raise ValueError("No project was provided for rendering.")
        if not task.image_relative_paths:
            raise ValueError("No pages were provided for rendering.")
        if not isinstance(task.config, dict):
            raise ValueError("Render worker requires a configuration dictionary.")

        total_pages = len(task.image_relative_paths)
        page_results_map: dict[str, RenderPageResult] = {
            str(image_relative_path): RenderPageResult(image_relative_path=str(image_relative_path))
            for image_relative_path in task.image_relative_paths
        }

        def on_progress(event: dict[str, Any]) -> None:
            payload = dict(event)
            payload.setdefault("stage", "render")
            bridge.event.emit(payload)
            message = str(payload.get("message", "") or "").strip()
            if message:
                bridge.message.emit(message)
            event_name = str(payload.get("event", "") or "")
            image_relative_path = str(payload.get("image_relative_path", "") or "")
            if event_name == "batch_page_start":
                page_index = max(int(payload.get("page_index", 1) or 1), 1)
                page_total = max(int(payload.get("page_total", total_pages) or total_pages), 1)
                bridge.progress.emit(int(((page_index - 1) / page_total) * 100))
                return
            if event_name == "page_done" and image_relative_path:
                page_result = page_results_map.setdefault(
                    image_relative_path,
                    RenderPageResult(image_relative_path=image_relative_path),
                )
                output_path = payload.get("output_path")
                if isinstance(output_path, str) and output_path.strip():
                    page_result.image_path = Path(output_path)
                summary = payload.get("summary")
                if isinstance(summary, dict):
                    page_result.summary = dict(summary)

        bridge.message.emit(f"Starting render for {total_pages} page(s).")
        bridge.progress.emit(0)
        if len(task.image_relative_paths) == 1:
            image_relative_path = task.image_relative_paths[0]
            image_path = run_render_for_page(
                task.project,
                image_relative_path,
                force=task.force,
                font_name=task.config.get("font_name"),
                font_path=task.config.get("font_path"),
                min_font_size=int(task.config.get("min_font_size", 12) or 12),
                max_font_size=int(task.config.get("max_font_size", 72) or 72),
                stroke_enabled=bool(task.config.get("stroke_enabled", True)),
                stroke_width=task.config.get("stroke_width"),
                text_color=task.config.get("text_color"),
                stroke_color=task.config.get("stroke_color"),
                auto_color=bool(task.config.get("auto_color", True)),
                auto_direction=bool(task.config.get("auto_direction", True)),
                vertical_cjk=bool(task.config.get("vertical_cjk", True)),
                save_sprites=bool(task.config.get("save_sprites", True)),
                logger=bridge.message.emit,
                progress_callback=on_progress,
            )
            render_data = load_render_json(render_json_path(task.project, image_relative_path))
            page_results_map[str(image_relative_path)] = RenderPageResult(
                image_relative_path=str(image_relative_path),
                image_path=image_path,
                error=None,
                summary=summarize_render_json(render_data),
            )
            bridge.progress.emit(100)
        else:
            run_render_for_pages(
                task.project,
                task.image_relative_paths,
                force=task.force,
                font_name=task.config.get("font_name"),
                font_path=task.config.get("font_path"),
                min_font_size=int(task.config.get("min_font_size", 12) or 12),
                max_font_size=int(task.config.get("max_font_size", 72) or 72),
                stroke_enabled=bool(task.config.get("stroke_enabled", True)),
                stroke_width=task.config.get("stroke_width"),
                text_color=task.config.get("text_color"),
                stroke_color=task.config.get("stroke_color"),
                auto_color=bool(task.config.get("auto_color", True)),
                auto_direction=bool(task.config.get("auto_direction", True)),
                vertical_cjk=bool(task.config.get("vertical_cjk", True)),
                save_sprites=bool(task.config.get("save_sprites", True)),
                logger=bridge.message.emit,
                progress_callback=on_progress,
            )
            for index, image_relative_path in enumerate(task.image_relative_paths, start=1):
                json_path = render_json_path(task.project, image_relative_path)
                image_path = render_image_path(task.project, image_relative_path)
                if json_path.exists():
                    render_data = load_render_json(json_path)
                    page_results_map[str(image_relative_path)] = RenderPageResult(
                        image_relative_path=str(image_relative_path),
                        image_path=image_path if image_path.exists() else None,
                        error=None if image_path.exists() else "Render output image is missing.",
                        summary=summarize_render_json(render_data),
                    )
                bridge.progress.emit(int((index / total_pages) * 100))

        page_results = list(page_results_map.values())
        successful_pages = [result for result in page_results if result.image_path is not None]
        if not successful_pages:
            first_error = page_results[0].error if page_results else "Unknown render failure."
            raise RuntimeError(f"Render failed for all pages. {first_error}")
        return RenderWorkerResult(page_results=page_results)


__all__ = ["RenderService"]
