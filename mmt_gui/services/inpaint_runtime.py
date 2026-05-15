"""Resident Python-threaded runtime for LaMa inpaint execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import Any

from mmt_core.crash_logging import write_crash_breadcrumb
from mmt_core.inpaint_io import inpaint_json_path, load_inpaint_json, summarize_inpaint_json
from mmt_core.inpaint_stage import (
    DEFAULT_CROP_MARGIN,
    DEFAULT_CROP_TRIGGER_SIZE,
    DEFAULT_PAD_MOD,
    DEFAULT_RESIZE_LIMIT,
    LamaInpainterManager,
    load_lama_model,
    run_inpaint_for_page,
    unload_lama_model,
)
from mmt_gui.workers import InpaintPageResult, InpaintWorkerResult, LamaModelTaskResult

from .models import CancelToken


LoggerCallback = Callable[[str], None] | None
StatusCallback = Callable[[str], None] | None
ProgressCallback = Callable[[int], None]
MessageCallback = Callable[[str], None]
EventCallback = Callable[[object], None]


class InpaintRuntimeBusyError(RuntimeError):
    """Raised when the resident inpaint runtime is already processing a command."""


class InpaintRuntimeCanceledError(RuntimeError):
    """Raised when the resident inpaint runtime command is canceled cooperatively."""


@dataclass(slots=True)
class InpaintRuntimeCallbacks:
    progress: ProgressCallback
    message: MessageCallback
    event: EventCallback


@dataclass(slots=True)
class InpaintRuntimeRequest:
    command_id: str
    action: str
    project: Any = None
    image_relative_paths: list[str] = field(default_factory=list)
    force: bool = False
    mask_padding: int = 0
    use_bubble_mask: bool = True
    use_crop_windows: bool = True
    device: str | None = None
    callbacks: InpaintRuntimeCallbacks | None = None
    cancel_token: CancelToken = field(default_factory=CancelToken)


@dataclass(slots=True)
class InpaintRuntimeOutcome:
    result: Any = None
    error: str = ""
    canceled: bool = False


@dataclass(slots=True)
class _RuntimeJob:
    request: InpaintRuntimeRequest
    done_event: threading.Event = field(default_factory=threading.Event)
    outcome: InpaintRuntimeOutcome = field(default_factory=InpaintRuntimeOutcome)


class InpaintRuntimeThread:
    def __init__(
        self,
        *,
        device: str | None,
        crop_trigger_size: int = DEFAULT_CROP_TRIGGER_SIZE,
        crop_margin: int = DEFAULT_CROP_MARGIN,
        resize_limit: int = DEFAULT_RESIZE_LIMIT,
        pad_mod: int = DEFAULT_PAD_MOD,
        logger: LoggerCallback = None,
        status_callback: StatusCallback = None,
    ) -> None:
        self._device = device
        self._crop_trigger_size = int(crop_trigger_size)
        self._crop_margin = int(crop_margin)
        self._resize_limit = int(resize_limit)
        self._pad_mod = int(pad_mod)
        self._logger = logger
        self._status_callback = status_callback
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._stop_requested = False
        self._ready = False
        self._startup_error = ""
        self._manager: LamaInpainterManager | None = None
        self._pending_job: _RuntimeJob | None = None
        self._active_job: _RuntimeJob | None = None
        self._ready_event = threading.Event()

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_requested = False
            self._ready = False
            self._startup_error = ""
            self._pending_job = None
            self._active_job = None
            self._ready_event.clear()
            self._thread = threading.Thread(
                target=self._thread_main,
                name="MMTInpaintRuntimeThread",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        thread_to_join: threading.Thread | None = None
        with self._condition:
            self._stop_requested = True
            active_job = self._active_job
            pending_job = self._pending_job
            if active_job is not None:
                active_job.request.cancel_token.request_cancel()
            if pending_job is not None:
                pending_job.request.cancel_token.request_cancel()
            self._condition.notify_all()
            thread_to_join = self._thread
        if thread_to_join is not None:
            thread_to_join.join(timeout=20.0)

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        return self._ready_event.wait(timeout)

    def is_ready(self) -> bool:
        with self._lock:
            return self._ready and not self._startup_error and self._manager is not None and bool(self._manager.status().get("loaded", False))

    def is_busy(self) -> bool:
        with self._lock:
            return self._active_job is not None or self._pending_job is not None or bool(self._manager is not None and self._manager.status().get("busy", False))

    def error_message(self) -> str:
        with self._lock:
            return self._startup_error

    def status(self) -> dict[str, Any]:
        with self._lock:
            manager_status = self._manager.status() if self._manager is not None else {
                "loaded": False,
                "device": "",
                "busy": False,
                "load_count": 0,
                "reload_count": 0,
                "signature": "unloaded",
                "message": self._startup_error or "LaMa Manga model is not loaded.",
            }
            return {
                **manager_status,
                "ready": self._ready and not self._startup_error,
                "runtime_error": self._startup_error,
            }

    def submit_and_wait(
        self,
        request: InpaintRuntimeRequest,
        *,
        accepted_callback: Callable[[], None] | None = None,
    ) -> InpaintRuntimeOutcome:
        with self._condition:
            thread = self._thread
            if thread is None or not thread.is_alive():
                raise RuntimeError("Inpaint runtime thread is not running.")
            if request.action in {"inpaint_page", "inpaint_pages"} and not self.is_ready():
                raise RuntimeError(self.error_message() or "Inpaint runtime is not ready.")
            if request.action not in {"inpaint_page", "inpaint_pages"} and self._startup_error:
                raise RuntimeError(self.error_message() or "Inpaint runtime is not ready.")
            if self._active_job is not None or self._pending_job is not None:
                raise InpaintRuntimeBusyError("Inpaint runtime is busy.")
            job = _RuntimeJob(request=request)
            self._pending_job = job
            self._condition.notify_all()
        write_crash_breadcrumb(
            "InpaintRuntimeThread submit accepted",
            command_id=request.command_id,
            action=request.action,
        )
        if accepted_callback is not None:
            try:
                accepted_callback()
            except Exception:
                pass

        while not job.done_event.wait(timeout=0.10):
            thread = self._thread
            if thread is not None and not thread.is_alive():
                job.outcome.error = job.outcome.error or "Inpaint runtime thread stopped unexpectedly."
                job.done_event.set()
                break
        return job.outcome

    def _thread_main(self) -> None:
        write_crash_breadcrumb("InpaintRuntimeThread thread started")
        self._emit_status("Starting inpaint runtime...")
        try:
            self._manager = LamaInpainterManager()
            write_crash_breadcrumb("InpaintRuntimeThread preload started")
            self._emit_status("Preloading LaMa Manga model...")
            result = load_lama_model(
                device=self._device,
                crop_trigger_size=self._crop_trigger_size,
                crop_margin=self._crop_margin,
                resize_limit=self._resize_limit,
                pad_mod=self._pad_mod,
                logger=self._startup_log,
                manager=self._manager,
            )
            write_crash_breadcrumb(
                "InpaintRuntimeThread preload done",
                device=str(result.get("device", "") or ""),
            )
            write_crash_breadcrumb(
                "InpaintRuntimeThread runtime ready",
                device=str(result.get("device", "") or ""),
            )
            self._startup_log(str(result.get("message", "") or "LaMa Manga model ready."))
            self._emit_status("Inpaint runtime ready.")
            with self._condition:
                self._ready = True
                self._ready_event.set()
        except Exception as exc:
            with self._condition:
                self._ready = False
                self._startup_error = str(exc)
                self._ready_event.set()
            self._startup_log(f"Inpaint runtime preload failed: {exc}", level="error")
            write_crash_breadcrumb(
                "InpaintRuntimeThread preload failed",
                level="critical",
                error=str(exc),
            )
            self._cleanup_runtime()
            return

        try:
            while True:
                with self._condition:
                    while not self._stop_requested and self._pending_job is None:
                        self._condition.wait(timeout=0.25)
                    if self._stop_requested:
                        break
                    job = self._pending_job
                    self._pending_job = None
                    self._active_job = job

                if job is None:
                    continue

                try:
                    job.outcome.result = self._execute_request(job.request)
                except InpaintRuntimeCanceledError as exc:
                    job.outcome.canceled = True
                    job.outcome.error = str(exc)
                except Exception as exc:
                    job.outcome.error = str(exc)
                    write_crash_breadcrumb(
                        "InpaintRuntimeThread command failed",
                        level="critical",
                        command_id=job.request.command_id,
                        action=job.request.action,
                        error=str(exc),
                    )
                finally:
                    with self._condition:
                        self._active_job = None
                    job.done_event.set()
        finally:
            self._cleanup_runtime()
            write_crash_breadcrumb("InpaintRuntimeThread thread stopped")

    def _execute_request(self, request: InpaintRuntimeRequest) -> Any:
        if request.action in {"inpaint_page", "inpaint_pages"}:
            return self._run_inpaint_request(request)
        if request.action in {"load", "reload", "unload"}:
            return self._run_model_request(request)
        if request.action == "status":
            return self.status()
        raise RuntimeError(f"Unsupported inpaint runtime action: {request.action}")

    def _run_inpaint_request(self, request: InpaintRuntimeRequest) -> InpaintWorkerResult:
        if self._manager is None:
            raise RuntimeError("Inpaint runtime is not initialized.")
        if request.project is None:
            raise ValueError("No project was provided for inpainting.")
        if not request.image_relative_paths:
            raise ValueError("No pages were provided for inpainting.")
        if request.callbacks is None:
            raise RuntimeError("Inpaint runtime callbacks are missing.")

        callbacks = request.callbacks
        total_pages = len(request.image_relative_paths)
        page_results: list[InpaintPageResult] = []
        model_status = self._manager.status()

        write_crash_breadcrumb(
            "InpaintRuntimeThread command received",
            command_id=request.command_id,
            action=request.action,
            page_total=total_pages,
        )
        callbacks.message(f"Starting inpaint for {total_pages} page(s).")
        callbacks.progress(0)
        callbacks.message(
            f"Using resident LaMa model on {str(model_status.get('device', '') or 'auto')} "
            f"(reload_count={int(model_status.get('reload_count', 0) or 0)})."
        )

        for index, image_relative_path in enumerate(request.image_relative_paths, start=1):
            self._check_canceled(request.cancel_token, message="Inpaint canceled before the next page.")
            page_name = Path(image_relative_path).name
            write_crash_breadcrumb(
                "InpaintRuntimeThread before page",
                command_id=request.command_id,
                page=str(image_relative_path),
                page_index=index,
                page_total=total_pages,
            )
            callbacks.event(
                {
                    "event": "batch_page_start" if total_pages > 1 else "page_start",
                    "image_relative_path": str(image_relative_path),
                    "page_index": index,
                    "page_total": total_pages,
                    "message": f"[{index}/{total_pages}] Inpainting {page_name}",
                }
            )
            try:
                write_crash_breadcrumb(
                    "InpaintRuntimeThread before run_inpaint_for_page",
                    command_id=request.command_id,
                    page=str(image_relative_path),
                )
                image_path = run_inpaint_for_page(
                    request.project,
                    image_relative_path,
                    force=request.force,
                    mask_padding=request.mask_padding,
                    use_bubble_mask=request.use_bubble_mask,
                    use_crop_windows=request.use_crop_windows,
                    device=request.device,
                    logger=callbacks.message,
                    progress_callback=callbacks.event,
                    manager=self._manager,
                    require_loaded_model=True,
                )
                write_crash_breadcrumb(
                    "InpaintRuntimeThread after run_inpaint_for_page",
                    command_id=request.command_id,
                    page=str(image_relative_path),
                )
                write_crash_breadcrumb(
                    "InpaintRuntimeThread before load metadata",
                    command_id=request.command_id,
                    page=str(image_relative_path),
                )
                metadata = load_inpaint_json(inpaint_json_path(request.project, image_relative_path))
                summary = summarize_inpaint_json(metadata)
            except InpaintRuntimeCanceledError:
                raise
            except Exception as exc:
                readable_error = f"{page_name}: {exc}"
                callbacks.message(f"Inpaint failed: {readable_error}")
                page_results.append(
                    InpaintPageResult(
                        image_relative_path=str(image_relative_path),
                        image_path=None,
                        error=str(exc),
                        summary={},
                    )
                )
                callbacks.event(
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
                    InpaintPageResult(
                        image_relative_path=str(image_relative_path),
                        image_path=image_path,
                        error=None,
                        summary=summary,
                    )
                )
            callbacks.progress(int((index / total_pages) * 100))
            self._check_canceled(request.cancel_token, message="Inpaint canceled after the current page.")

        successful_pages = [result for result in page_results if result.image_path is not None]
        if not successful_pages:
            first_error = page_results[0].error if page_results else "Unknown inpaint failure."
            raise RuntimeError(f"Inpaint failed for all pages. {first_error}")

        failed_count = len([result for result in page_results if result.error is not None])
        if failed_count:
            callbacks.message(
                f"Inpaint finished with {failed_count} failed page(s) and {len(successful_pages)} successful page(s)."
            )
        else:
            callbacks.message(f"Inpaint finished successfully for {len(successful_pages)} page(s).")

        write_crash_breadcrumb(
            "InpaintRuntimeThread command done",
            command_id=request.command_id,
            action=request.action,
            page_total=total_pages,
        )
        return InpaintWorkerResult(page_results=page_results)

    def _run_model_request(self, request: InpaintRuntimeRequest) -> LamaModelTaskResult:
        if self._manager is None:
            raise RuntimeError("Inpaint runtime is not initialized.")

        action = str(request.action or "").strip().lower()
        if action in {"load", "reload"}:
            result = load_lama_model(
                device=request.device,
                crop_trigger_size=self._crop_trigger_size,
                crop_margin=self._crop_margin,
                resize_limit=self._resize_limit,
                pad_mod=self._pad_mod,
                explicit_reload=(action == "reload"),
                logger=self._logger,
                manager=self._manager,
            )
        elif action == "unload":
            result = unload_lama_model(logger=self._logger, manager=self._manager)
        else:
            raise RuntimeError(f"Unsupported LaMa model action: {request.action}")

        return LamaModelTaskResult(
            loaded=bool(result.get("loaded", False)),
            device=str(result.get("device", "") or ""),
            message=str(result.get("message", "") or ""),
        )

    def _check_canceled(self, cancel_token: CancelToken, *, message: str) -> None:
        if cancel_token.is_cancel_requested():
            raise InpaintRuntimeCanceledError(message)

    def _startup_log(self, message: str, *, level: str = "info") -> None:
        if self._logger is not None:
            self._logger(str(message or ""))
        write_crash_breadcrumb(
            "InpaintRuntimeThread startup log",
            level=level,
            message_text=str(message or ""),
        )

    def _emit_status(self, message: str) -> None:
        if self._status_callback is not None:
            self._status_callback(str(message or ""))

    def _cleanup_runtime(self) -> None:
        manager = self._manager
        self._manager = None
        with self._lock:
            self._ready = False
        if manager is not None:
            try:
                unload_lama_model(logger=self._logger, manager=manager)
            except Exception:
                pass


__all__ = [
    "InpaintRuntimeBusyError",
    "InpaintRuntimeCallbacks",
    "InpaintRuntimeCanceledError",
    "InpaintRuntimeOutcome",
    "InpaintRuntimeRequest",
    "InpaintRuntimeThread",
]
