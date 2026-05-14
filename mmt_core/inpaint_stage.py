"""Inpaint stage helpers for GUI-driven mask preparation and LaMa execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
from pathlib import Path
from typing import Any, Protocol

from .detection_io import detection_json_path, load_detection_json
from .image_io import load_image_bgr, load_image_grayscale, save_png_image
from .inpaint_io import (
    build_inpaint_metadata,
    bubble_mask_path,
    inpaint_image_path,
    inpaint_json_path,
    inpaint_preview_mask_path,
    load_inpaint_json,
    save_inpaint_json,
    text_mask_path,
)
from .inpaint_masks import (
    build_bubble_guidance_mask,
    build_crop_windows_from_boxes,
    build_preview_mask,
    build_text_mask_from_ocr,
)
from .ocr_io import load_ocr_json, ocr_json_path

DEFAULT_MASK_PADDING = 8
DEFAULT_CROP_TRIGGER_SIZE = 800
DEFAULT_CROP_MARGIN = 128
DEFAULT_RESIZE_LIMIT = 1280
DEFAULT_PAD_MOD = 8

Logger = Callable[[str], None]
ProgressCallback = Callable[[dict[str, Any]], None]


class ProjectLike(Protocol):
    root_dir: Path
    cache_dir: Path


@dataclass(slots=True)
class _PreparedMaskBundle:
    source_image_path: Path
    ocr_cache_path: Path
    detection_cache_path: Path | None
    output_image_path: Path
    metadata_path: Path
    text_mask_file: Path
    bubble_mask_file: Path | None
    preview_mask_file: Path
    image_shape: tuple[int, int, int]
    item_count: int
    masked_pixel_count: int
    valid_boxes: list[tuple[int, int, int, int]]
    metadata: dict[str, Any]


class LamaInpainterManager:
    """Process-level cache for the existing LaMa Manga inpainter."""

    def __init__(self) -> None:
        self._inpainter: Any | None = None
        self._signature: tuple[str | None, str | None, int, int, int, int] | None = None
        self._loaded = False
        self._last_device: str | None = None

    def get_inpainter(
        self,
        *,
        device: str | None,
        crop_trigger_size: int = DEFAULT_CROP_TRIGGER_SIZE,
        crop_margin: int = DEFAULT_CROP_MARGIN,
        resize_limit: int = DEFAULT_RESIZE_LIMIT,
        pad_mod: int = DEFAULT_PAD_MOD,
        model_path: str | None = None,
        preload: bool = False,
    ) -> Any:
        normalized_device = _normalize_device_value(device)
        signature = (
            model_path,
            normalized_device,
            int(crop_trigger_size),
            int(crop_margin),
            int(resize_limit),
            int(pad_mod),
        )
        if self._inpainter is None or self._signature != signature:
            self.unload()
            inpainting_module = _import_inpainting()
            self._inpainter = inpainting_module.LamaMangaInpainter(
                model_path=model_path,
                device=normalized_device,
                crop_trigger_size=int(crop_trigger_size),
                crop_margin=int(crop_margin),
                resize_limit=int(resize_limit),
                pad_mod=int(pad_mod),
            )
            self._signature = signature
            self._loaded = False

        if preload and not self._loaded:
            self._inpainter.load()
            self._loaded = True
            self._last_device = str(getattr(self._inpainter, "device", normalized_device) or normalized_device or "")

        return self._inpainter

    def load(
        self,
        *,
        device: str | None,
        crop_trigger_size: int = DEFAULT_CROP_TRIGGER_SIZE,
        crop_margin: int = DEFAULT_CROP_MARGIN,
        resize_limit: int = DEFAULT_RESIZE_LIMIT,
        pad_mod: int = DEFAULT_PAD_MOD,
        model_path: str | None = None,
    ) -> dict[str, Any]:
        inpainter = self.get_inpainter(
            device=device,
            crop_trigger_size=crop_trigger_size,
            crop_margin=crop_margin,
            resize_limit=resize_limit,
            pad_mod=pad_mod,
            model_path=model_path,
            preload=True,
        )
        resolved_device = str(getattr(inpainter, "device", _normalize_device_value(device) or "") or "")
        self._last_device = resolved_device
        return {
            "loaded": True,
            "device": resolved_device,
            "message": f"LaMa Manga model is ready on {resolved_device or 'auto'}.",
        }

    def unload(self) -> dict[str, Any]:
        had_model = self._inpainter is not None
        self._inpainter = None
        self._signature = None
        self._loaded = False

        try:
            import torch
        except Exception:
            torch = None

        if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        gc.collect()
        return {
            "loaded": False,
            "device": self._last_device or "",
            "message": "LaMa Manga model cache cleared." if had_model else "LaMa Manga model was not loaded.",
        }

    def status(self) -> dict[str, Any]:
        return {
            "loaded": bool(self._inpainter is not None and self._loaded),
            "device": self._last_device or "",
            "message": (
                f"LaMa Manga model ready on {self._last_device or 'auto'}."
                if self._inpainter is not None and self._loaded
                else "LaMa Manga model is not loaded."
            ),
        }


_LAMA_MANAGER = LamaInpainterManager()


def get_lama_model_manager() -> LamaInpainterManager:
    """Return the process-level LaMa inpainter manager."""

    return _LAMA_MANAGER


def load_lama_model(
    *,
    device: str | None = None,
    crop_trigger_size: int = DEFAULT_CROP_TRIGGER_SIZE,
    crop_margin: int = DEFAULT_CROP_MARGIN,
    resize_limit: int = DEFAULT_RESIZE_LIMIT,
    pad_mod: int = DEFAULT_PAD_MOD,
    model_path: str | None = None,
    logger: Logger | None = None,
) -> dict[str, Any]:
    """Preload the cached LaMa Manga model without running inpaint yet."""

    manager = get_lama_model_manager()
    _log(logger, "Loading LaMa Manga model...")
    try:
        result = manager.load(
            device=device,
            crop_trigger_size=crop_trigger_size,
            crop_margin=crop_margin,
            resize_limit=resize_limit,
            pad_mod=pad_mod,
            model_path=model_path,
        )
    except Exception as exc:
        raise RuntimeError(_friendly_inpaint_error(exc)) from exc

    _log(logger, result["message"])
    return result


def unload_lama_model(logger: Logger | None = None) -> dict[str, Any]:
    """Release the cached LaMa Manga model and free CUDA cache when available."""

    manager = get_lama_model_manager()
    result = manager.unload()
    _log(logger, result["message"])
    return result


def prepare_inpaint_mask_for_page(
    project: ProjectLike,
    image_relative_path: str | Path,
    *,
    mask_padding: int = DEFAULT_MASK_PADDING,
    use_bubble_mask: bool = True,
    force: bool = False,
    logger: Logger | None = None,
) -> Path:
    """Build and cache the inpaint masks for one page."""

    try:
        bundle = _prepare_mask_bundle(
            project,
            image_relative_path,
            mask_padding=mask_padding,
            use_bubble_mask=use_bubble_mask,
            use_crop_windows=True,
            force=force,
            device="",
            logger=logger,
        )
    except Exception as exc:
        raise RuntimeError(_friendly_inpaint_error(exc)) from exc

    return bundle.text_mask_file


def run_inpaint_for_page(
    project: ProjectLike,
    image_relative_path: str | Path,
    *,
    force: bool = False,
    mask_padding: int = DEFAULT_MASK_PADDING,
    use_bubble_mask: bool = True,
    use_crop_windows: bool = True,
    device: str | None = None,
    logger: Logger | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Run LaMa Manga inpainting for one page using cached OCR/detection data."""

    image_relative = str(Path(image_relative_path).as_posix())
    _emit_progress(
        progress_callback,
        event="page_start",
        image_relative_path=image_relative,
        message=f"Preparing inpaint mask for {Path(image_relative).name}",
    )

    existing_output_path = inpaint_image_path(project, image_relative)
    existing_metadata_path = inpaint_json_path(project, image_relative)
    existing_metadata = _load_existing_metadata(existing_metadata_path)
    if (
        not force
        and existing_output_path.exists()
        and existing_metadata is not None
        and str(existing_metadata.get("status", "") or "").strip().lower() == "done"
    ):
        _log(logger, f"Reusing cached inpaint result: {existing_output_path}")
        _emit_progress(
            progress_callback,
            event="page_done",
            image_relative_path=image_relative,
            output_path=str(existing_output_path),
            summary={
                "item_count": int(existing_metadata.get("item_count", 0) or 0),
                "masked_pixel_count": int(existing_metadata.get("masked_pixel_count", 0) or 0),
            },
            message=f"Reused cached inpaint result for {Path(image_relative).name}",
        )
        return existing_output_path

    bundle = _prepare_mask_bundle(
        project,
        image_relative,
        mask_padding=mask_padding,
        use_bubble_mask=use_bubble_mask,
        use_crop_windows=use_crop_windows,
        force=force,
        device=device or "",
        logger=logger,
    )

    source_image = load_image_bgr(bundle.source_image_path)
    text_mask = load_image_grayscale(bundle.text_mask_file)
    bubble_mask = load_image_grayscale(bundle.bubble_mask_file) if bundle.bubble_mask_file else None
    crop_windows = build_crop_windows_from_boxes(bundle.valid_boxes, source_image.shape) if use_crop_windows else []

    metadata = dict(bundle.metadata)
    metadata["status"] = "running"
    metadata["error"] = ""
    metadata["device"] = _normalize_device_value(device) or ""
    metadata["updated_at"] = _timestamp()
    save_inpaint_json(bundle.metadata_path, metadata)

    _emit_progress(
        progress_callback,
        event="mask_ready",
        image_relative_path=image_relative,
        output_path=str(bundle.text_mask_file),
        summary={
            "item_count": bundle.item_count,
            "masked_pixel_count": bundle.masked_pixel_count,
        },
        message=f"Inpaint mask ready for {Path(image_relative).name}",
    )

    manager = get_lama_model_manager()
    try:
        inpainter = manager.get_inpainter(
            device=device,
            crop_trigger_size=DEFAULT_CROP_TRIGGER_SIZE,
            crop_margin=DEFAULT_CROP_MARGIN,
            resize_limit=DEFAULT_RESIZE_LIMIT,
            pad_mod=DEFAULT_PAD_MOD,
            preload=False,
        )
        output_image = inpainter.inpaint(
            source_image,
            text_mask,
            bubble_mask=bubble_mask,
            crop_windows=crop_windows if use_crop_windows else None,
        )
        save_png_image(output_image, bundle.output_image_path)
    except Exception as exc:
        readable_error = _friendly_inpaint_error(exc)
        metadata["status"] = "error"
        metadata["error"] = readable_error
        metadata["updated_at"] = _timestamp()
        save_inpaint_json(bundle.metadata_path, metadata)
        _emit_progress(
            progress_callback,
            event="page_error",
            image_relative_path=image_relative,
            output_path=str(bundle.output_image_path),
            summary={
                "item_count": bundle.item_count,
                "masked_pixel_count": bundle.masked_pixel_count,
            },
            message=f"Inpaint failed for {Path(image_relative).name}: {readable_error}",
        )
        raise RuntimeError(readable_error) from exc

    metadata["status"] = "done"
    metadata["error"] = ""
    metadata["device"] = str(getattr(inpainter, "device", _normalize_device_value(device) or "") or "")
    metadata["updated_at"] = _timestamp()
    save_inpaint_json(bundle.metadata_path, metadata)
    manager._loaded = True
    manager._last_device = metadata["device"]

    _log(logger, f"Inpainted page saved to {bundle.output_image_path}")
    _emit_progress(
        progress_callback,
        event="page_done",
        image_relative_path=image_relative,
        output_path=str(bundle.output_image_path),
        summary={
            "item_count": bundle.item_count,
            "masked_pixel_count": bundle.masked_pixel_count,
        },
        message=f"Inpaint complete for {Path(image_relative).name}",
    )
    return bundle.output_image_path


def run_inpaint_for_pages(
    project: ProjectLike,
    image_relative_paths: Sequence[str | Path],
    *,
    force: bool = False,
    mask_padding: int = DEFAULT_MASK_PADDING,
    use_bubble_mask: bool = True,
    use_crop_windows: bool = True,
    device: str | None = None,
    logger: Logger | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[Path]:
    """Run inpainting sequentially across multiple pages."""

    if not image_relative_paths:
        raise ValueError("No pages were provided for inpainting.")

    output_paths: list[Path] = []
    total_pages = len(image_relative_paths)
    for page_index, image_relative_path in enumerate(image_relative_paths, start=1):
        image_relative = str(Path(image_relative_path).as_posix())
        _emit_progress(
            progress_callback,
            event="batch_page_start",
            page_index=page_index,
            page_total=total_pages,
            image_relative_path=image_relative,
            message=f"[{page_index}/{total_pages}] Inpainting {Path(image_relative).name}",
        )
        output_path = run_inpaint_for_page(
            project,
            image_relative,
            force=force,
            mask_padding=mask_padding,
            use_bubble_mask=use_bubble_mask,
            use_crop_windows=use_crop_windows,
            device=device,
            logger=logger,
            progress_callback=progress_callback,
        )
        output_paths.append(output_path)

    return output_paths


def _prepare_mask_bundle(
    project: ProjectLike,
    image_relative_path: str | Path,
    *,
    mask_padding: int,
    use_bubble_mask: bool,
    use_crop_windows: bool,
    force: bool,
    device: str,
    logger: Logger | None,
) -> _PreparedMaskBundle:
    image_relative = str(Path(image_relative_path).as_posix())
    source_image_path = _project_source_path(project, image_relative)
    if not source_image_path.exists():
        raise FileNotFoundError(f"Source image is missing: {source_image_path}")

    ocr_cache_file = ocr_json_path(project, image_relative)
    if not ocr_cache_file.exists():
        raise FileNotFoundError(
            "OCR cache is missing for this page. Prepare OCR items first before inpainting."
        )

    detection_cache_file = detection_json_path(project, image_relative)
    detection_cache_path_value = detection_cache_file if detection_cache_file.exists() else None

    metadata_path = inpaint_json_path(project, image_relative)
    output_image_file = inpaint_image_path(project, image_relative)
    text_mask_file = text_mask_path(project, image_relative)
    bubble_mask_file = bubble_mask_path(project, image_relative)
    preview_mask_file = inpaint_preview_mask_path(project, image_relative)
    existing_metadata = _load_existing_metadata(metadata_path)

    if (
        not force
        and text_mask_file.exists()
        and preview_mask_file.exists()
        and existing_metadata is not None
        and str(existing_metadata.get("status", "") or "").strip().lower() in {"pending", "running", "done"}
    ):
        _log(logger, f"Reusing cached inpaint mask: {text_mask_file}")
        try:
            ocr_data = load_ocr_json(ocr_cache_file)
            source_shape = load_image_bgr(source_image_path).shape
            _, valid_boxes, _ = build_text_mask_from_ocr(
                source_shape,
                ocr_data.get("items", []),
                padding=mask_padding,
                include_skipped_items=True,
            )
        except Exception:
            valid_boxes = []
        image_shape = (
            int(existing_metadata.get("image_height", 0) or 0),
            int(existing_metadata.get("image_width", 0) or 0),
            3,
        )
        return _PreparedMaskBundle(
            source_image_path=source_image_path,
            ocr_cache_path=ocr_cache_file,
            detection_cache_path=detection_cache_path_value,
            output_image_path=output_image_file,
            metadata_path=metadata_path,
            text_mask_file=text_mask_file,
            bubble_mask_file=bubble_mask_file if bubble_mask_file.exists() else None,
            preview_mask_file=preview_mask_file,
            image_shape=image_shape,
            item_count=int(existing_metadata.get("item_count", 0) or 0),
            masked_pixel_count=int(existing_metadata.get("masked_pixel_count", 0) or 0),
            valid_boxes=valid_boxes,
            metadata=existing_metadata,
        )

    source_image = load_image_bgr(source_image_path)
    try:
        ocr_data = load_ocr_json(ocr_cache_file)
    except Exception as exc:
        raise ValueError(f"Invalid OCR cache: {exc}") from exc

    detection_data: dict[str, Any] | None = None
    if detection_cache_path_value is not None:
        try:
            detection_data = load_detection_json(detection_cache_path_value)
        except Exception as exc:
            _log(logger, f"Detection cache unavailable for bubble mask guidance: {exc}")
            detection_data = None

    text_mask, valid_boxes, masked_pixel_count = build_text_mask_from_ocr(
        source_image.shape,
        ocr_data.get("items", []),
        padding=mask_padding,
        include_skipped_items=True,
    )
    if not valid_boxes or masked_pixel_count <= 0:
        error_message = "No valid OCR bounding boxes were available to build an inpaint mask."
        error_metadata = build_inpaint_metadata(
            project_root=project.root_dir,
            image_relative_path=image_relative,
            ocr_cache_path=ocr_cache_file,
            detection_cache_path=detection_cache_path_value,
            output_image_path_value=output_image_file,
            text_mask_path_value=text_mask_file,
            bubble_mask_path_value=bubble_mask_file if use_bubble_mask else None,
            image_shape=source_image.shape,
            item_count=0,
            masked_pixel_count=0,
            device=device,
            status="error",
            error=error_message,
            created_at=(existing_metadata or {}).get("created_at"),
            updated_at=_timestamp(),
            settings=_settings_payload(
                mask_padding=mask_padding,
                use_bubble_mask=use_bubble_mask,
                use_crop_windows=use_crop_windows,
            ),
        )
        save_inpaint_json(metadata_path, error_metadata)
        raise ValueError(error_message)

    bubble_guidance = None
    saved_bubble_mask_file: Path | None = None
    if use_bubble_mask and detection_data is not None:
        bubble_guidance = build_bubble_guidance_mask(
            source_image.shape,
            detection_data,
            project_root=project.root_dir,
        )
        if bubble_guidance is not None:
            save_png_image(bubble_guidance, bubble_mask_file)
            saved_bubble_mask_file = bubble_mask_file
        elif bubble_mask_file.exists():
            bubble_mask_file.unlink()
    elif bubble_mask_file.exists():
        bubble_mask_file.unlink()

    save_png_image(text_mask, text_mask_file)
    save_png_image(build_preview_mask(text_mask, bubble_guidance), preview_mask_file)

    metadata = build_inpaint_metadata(
        project_root=project.root_dir,
        image_relative_path=image_relative,
        ocr_cache_path=ocr_cache_file,
        detection_cache_path=detection_cache_path_value,
        output_image_path_value=output_image_file,
        text_mask_path_value=text_mask_file,
        bubble_mask_path_value=saved_bubble_mask_file,
        image_shape=source_image.shape,
        item_count=len(valid_boxes),
        masked_pixel_count=masked_pixel_count,
        device=device,
        status="pending",
        error="",
        created_at=(existing_metadata or {}).get("created_at"),
        updated_at=_timestamp(),
        settings=_settings_payload(
            mask_padding=mask_padding,
            use_bubble_mask=use_bubble_mask,
            use_crop_windows=use_crop_windows,
        ),
    )
    save_inpaint_json(metadata_path, metadata)
    _log(logger, f"Prepared inpaint mask for {Path(image_relative).name}: {text_mask_file}")

    return _PreparedMaskBundle(
        source_image_path=source_image_path,
        ocr_cache_path=ocr_cache_file,
        detection_cache_path=detection_cache_path_value,
        output_image_path=output_image_file,
        metadata_path=metadata_path,
        text_mask_file=text_mask_file,
        bubble_mask_file=saved_bubble_mask_file,
        preview_mask_file=preview_mask_file,
        image_shape=tuple(int(value) for value in source_image.shape),
        item_count=len(valid_boxes),
        masked_pixel_count=masked_pixel_count,
        valid_boxes=valid_boxes,
        metadata=metadata,
    )


def _project_source_path(project: ProjectLike, image_relative_path: str) -> Path:
    return project.root_dir / Path(image_relative_path)


def _import_inpainting():
    try:
        import inpainting
    except Exception as exc:
        raise RuntimeError(
            "LaMa Manga dependencies are not available. Install the inpainting runtime dependencies first."
        ) from exc
    return inpainting


def _settings_payload(
    *,
    mask_padding: int,
    use_bubble_mask: bool,
    use_crop_windows: bool,
) -> dict[str, Any]:
    return {
        "mask_padding": int(mask_padding),
        "use_bubble_mask": bool(use_bubble_mask),
        "use_crop_windows": bool(use_crop_windows),
        "crop_trigger_size": DEFAULT_CROP_TRIGGER_SIZE,
        "crop_margin": DEFAULT_CROP_MARGIN,
        "resize_limit": DEFAULT_RESIZE_LIMIT,
    }


def _normalize_device_value(device: str | None) -> str | None:
    if device is None:
        return None
    normalized = str(device).strip()
    if not normalized or normalized.lower() == "auto":
        return None
    return normalized


def _friendly_inpaint_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    lower_message = message.lower()
    if "no valid ocr bounding boxes" in lower_message:
        return "No valid OCR bounding boxes were available to build an inpaint mask."
    if "ocr cache is missing" in lower_message:
        return message
    if "source image is missing" in lower_message:
        return message
    if "cuda out of memory" in lower_message or "out of memory" in lower_message:
        return "LaMa Manga ran out of GPU memory. Try CPU mode or process fewer pages."
    if "huggingface_hub" in lower_message:
        return "LaMa Manga weights could not be downloaded. Check huggingface_hub and network access."
    if "safetensors" in lower_message or "torch" in lower_message or "numpy" in lower_message:
        return f"LaMa Manga dependencies are missing or failed to load. {message}"
    return message


def _emit_progress(callback: ProgressCallback | None, **payload: Any) -> None:
    if callback is None:
        return
    callback(payload)


def _log(logger: Logger | None, message: str) -> None:
    if logger is not None and message:
        logger(str(message))


def _load_existing_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return load_inpaint_json(path)
    except Exception:
        return None


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


__all__ = [
    "DEFAULT_CROP_MARGIN",
    "DEFAULT_CROP_TRIGGER_SIZE",
    "DEFAULT_MASK_PADDING",
    "DEFAULT_PAD_MOD",
    "DEFAULT_RESIZE_LIMIT",
    "LamaInpainterManager",
    "get_lama_model_manager",
    "load_lama_model",
    "prepare_inpaint_mask_for_page",
    "run_inpaint_for_page",
    "run_inpaint_for_pages",
    "unload_lama_model",
]
