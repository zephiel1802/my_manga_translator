"""OCR preparation and OCR inference stages backed by on-disk cache files."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .detection_io import detection_json_path, load_detection_json
from .image_io import load_image_bgr, project_relative_path, save_png_image
from .ocr_io import (
    load_ocr_json,
    normalize_ocr_item,
    ocr_crop_dir_for_page,
    ocr_json_path,
    save_ocr_items_result,
    save_ocr_payload,
)
from .ocr_items import build_ocr_items_from_detection, crop_image_to_bbox
from .paddleocr_vl_client import PaddleOCRVLClient, PaddleOCRVLClientError


ProgressCallback = Callable[[int, int, dict[str, Any]], None]


def prepare_ocr_items_for_image(
    project,
    image_relative_path,
    force: bool = False,
    save_crops: bool = True,
    logger: Callable[[str], None] | None = None,
) -> Path:
    """Prepare OCR items and crop files from cached detection JSON."""

    relative_path = Path(str(image_relative_path))
    source_image_path = project.root_dir / relative_path
    if not source_image_path.exists():
        raise FileNotFoundError(f"Source image does not exist: {source_image_path}")

    output_json_path = ocr_json_path(project, relative_path)
    crop_dir = ocr_crop_dir_for_page(project, relative_path)

    if not force and output_json_path.exists():
        _log(logger, f"Reusing cached OCR items for {relative_path.name}")
        return output_json_path

    detection_path = detection_json_path(project, relative_path)
    if not detection_path.exists():
        raise RuntimeError(
            f"Detection cache is missing for {relative_path.name}. Run Detection first."
        )

    _log(logger, f"Loading detection cache for OCR prep: {detection_path}")
    detection_data = load_detection_json(detection_path)

    _log(logger, f"Loading source image for OCR prep: {source_image_path.name}")
    image = load_image_bgr(source_image_path)

    _log(logger, f"Preparing OCR items from detection cache: {relative_path.name}")
    items = build_ocr_items_from_detection(detection_data, image.shape, logger=logger)

    if save_crops:
        if crop_dir.exists():
            for existing_crop in crop_dir.glob("*.png"):
                existing_crop.unlink()
        crop_dir.mkdir(parents=True, exist_ok=True)

        for item in items:
            crop_bbox = item.get("ocr_bbox")
            crop_image = crop_image_to_bbox(image, crop_bbox)
            crop_path = crop_dir / f"item_{int(item['id']):03d}.png"
            save_png_image(crop_image, crop_path)
            item["crop_path"] = project_relative_path(project.root_dir, crop_path)
    else:
        for item in items:
            item["crop_path"] = None

    output_path = save_ocr_items_result(
        items,
        image_path=source_image_path,
        detection_cache_path=detection_path,
        image_shape=image.shape,
        output_path=output_json_path,
        project_root=project.root_dir,
    )
    _log(logger, f"Saved OCR cache: {output_path}")
    return output_path


def run_ocr_for_page(
    project,
    image_relative_path,
    server_url: str,
    force: bool = False,
    selected_item_ids: Sequence[int] | None = None,
    timeout: float = 120,
    logger: Callable[[str], None] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Run OCR inference for one prepared page via the persistent llama.cpp server."""

    relative_path = Path(str(image_relative_path))
    ocr_path = ocr_json_path(project, relative_path)
    if not ocr_path.exists():
        raise RuntimeError(
            f"OCR items are not prepared for {relative_path.name}. Prepare OCR items first."
        )

    payload = load_ocr_json(ocr_path)
    raw_items = payload.get("items", [])
    if not isinstance(raw_items, list):
        raise ValueError(f"OCR cache field 'items' must be a list in {ocr_path}")

    payload["items"] = [normalize_ocr_item(item) for item in raw_items]
    target_item_ids = _normalize_selected_item_ids(selected_item_ids)
    items_to_process = [
        item
        for item in payload["items"]
        if target_item_ids is None or _safe_int(item.get("id")) in target_item_ids
    ]
    if not items_to_process:
        _log(logger, f"No OCR items selected for {relative_path.name}; nothing to run.")
        return save_ocr_payload(payload, ocr_path)

    client = PaddleOCRVLClient(server_url=server_url, timeout=timeout)
    _log(logger, f"Checking PaddleOCR-VL server for {relative_path.name}: {client.server_url}")
    client.check_server()

    total_items = len(items_to_process)
    _log(logger, f"Running OCR for {relative_path.name}: {total_items} item(s)")

    processed_count = 0
    for item in items_to_process:
        item_id = _safe_int(item.get("id"))
        current_status = str(item.get("status", "") or "").strip().lower()
        current_text = str(item.get("text", "") or "").strip()

        if not force and current_status == "done" and current_text:
            processed_count += 1
            _emit_progress(
                progress_callback,
                processed_count,
                total_items,
                item_id=item_id,
                status="skipped",
                message=f"Skipped OCR item {item_id}: already recognized.",
            )
            _log(logger, f"Skipping OCR item {item_id} for {relative_path.name}: already done.")
            continue

        crop_relative_path = item.get("crop_path")
        if not crop_relative_path:
            item["status"] = "error"
            item["error"] = f"OCR crop is missing for item {item_id}."
            item["updated_at"] = _timestamp()
            item["ocr_engine"] = "paddleocr_vl_llama"
            item["server_url"] = client.server_url
            _save_ocr_progress(payload, ocr_path)
            processed_count += 1
            _emit_progress(
                progress_callback,
                processed_count,
                total_items,
                item_id=item_id,
                status="error",
                message=item["error"],
            )
            _log(logger, f"OCR item {item_id} failed: {item['error']}")
            continue

        crop_path = project.root_dir / str(crop_relative_path)
        if not crop_path.exists():
            item["status"] = "error"
            item["error"] = f"OCR crop file is missing: {crop_path}"
            item["updated_at"] = _timestamp()
            item["ocr_engine"] = "paddleocr_vl_llama"
            item["server_url"] = client.server_url
            _save_ocr_progress(payload, ocr_path)
            processed_count += 1
            _emit_progress(
                progress_callback,
                processed_count,
                total_items,
                item_id=item_id,
                status="error",
                message=item["error"],
            )
            _log(logger, f"OCR item {item_id} failed: {item['error']}")
            continue

        item["status"] = "running"
        item["error"] = ""
        item["updated_at"] = _timestamp()
        item["ocr_engine"] = "paddleocr_vl_llama"
        item["server_url"] = client.server_url
        _save_ocr_progress(payload, ocr_path)
        _emit_progress(
            progress_callback,
            processed_count,
            total_items,
            item_id=item_id,
            status="running",
            message=f"Running OCR item {item_id}",
        )

        try:
            recognized_text = client.recognize_image(crop_path)
        except (PaddleOCRVLClientError, FileNotFoundError, TimeoutError) as exc:
            item["status"] = "error"
            item["error"] = str(exc)
            item["updated_at"] = _timestamp()
            _log(logger, f"OCR item {item_id} failed: {exc}")
        except Exception as exc:
            item["status"] = "error"
            item["error"] = str(exc)
            item["updated_at"] = _timestamp()
            _log(logger, f"OCR item {item_id} failed with an unexpected error: {exc}")
        else:
            item["text"] = recognized_text
            item["status"] = "done"
            item["error"] = ""
            item["updated_at"] = _timestamp()
            _log(logger, f"OCR item {item_id} complete.")

        _save_ocr_progress(payload, ocr_path)
        processed_count += 1
        _emit_progress(
            progress_callback,
            processed_count,
            total_items,
            item_id=item_id,
            status=str(item.get("status", "")),
            message=_item_progress_message(item_id, item),
        )

    return ocr_path


def _item_progress_message(item_id: int, item: dict[str, Any]) -> str:
    status = str(item.get("status", "") or "prepared")
    error = str(item.get("error", "") or "").strip()
    if error:
        return f"OCR item {item_id}: {status} ({error})"
    return f"OCR item {item_id}: {status}"


def _normalize_selected_item_ids(selected_item_ids: Sequence[int] | None) -> set[int] | None:
    if selected_item_ids is None:
        return None

    normalized = {_safe_int(item_id) for item_id in selected_item_ids}
    return normalized if normalized else set()


def _save_ocr_progress(payload: dict[str, Any], output_path: Path) -> None:
    try:
        save_ocr_payload(payload, output_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to save OCR cache: {output_path}. {exc}") from exc


def _emit_progress(
    callback: ProgressCallback | None,
    current: int,
    total: int,
    *,
    item_id: int,
    status: str,
    message: str,
) -> None:
    if callback is None:
        return

    callback(
        current,
        total,
        {
            "item_id": item_id,
            "status": status,
            "message": message,
        },
    )


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _log(logger: Callable[[str], None] | None, message: str) -> None:
    if logger is not None:
        logger(message)


__all__ = ["prepare_ocr_items_for_image", "run_ocr_for_page"]
