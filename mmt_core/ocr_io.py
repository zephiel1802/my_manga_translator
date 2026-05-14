"""Disk I/O helpers for OCR preparation and OCR result cache files."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Protocol, Sequence

from .image_io import ensure_path, project_relative_path

OCR_SCHEMA_VERSION = 1


class ProjectLike(Protocol):
    root_dir: Path
    cache_dir: Path


def ocr_json_path(project: ProjectLike, image_relative_path: Path | str) -> Path:
    """Return the canonical OCR cache JSON path for a project page."""

    relative_path = Path(image_relative_path)
    return ensure_path(project.cache_dir) / "ocr" / f"{relative_path.stem}.json"


def ocr_crop_dir_for_page(project: ProjectLike, image_relative_path: Path | str) -> Path:
    """Return the canonical OCR crop directory for a project page."""

    relative_path = Path(image_relative_path)
    return ensure_path(project.cache_dir) / "ocr_crops" / relative_path.stem


def save_ocr_items_result(
    items: Sequence[dict[str, Any]],
    *,
    image_path: Path | str,
    detection_cache_path: Path | str,
    image_shape: Sequence[int],
    output_path: Path | str,
    project_root: Path | str,
) -> Path:
    """Serialize prepared OCR items to JSON."""

    project_root_path = ensure_path(project_root)
    payload = {
        "schema_version": OCR_SCHEMA_VERSION,
        "stage": "ocr",
        "source_image": project_relative_path(project_root_path, image_path),
        "detection_cache_path": project_relative_path(project_root_path, detection_cache_path),
        "image_width": int(image_shape[1]),
        "image_height": int(image_shape[0]),
        "items": [normalize_ocr_item(item) for item in items],
    }
    return save_ocr_payload(payload, output_path)


def save_ocr_payload(payload: dict[str, Any], output_path: Path | str) -> Path:
    """Write a complete OCR JSON payload to disk."""

    json_path = ensure_path(output_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_payload = {
        "schema_version": int(payload.get("schema_version", OCR_SCHEMA_VERSION)),
        "stage": "ocr",
        "source_image": str(payload.get("source_image", "")),
        "detection_cache_path": str(payload.get("detection_cache_path", "")),
        "image_width": int(payload.get("image_width", 0) or 0),
        "image_height": int(payload.get("image_height", 0) or 0),
        "items": [normalize_ocr_item(item) for item in payload.get("items", [])],
    }
    json_path.write_text(json.dumps(_json_safe(normalized_payload), indent=2), encoding="utf-8")
    return json_path


def load_ocr_json(path: Path | str) -> dict[str, Any]:
    """Load and lightly validate an OCR preparation JSON file."""

    json_path = ensure_path(path)
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"OCR cache is not valid JSON: {json_path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"OCR cache root must be an object: {json_path}")

    if payload.get("stage") != "ocr":
        raise ValueError(f"Unsupported OCR cache stage in {json_path}")

    items = payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError(f"OCR cache field 'items' must be a list in {json_path}")

    payload.setdefault("schema_version", OCR_SCHEMA_VERSION)
    payload.setdefault("source_image", "")
    payload.setdefault("detection_cache_path", "")
    payload.setdefault("image_width", 0)
    payload.setdefault("image_height", 0)
    payload["items"] = [normalize_ocr_item(item) for item in items]
    return payload


def normalize_ocr_item(item: dict[str, Any]) -> dict[str, Any]:
    """Return one OCR item with stable JSON-safe defaults."""

    if not isinstance(item, dict):
        item = {}

    normalized = {
        str(key): _json_safe(value)
        for key, value in dict(item).items()
    }
    normalized.setdefault("id", 0)
    normalized.setdefault("kind", "")
    normalized.setdefault("bbox", None)
    normalized.setdefault("ocr_bbox", None)
    normalized.setdefault("crop_path", None)
    normalized.setdefault("bubble_id", None)
    normalized.setdefault("reading_order", None)
    normalized.setdefault("detector_sources", [])
    normalized.setdefault("source_direction", "")
    normalized.setdefault("text", "")
    normalized.setdefault("status", "prepared")
    normalized.setdefault("error", "")
    normalized.setdefault("updated_at", "")
    normalized.setdefault("ocr_engine", "")
    normalized.setdefault("server_url", "")
    return normalized


def summarize_ocr_items(items: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Return compact counts by OCR item status."""

    summary = {
        "total": 0,
        "prepared": 0,
        "running": 0,
        "done": 0,
        "error": 0,
        "skipped": 0,
    }
    for item in items:
        summary["total"] += 1
        status = str(item.get("status", "prepared") or "prepared").strip().lower()
        if status in summary:
            summary[status] += 1
    return summary


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value

    if isinstance(value, Path):
        return value.as_posix()

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if is_dataclass(value):
        return _json_safe(asdict(value))

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return str(value)


__all__ = [
    "OCR_SCHEMA_VERSION",
    "load_ocr_json",
    "normalize_ocr_item",
    "ocr_crop_dir_for_page",
    "ocr_json_path",
    "save_ocr_payload",
    "save_ocr_items_result",
    "summarize_ocr_items",
]
