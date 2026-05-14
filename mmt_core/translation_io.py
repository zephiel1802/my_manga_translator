"""Disk I/O helpers for translation cache files."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Protocol, Sequence

from .image_io import ensure_path, project_relative_path
from .ocr_io import load_ocr_json, ocr_json_path
from .translation_models import TranslationConfig

TRANSLATION_SCHEMA_VERSION = 1


class ProjectLike(Protocol):
    root_dir: Path
    cache_dir: Path


def translation_json_path(project: ProjectLike, image_relative_path: Path | str) -> Path:
    """Return the canonical translation cache path for one project page."""

    relative_path = Path(image_relative_path)
    return ensure_path(project.cache_dir) / "translation" / f"{relative_path.stem}.json"


def load_translation_json(path: Path | str) -> dict[str, Any]:
    """Load and lightly validate a cached translation JSON file."""

    json_path = ensure_path(path)
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Translation cache is not valid JSON: {json_path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Translation cache root must be an object: {json_path}")
    if payload.get("stage") != "translation":
        raise ValueError(f"Unsupported translation cache stage in {json_path}")

    items = payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError(f"Translation cache field 'items' must be a list in {json_path}")

    payload.setdefault("schema_version", TRANSLATION_SCHEMA_VERSION)
    payload.setdefault("source_image", "")
    payload.setdefault("ocr_cache_path", "")
    payload.setdefault("source_language", "")
    payload.setdefault("target_language", "")
    payload.setdefault("translator", "")
    payload.setdefault("style", "")
    payload.setdefault("custom_prompt", "")
    payload.setdefault("created_at", "")
    payload.setdefault("updated_at", "")
    payload["items"] = [normalize_translation_item(item) for item in items]
    return payload


def save_translation_json(path: Path | str, data: dict[str, Any]) -> Path:
    """Persist one translation JSON payload to disk."""

    json_path = ensure_path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": int(data.get("schema_version", TRANSLATION_SCHEMA_VERSION)),
        "stage": "translation",
        "source_image": str(data.get("source_image", "")),
        "ocr_cache_path": str(data.get("ocr_cache_path", "")),
        "source_language": str(data.get("source_language", "")),
        "target_language": str(data.get("target_language", "")),
        "translator": str(data.get("translator", "")),
        "style": str(data.get("style", "")),
        "custom_prompt": str(data.get("custom_prompt", "")),
        "created_at": str(data.get("created_at", "")),
        "updated_at": str(data.get("updated_at", "")),
        "items": [normalize_translation_item(item) for item in data.get("items", [])],
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return json_path


def initialize_translation_from_ocr(
    project: ProjectLike,
    image_relative_path: Path | str,
    translator_config: TranslationConfig | dict[str, Any] | None,
    force: bool = False,
) -> Path:
    """Create or refresh translation JSON from OCR cache without loading source images."""

    config = TranslationConfig.from_value(translator_config)
    relative_path = Path(str(image_relative_path))
    ocr_path = ocr_json_path(project, relative_path)
    if not ocr_path.exists():
        raise RuntimeError(
            f"OCR cache is missing for {relative_path.name}. Prepare and run OCR first."
        )

    ocr_data = load_ocr_json(ocr_path)
    output_path = translation_json_path(project, relative_path)
    existing_data = load_translation_json(output_path) if output_path.exists() else None
    existing_items_by_ocr_id = _existing_items_by_ocr_id(existing_data)

    created_at = (
        str(existing_data.get("created_at", "")).strip()
        if isinstance(existing_data, dict)
        else ""
    ) or _timestamp()

    translation_items: list[dict[str, Any]] = []
    for item_index, ocr_item in enumerate(ocr_data.get("items", [])):
        normalized_ocr_item = dict(ocr_item) if isinstance(ocr_item, dict) else {}
        ocr_item_id = _coerce_int(normalized_ocr_item.get("id"), item_index)
        source_text = str(normalized_ocr_item.get("text", "") or "").strip()
        existing_item = existing_items_by_ocr_id.get(ocr_item_id)

        translation_item = {
            "id": item_index,
            "ocr_item_id": ocr_item_id,
            "kind": str(normalized_ocr_item.get("kind", "")),
            "bbox": normalized_ocr_item.get("bbox"),
            "ocr_bbox": normalized_ocr_item.get("ocr_bbox"),
            "source_text": source_text,
            "translated_text": "",
            "status": "pending" if source_text else "skipped",
            "error": "",
            "updated_at": "",
            "translator": "",
        }

        if existing_item is not None:
            _merge_existing_translation_item(
                translation_item,
                existing_item,
                force=force,
            )

        preserve_manual = translation_item.get("status") == "manually_edited" and not force
        if not source_text and not preserve_manual:
            translation_item["translated_text"] = ""
            translation_item["status"] = "skipped"
            translation_item["error"] = ""
            if not translation_item.get("updated_at"):
                translation_item["updated_at"] = _timestamp()

        translation_items.append(normalize_translation_item(translation_item))

    payload = {
        "schema_version": TRANSLATION_SCHEMA_VERSION,
        "stage": "translation",
        "source_image": project_relative_path(project.root_dir, project.root_dir / relative_path),
        "ocr_cache_path": project_relative_path(project.root_dir, ocr_path),
        "source_language": config.source_language,
        "target_language": config.target_language,
        "translator": config.translator,
        "style": config.style,
        "custom_prompt": config.effective_prompt(),
        "created_at": created_at,
        "updated_at": _timestamp(),
        "items": translation_items,
    }
    return save_translation_json(output_path, payload)


def summarize_translation_json(data: dict[str, Any]) -> dict[str, int]:
    """Return compact status counts for a translation page payload."""

    summary = {
        "total": 0,
        "pending": 0,
        "done": 0,
        "error": 0,
        "skipped": 0,
        "manually_edited": 0,
    }
    for item in data.get("items", []):
        if not isinstance(item, dict):
            continue
        summary["total"] += 1
        status = str(item.get("status", "pending") or "pending").strip().lower()
        if status in summary:
            summary[status] += 1
    return summary


def normalize_translation_item(item: dict[str, Any] | Any) -> dict[str, Any]:
    """Normalize one translation item with stable JSON-safe defaults."""

    if not isinstance(item, dict):
        item = {}

    normalized = {str(key): value for key, value in item.items()}
    normalized.setdefault("id", 0)
    normalized.setdefault("ocr_item_id", 0)
    normalized.setdefault("kind", "")
    normalized.setdefault("bbox", None)
    normalized.setdefault("ocr_bbox", None)
    normalized.setdefault("source_text", "")
    normalized.setdefault("translated_text", "")
    normalized.setdefault("status", "pending")
    normalized.setdefault("error", "")
    normalized.setdefault("updated_at", "")
    normalized.setdefault("translator", "")
    return normalized


def _existing_items_by_ocr_id(existing_data: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    if not isinstance(existing_data, dict):
        return {}
    items = existing_data.get("items", [])
    if not isinstance(items, list):
        return {}
    return {
        _coerce_int(item.get("ocr_item_id"), index): normalize_translation_item(item)
        for index, item in enumerate(items)
        if isinstance(item, dict)
    }


def _merge_existing_translation_item(
    new_item: dict[str, Any],
    existing_item: dict[str, Any],
    *,
    force: bool,
) -> None:
    existing_status = str(existing_item.get("status", "pending") or "pending").strip().lower()
    existing_source_text = str(existing_item.get("source_text", "") or "").strip()
    source_changed = existing_source_text != str(new_item.get("source_text", "") or "").strip()

    if force:
        return

    if existing_status == "manually_edited":
        new_item["translated_text"] = str(existing_item.get("translated_text", "") or "")
        new_item["status"] = "manually_edited"
        new_item["error"] = str(existing_item.get("error", "") or "")
        new_item["updated_at"] = str(existing_item.get("updated_at", "") or "")
        new_item["translator"] = str(existing_item.get("translator", "") or "manual_edit")
        return

    if source_changed:
        return

    preserved_status = existing_status if existing_status != "running" else "pending"
    new_item["translated_text"] = str(existing_item.get("translated_text", "") or "")
    new_item["status"] = preserved_status
    new_item["error"] = str(existing_item.get("error", "") or "")
    new_item["updated_at"] = str(existing_item.get("updated_at", "") or "")
    new_item["translator"] = str(existing_item.get("translator", "") or "")


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


__all__ = [
    "TRANSLATION_SCHEMA_VERSION",
    "initialize_translation_from_ocr",
    "load_translation_json",
    "normalize_translation_item",
    "save_translation_json",
    "summarize_translation_json",
    "translation_json_path",
]
