"""Sanitize raw detection text regions against bubble ownership."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any

from .ocr_items import bbox_intersection_area, bbox_to_list, clamp_bbox_to_image, intersect_bboxes

Logger = Callable[[str], None] | None


def sanitize_detection_payload(
    detection_data: dict[str, Any],
    image_shape: Sequence[int],
    *,
    logger: Logger = None,
) -> dict[str, Any]:
    """Return detection data with bubble-owned text regions clipped to bubble bounds."""

    if not isinstance(detection_data, dict):
        return detection_data

    sanitized = deepcopy(detection_data)
    bubbles = [dict(item) for item in sanitized.get("bubbles", []) if isinstance(item, dict)]
    text_regions = [dict(item) for item in sanitized.get("text_regions", []) if isinstance(item, dict)]

    bubble_records = _bubble_records(bubbles, image_shape)
    if not bubble_records or not text_regions:
        sanitized["bubbles"] = bubbles
        sanitized["text_regions"] = text_regions
        return sanitized

    sanitized_text_regions: list[dict[str, Any]] = []
    next_region_id = 0
    for fallback_index, region in enumerate(text_regions):
        original_region_id = _coerce_int(region.get("id"), fallback_index)
        text_bbox = clamp_bbox_to_image(region.get("bbox"), image_shape)
        if text_bbox is None:
            continue

        owners = _owners_for_text_region(text_bbox, bubble_records, preferred_bubble_id=region.get("bubble_id"))
        if not owners:
            outside_region = dict(region)
            outside_region["id"] = next_region_id
            next_region_id += 1
            if region.get("bubble_id") not in (None, ""):
                outside_region["original_bbox"] = bbox_to_list(text_bbox)
                outside_region["sanitized"] = True
                outside_region["sanitization_reason"] = "cleared_invalid_bubble_owner"
                outside_region["bubble_id"] = None
                _log(
                    logger,
                    f"[detection_sanitizer] Cleared bubble ownership for text region {original_region_id}; kept as outside text.",
                )
            sanitized_text_regions.append(outside_region)
            continue

        if len(owners) == 1:
            owner = owners[0]
            sanitized_region = _sanitize_text_region_for_owner(
                region,
                text_bbox=text_bbox,
                owner=owner,
                image_shape=image_shape,
                region_id=next_region_id,
                logger=logger,
                original_region_id=original_region_id,
            )
            if sanitized_region is not None:
                sanitized_text_regions.append(sanitized_region)
                next_region_id += 1
            continue

        split_regions = _split_text_region_across_bubbles(
            region,
            text_bbox=text_bbox,
            owners=owners,
            image_shape=image_shape,
            start_region_id=next_region_id,
            logger=logger,
            original_region_id=original_region_id,
        )
        sanitized_text_regions.extend(split_regions)
        next_region_id += len(split_regions)

    sanitized["bubbles"] = bubbles
    sanitized["text_regions"] = sanitized_text_regions
    return sanitized


def _bubble_records(
    bubbles: Sequence[dict[str, Any]],
    image_shape: Sequence[int],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for fallback_index, bubble in enumerate(bubbles):
        bubble_bbox = clamp_bbox_to_image(bubble.get("bbox"), image_shape)
        if bubble_bbox is None:
            continue
        records.append(
            {
                "id": _coerce_int(bubble.get("id"), fallback_index),
                "bbox": bubble_bbox,
            }
        )
    return records


def _owners_for_text_region(
    text_bbox: tuple[int, int, int, int],
    bubble_records: Sequence[dict[str, Any]],
    *,
    preferred_bubble_id: Any = None,
) -> list[dict[str, Any]]:
    matching_records = [
        record
        for record in bubble_records
        if _text_region_belongs_to_bubble(text_bbox, record["bbox"])
    ]
    if len(matching_records) <= 1:
        return matching_records

    preferred_id = _coerce_optional_int(preferred_bubble_id)
    if preferred_id is not None:
        matching_records = sorted(
            matching_records,
            key=lambda record: (0 if int(record["id"]) == preferred_id else 1, int(record["id"])),
        )
    else:
        matching_records = sorted(
            matching_records,
            key=lambda record: (
                -_bubble_overlap_ratio(text_bbox, record["bbox"]),
                int(record["id"]),
            ),
        )
    return matching_records


def _sanitize_text_region_for_owner(
    region: dict[str, Any],
    *,
    text_bbox: tuple[int, int, int, int],
    owner: dict[str, Any],
    image_shape: Sequence[int],
    region_id: int,
    logger: Logger,
    original_region_id: int,
) -> dict[str, Any] | None:
    clipped_bbox = intersect_bboxes(text_bbox, owner["bbox"], image_shape)
    if clipped_bbox is None:
        _log(
            logger,
            f"[detection_sanitizer] Dropped invalid clipped text region {original_region_id} for bubble {owner['id']}.",
        )
        return None

    sanitized_region = dict(region)
    sanitized_region["id"] = region_id
    original_bbox = bbox_to_list(text_bbox)
    original_bubble_id = _coerce_optional_int(region.get("bubble_id"))
    bbox_changed = list(clipped_bbox) != original_bbox
    owner_changed = original_bubble_id != int(owner["id"])

    sanitized_region["bbox"] = bbox_to_list(clipped_bbox)
    sanitized_region["bubble_id"] = int(owner["id"])
    if bbox_changed or owner_changed:
        sanitized_region["original_bbox"] = original_bbox
        sanitized_region["sanitized"] = True
        sanitized_region["sanitization_reason"] = (
            "clipped_to_bubble_bbox" if bbox_changed else "assigned_to_bubble_bbox"
        )
        _log(
            logger,
            f"[detection_sanitizer] Clipped text region {original_region_id} to bubble {owner['id']}.",
        )
    return sanitized_region


def _split_text_region_across_bubbles(
    region: dict[str, Any],
    *,
    text_bbox: tuple[int, int, int, int],
    owners: Sequence[dict[str, Any]],
    image_shape: Sequence[int],
    start_region_id: int,
    logger: Logger,
    original_region_id: int,
) -> list[dict[str, Any]]:
    split_regions: list[dict[str, Any]] = []
    for offset, owner in enumerate(owners):
        clipped_bbox = intersect_bboxes(text_bbox, owner["bbox"], image_shape)
        if clipped_bbox is None:
            continue
        split_region = dict(region)
        split_region["id"] = start_region_id + offset
        split_region["bbox"] = bbox_to_list(clipped_bbox)
        split_region["bubble_id"] = int(owner["id"])
        split_region["original_bbox"] = bbox_to_list(text_bbox)
        split_region["split_from_text_region_id"] = original_region_id
        split_region["sanitized"] = True
        split_region["sanitization_reason"] = "split_across_bubbles"
        split_regions.append(split_region)

    if split_regions:
        _log(
            logger,
            f"[detection_sanitizer] Split text region {original_region_id} into {len(split_regions)} bubble-owned regions.",
        )
    else:
        _log(
            logger,
            f"[detection_sanitizer] Dropped invalid split text region {original_region_id}.",
        )
    return split_regions


def _text_region_belongs_to_bubble(
    text_bbox: tuple[int, int, int, int],
    bubble_bbox: tuple[int, int, int, int],
) -> bool:
    center_x = (float(text_bbox[0]) + float(text_bbox[2])) / 2.0
    center_y = (float(text_bbox[1]) + float(text_bbox[3])) / 2.0
    if (
        float(bubble_bbox[0]) <= center_x <= float(bubble_bbox[2])
        and float(bubble_bbox[1]) <= center_y <= float(bubble_bbox[3])
    ):
        return True

    text_area = max(1, _bbox_area(text_bbox))
    return (bbox_intersection_area(text_bbox, bubble_bbox) / float(text_area)) >= 0.50


def _bubble_overlap_ratio(
    text_bbox: tuple[int, int, int, int],
    bubble_bbox: tuple[int, int, int, int],
) -> float:
    text_area = max(1, _bbox_area(text_bbox))
    return bbox_intersection_area(text_bbox, bubble_bbox) / float(text_area)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    return max(0, int(bbox[2]) - int(bbox[0])) * max(0, int(bbox[3]) - int(bbox[1]))


def _coerce_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _coerce_int(value: Any, default: int) -> int:
    resolved = _coerce_optional_int(value)
    if resolved is None:
        return int(default)
    return resolved


def _log(logger: Logger, message: str) -> None:
    if logger is not None:
        logger(message)


__all__ = ["sanitize_detection_payload"]
