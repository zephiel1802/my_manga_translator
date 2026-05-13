from __future__ import annotations

from dataclasses import replace
from statistics import median
from typing import Iterable, Sequence, TypeVar

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from .base import TextRegion


T = TypeVar("T")


def bbox_area(bbox: Sequence[int | float]) -> float:
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_intersection_area(
    a: Sequence[int | float],
    b: Sequence[int | float],
) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in a[:4]]
    bx1, by1, bx2, by2 = [float(value) for value in b[:4]]
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    return width * height


def bbox_iou(
    a: Sequence[int | float],
    b: Sequence[int | float],
) -> float:
    intersection = bbox_intersection_area(a, b)
    if intersection <= 0.0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def overlap_over_area(
    a: Sequence[int | float],
    b: Sequence[int | float],
) -> tuple[float, float]:
    intersection = bbox_intersection_area(a, b)
    area_a = bbox_area(a)
    area_b = bbox_area(b)
    return (
        float(intersection / area_a) if area_a > 0 else 0.0,
        float(intersection / area_b) if area_b > 0 else 0.0,
    )


def bbox_center(bbox: Sequence[int | float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def center_in_bbox(
    center: Sequence[int | float],
    bbox: Sequence[int | float],
    padding: int = 0,
) -> bool:
    x, y = float(center[0]), float(center[1])
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    pad = float(padding)
    return (x1 - pad) <= x <= (x2 + pad) and (y1 - pad) <= y <= (y2 + pad)


def is_near_duplicate_bbox(
    a: Sequence[int | float],
    b: Sequence[int | float],
    *,
    iou_threshold: float = 0.35,
    overlap_threshold: float = 0.80,
    containment_threshold: float = 0.70,
    strict_overlap_threshold: float = 0.90,
) -> bool:
    intersection = bbox_intersection_area(a, b)
    if intersection <= 0.0:
        return False

    overlap_a, overlap_b = overlap_over_area(a, b)
    min_area = max(min(bbox_area(a), bbox_area(b)), 1.0)
    intersection_ratio = float(intersection / min_area)
    center_a = bbox_center(a)
    center_b = bbox_center(b)

    return (
        bbox_iou(a, b) >= float(iou_threshold)
        or overlap_a >= float(strict_overlap_threshold)
        or overlap_b >= float(strict_overlap_threshold)
        or intersection_ratio >= float(overlap_threshold)
        or (
            (
                center_in_bbox(center_a, b)
                or center_in_bbox(center_b, a)
            )
            and intersection_ratio >= float(containment_threshold)
        )
    )


def _normalize_mask(mask, image_shape):
    if mask is None or np is None:
        return None

    current = mask
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "numpy"):
        current = current.numpy()

    array = np.asarray(current)
    if array.ndim == 3:
        if array.shape[0] == 1:
            array = array[0]
        elif array.shape[-1] == 1:
            array = array[..., 0]
    if array.ndim != 2:
        return None

    height = int(image_shape[0])
    width = int(image_shape[1])
    if array.shape != (height, width):
        if array.shape[0] <= 0 or array.shape[1] <= 0:
            return np.zeros((height, width), dtype=np.uint8)
        y_indices = np.clip(
            np.floor(np.arange(height) * (array.shape[0] / height)).astype(int),
            0,
            array.shape[0] - 1,
        )
        x_indices = np.clip(
            np.floor(np.arange(width) * (array.shape[1] / width)).astype(int),
            0,
            array.shape[1] - 1,
        )
        array = array[y_indices][:, x_indices]

    return np.where(array > 0, 255, 0).astype(np.uint8)


def _merge_masks(mask_a, mask_b, image_shape):
    normalized_a = _normalize_mask(mask_a, image_shape)
    normalized_b = _normalize_mask(mask_b, image_shape)
    if normalized_a is None:
        return normalized_b
    if normalized_b is None:
        return normalized_a
    return np.maximum(normalized_a, normalized_b).astype(np.uint8)


def _detector_rank(detector: str | None) -> int:
    if detector == "comic_text_detector":
        return 0
    if detector in (None, ""):
        return 1
    if detector == "pp_doclayout_v3":
        return 2
    return 1


def _prefer_region(
    existing: TextRegion,
    candidate: TextRegion,
    *,
    prefer_comic: bool,
) -> TextRegion:
    if prefer_comic and existing.detector != candidate.detector:
        existing_rank = _detector_rank(existing.detector)
        candidate_rank = _detector_rank(candidate.detector)
        if existing_rank != candidate_rank:
            return existing if existing_rank < candidate_rank else candidate

    if existing.confidence != candidate.confidence:
        return existing if existing.confidence > candidate.confidence else candidate

    existing_area = bbox_area(existing.bbox)
    candidate_area = bbox_area(candidate.bbox)
    if existing_area != candidate_area:
        return existing if existing_area < candidate_area else candidate

    existing_order = existing.reading_order if existing.reading_order is not None else 10**9
    candidate_order = candidate.reading_order if candidate.reading_order is not None else 10**9
    return existing if existing_order <= candidate_order else candidate


def _union_bbox(
    a: Sequence[int | float],
    b: Sequence[int | float],
) -> tuple[int, int, int, int]:
    return (
        int(min(float(a[0]), float(b[0]))),
        int(min(float(a[1]), float(b[1]))),
        int(max(float(a[2]), float(b[2]))),
        int(max(float(a[3]), float(b[3]))),
    )


def _merge_text_regions(
    existing: TextRegion,
    candidate: TextRegion,
    *,
    image_shape,
    prefer_comic: bool,
) -> TextRegion:
    preferred = _prefer_region(existing, candidate, prefer_comic=prefer_comic)
    alternate = candidate if preferred is existing else existing
    overlap_existing, overlap_candidate = overlap_over_area(existing.bbox, candidate.bbox)

    if (
        prefer_comic
        and preferred.detector == "comic_text_detector"
        and (overlap_existing >= 0.60 or overlap_candidate >= 0.60)
    ):
        merged_bbox = preferred.bbox
    elif existing.detector == candidate.detector == "pp_doclayout_v3":
        merged_bbox = _union_bbox(existing.bbox, candidate.bbox)
    elif overlap_existing >= 0.90 and bbox_area(existing.bbox) <= bbox_area(candidate.bbox):
        merged_bbox = existing.bbox
    elif overlap_candidate >= 0.90 and bbox_area(candidate.bbox) <= bbox_area(existing.bbox):
        merged_bbox = candidate.bbox
    else:
        merged_bbox = preferred.bbox

    merged_reading_order_candidates = [
        value
        for value in (existing.reading_order, candidate.reading_order)
        if value is not None
    ]
    merged_text = preferred.text if preferred.text else alternate.text

    return TextRegion(
        bbox=merged_bbox,
        score=max(existing.score, candidate.score),
        class_id=preferred.class_id if preferred.class_id is not None else alternate.class_id,
        mask=_merge_masks(existing.mask, candidate.mask, image_shape),
        text=merged_text,
        confidence=max(existing.confidence, candidate.confidence),
        bubble_id=existing.bubble_id if existing.bubble_id is not None else candidate.bubble_id,
        reading_order=min(merged_reading_order_candidates) if merged_reading_order_candidates else None,
        detector=preferred.detector if preferred.detector is not None else alternate.detector,
    )


def _fallback_sort_key(item, order: str):
    bbox = getattr(item, "bbox", None)
    if bbox is None and isinstance(item, dict):
        bbox = item.get("bbox")
    bbox = bbox or (0, 0, 0, 0)
    reading_order = getattr(item, "reading_order", None)
    if reading_order is None and isinstance(item, dict):
        reading_order = item.get("reading_order")
    x_key = float(bbox[0]) if order == "ltr" else -float(bbox[2])
    return (
        reading_order if reading_order is not None else 10**9,
        float(bbox[1]),
        x_key,
    )


def _largest_gap(sorted_items, axis: str):
    best_gap = 0.0
    best_index = None
    for index in range(len(sorted_items) - 1):
        current_bbox = sorted_items[index][1].bbox
        next_bbox = sorted_items[index + 1][1].bbox
        if axis == "x":
            gap = float(next_bbox[0] - current_bbox[2])
        else:
            gap = float(next_bbox[1] - current_bbox[3])
        if gap > best_gap:
            best_gap = gap
            best_index = index
    return best_gap, best_index


def sort_manga_reading_order(
    regions: Iterable[T],
    *,
    order: str = "ltr",
) -> list[T]:
    indexed_regions = list(enumerate(regions))
    if len(indexed_regions) <= 1:
        return [region for _, region in indexed_regions]

    def recurse(items):
        if len(items) <= 1:
            return items

        widths = [max(1.0, float(item[1].bbox[2] - item[1].bbox[0])) for item in items]
        heights = [max(1.0, float(item[1].bbox[3] - item[1].bbox[1])) for item in items]
        min_gap_x = max(float(median(widths)) * 0.15, 10.0)
        min_gap_y = max(float(median(heights)) * 0.10, 8.0)

        x_sorted = sorted(items, key=lambda item: (float(item[1].bbox[0]), float(item[1].bbox[1]), item[0]))
        y_sorted = sorted(items, key=lambda item: (float(item[1].bbox[1]), float(item[1].bbox[0]), item[0]))
        gap_x, split_x = _largest_gap(x_sorted, "x")
        gap_y, split_y = _largest_gap(y_sorted, "y")

        if split_y is not None and gap_y >= min_gap_y:
            top = recurse(y_sorted[: split_y + 1])
            bottom = recurse(y_sorted[split_y + 1 :])
            return top + bottom

        if split_x is not None and gap_x >= min_gap_x:
            left = recurse(x_sorted[: split_x + 1])
            right = recurse(x_sorted[split_x + 1 :])
            return left + right if order == "ltr" else right + left

        return sorted(items, key=lambda item: (*_fallback_sort_key(item[1], order), item[0]))

    sorted_items = recurse(indexed_regions)
    return [region for _, region in sorted_items]


def dedupe_text_regions_koharu_style(
    text_regions: Sequence[TextRegion],
    *,
    image_shape,
    iou_threshold: float = 0.35,
    overlap_threshold: float = 0.80,
    strict_overlap_threshold: float = 0.90,
    prefer_comic: bool = True,
) -> list[TextRegion]:
    deduped: list[TextRegion] = []
    ordered_regions = sort_manga_reading_order(text_regions, order="ltr")

    for candidate in ordered_regions:
        match_index = None
        best_score = -1.0

        for index, existing in enumerate(deduped):
            if not is_near_duplicate_bbox(
                existing.bbox,
                candidate.bbox,
                iou_threshold=iou_threshold,
                overlap_threshold=overlap_threshold,
                strict_overlap_threshold=strict_overlap_threshold,
            ):
                continue

            intersection = bbox_intersection_area(existing.bbox, candidate.bbox)
            smaller_area = max(
                min(bbox_area(existing.bbox), bbox_area(candidate.bbox)),
                1.0,
            )
            score = max(
                bbox_iou(existing.bbox, candidate.bbox),
                float(intersection / smaller_area),
            )
            if score > best_score:
                best_score = score
                match_index = index

        if match_index is None:
            deduped.append(candidate)
            continue

        deduped[match_index] = _merge_text_regions(
            deduped[match_index],
            candidate,
            image_shape=image_shape,
            prefer_comic=prefer_comic,
        )

    return sort_manga_reading_order(deduped, order="ltr")


__all__ = [
    "bbox_area",
    "bbox_center",
    "bbox_intersection_area",
    "bbox_iou",
    "center_in_bbox",
    "dedupe_text_regions_koharu_style",
    "is_near_duplicate_bbox",
    "overlap_over_area",
    "sort_manga_reading_order",
]
