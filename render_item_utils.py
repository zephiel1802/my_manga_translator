from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import replace
from typing import Callable, Sequence

from detectors.base import BubbleRegion, TextRegion
from detectors.matching import point_in_mask
from detectors.runtime_utils import clamp_bbox_to_image
from detectors.selection import (
    bbox_area,
    bbox_center,
    bbox_intersection_area,
    bbox_iou,
    center_in_bbox,
    dedupe_text_regions_koharu_style,
    overlap_over_area,
    sort_manga_reading_order,
)


def _log(logger: Callable[[str], None] | None, message: str) -> None:
    if logger is not None:
        logger(message)


def _bbox_union(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return (
        min(int(a[0]), int(b[0])),
        min(int(a[1]), int(b[1])),
        max(int(a[2]), int(b[2])),
        max(int(a[3]), int(b[3])),
    )


def _dedupe_bboxes(boxes, image_shape):
    deduped = []
    seen = set()
    for box in boxes:
        if box is None:
            continue
        normalized = clamp_bbox_to_image(box, image_shape)
        if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _collect_text_regions(item) -> list[TextRegion]:
    text_regions = list(item.get("text_regions") or [])
    if (
        item.get("kind") == "outside_text"
        and item.get("text_region") is not None
        and not any(region is item.get("text_region") for region in text_regions)
    ):
        text_regions.append(item["text_region"])
    return text_regions


def _combine_text_regions(regions, image_shape):
    if not regions:
        return []
    return dedupe_text_regions_koharu_style(
        list(regions),
        image_shape=image_shape,
        prefer_comic=True,
    )


def _sort_container_items(items):
    return [
        item
        for _, item in sorted(
            enumerate(items),
            key=lambda entry: (
                entry[1].get("reading_order") if entry[1].get("reading_order") is not None else 10**9,
                (entry[1].get("container_bbox") or entry[1].get("render_bbox") or entry[1].get("ocr_bbox") or (0, 0, 0, 0))[1],
                (entry[1].get("container_bbox") or entry[1].get("render_bbox") or entry[1].get("ocr_bbox") or (0, 0, 0, 0))[0],
                entry[0],
            ),
        )
    ]


def _width_height(bbox):
    return max(0, int(bbox[2]) - int(bbox[0])), max(0, int(bbox[3]) - int(bbox[1]))


def _is_tiny_bbox(bbox, *, min_width: int, min_height: int, min_area: int) -> bool:
    width, height = _width_height(bbox)
    return width < min_width or height < min_height or bbox_area(bbox) < int(min_area)


def _is_huge_bbox(bbox, image_shape, *, max_region_ratio: float = 0.35) -> bool:
    page_area = max(1, int(image_shape[0]) * int(image_shape[1]))
    return bbox_area(bbox) > page_area * float(max_region_ratio)


def _region_inside_bubble(
    region: TextRegion,
    bubble: BubbleRegion,
    *,
    min_overlap_ratio: float = 0.50,
    padding: int = 4,
) -> bool:
    region_bbox = region.bbox
    region_area = max(bbox_area(region_bbox), 1.0)
    region_center = bbox_center(region_bbox)

    if bubble.mask is not None:
        point = (int(round(region_center[0])), int(round(region_center[1])))
        if point_in_mask(point, bubble.mask):
            return True

    if center_in_bbox(region_center, bubble.bbox, padding=padding):
        return True

    overlap = bbox_intersection_area(region_bbox, bubble.bbox) / region_area
    return overlap >= float(min_overlap_ratio)


def _region_inside_any_bubble(
    region: TextRegion,
    bubbles: Sequence[BubbleRegion],
    *,
    min_overlap_ratio: float = 0.50,
    padding: int = 4,
) -> bool:
    return any(
        _region_inside_bubble(
            region,
            bubble,
            min_overlap_ratio=min_overlap_ratio,
            padding=padding,
        )
        for bubble in bubbles
    )


def _dedupe_pp_blocks(pp_text_regions, image_shape):
    deduped: list[TextRegion] = []
    for region in sort_manga_reading_order(pp_text_regions, order="ltr"):
        match_index = None
        for index, existing in enumerate(deduped):
            overlap_existing, overlap_region = overlap_over_area(existing.bbox, region.bbox)
            if (
                overlap_existing >= 0.90
                or overlap_region >= 0.90
                or bbox_iou(existing.bbox, region.bbox) >= 0.70
            ):
                match_index = index
                break

        if match_index is None:
            deduped.append(region)
            continue

        existing = deduped[match_index]
        existing_area = bbox_area(existing.bbox)
        region_area = bbox_area(region.bbox)
        if region.confidence > existing.confidence:
            preferred = region
        elif region.confidence < existing.confidence:
            preferred = existing
        else:
            preferred = region if region_area < existing_area else existing
        deduped[match_index] = replace(
            preferred,
            reading_order=min(
                value
                for value in (existing.reading_order, region.reading_order)
                if value is not None
            )
            if (existing.reading_order is not None or region.reading_order is not None)
            else preferred.reading_order,
        )

    return sort_manga_reading_order(deduped, order="ltr")


def _choose_outside_ocr_bbox(
    container_bbox,
    evidence_regions: Sequence[TextRegion],
    image_shape,
):
    if not evidence_regions:
        return clamp_bbox_to_image(container_bbox, image_shape)

    x1 = min(region.bbox[0] for region in evidence_regions)
    y1 = min(region.bbox[1] for region in evidence_regions)
    x2 = max(region.bbox[2] for region in evidence_regions)
    y2 = max(region.bbox[3] for region in evidence_regions)
    evidence_bbox = clamp_bbox_to_image((x1, y1, x2, y2), image_shape)
    container_bbox = clamp_bbox_to_image(container_bbox, image_shape)

    container_area = max(bbox_area(container_bbox), 1.0)
    evidence_area = bbox_area(evidence_bbox)
    overlap_container, overlap_evidence = overlap_over_area(container_bbox, evidence_bbox)
    if (
        evidence_area >= max(120.0, container_area * 0.18)
        and evidence_area <= (container_area * 1.10)
        and overlap_evidence >= 0.65
        and overlap_container >= 0.15
    ):
        return evidence_bbox
    return container_bbox


def _choose_primary_block(existing, candidate):
    if existing.get("outside_source") != candidate.get("outside_source"):
        if existing.get("outside_source") == "pp":
            return existing
        if candidate.get("outside_source") == "pp":
            return candidate

    existing_conf = float(getattr(existing.get("text_region"), "confidence", 0.0) or 0.0)
    candidate_conf = float(getattr(candidate.get("text_region"), "confidence", 0.0) or 0.0)
    if existing_conf != candidate_conf:
        return existing if existing_conf >= candidate_conf else candidate

    existing_area = bbox_area(existing.get("container_bbox") or existing.get("render_bbox") or (0, 0, 0, 0))
    candidate_area = bbox_area(candidate.get("container_bbox") or candidate.get("render_bbox") or (0, 0, 0, 0))
    return existing if existing_area <= candidate_area else candidate


def build_outside_text_blocks(
    pp_text_regions,
    comic_text_regions,
    bubbles,
    image_shape,
    logger: Callable[[str], None] | None = None,
) -> tuple[list[dict], dict]:
    ordered_pp = sort_manga_reading_order(pp_text_regions, order="ltr")
    accepted_pp: list[TextRegion] = []
    skipped_huge_pp: list[TextRegion] = []

    for region in ordered_pp:
        bbox = clamp_bbox_to_image(region.bbox, image_shape)
        region = replace(region, bbox=bbox, detector="pp_doclayout_v3")
        if _is_tiny_bbox(bbox, min_width=6, min_height=6, min_area=48):
            continue
        if _region_inside_any_bubble(region, bubbles, min_overlap_ratio=0.55, padding=2):
            continue
        if _is_huge_bbox(bbox, image_shape, max_region_ratio=0.35):
            skipped_huge_pp.append(region)
            _log(logger, f"Skipped huge PP outside block {bbox}")
            continue
        accepted_pp.append(region)

    pp_blocks = _dedupe_pp_blocks(accepted_pp, image_shape)
    if not pp_blocks and skipped_huge_pp:
        smallest = min(skipped_huge_pp, key=lambda region: bbox_area(region.bbox))
        pp_blocks = [smallest]

    unmatched_comics: list[TextRegion] = []
    for comic_region in comic_text_regions:
        bbox = clamp_bbox_to_image(comic_region.bbox, image_shape)
        comic_region = replace(comic_region, bbox=bbox, detector="comic_text_detector")
        if _region_inside_any_bubble(comic_region, bubbles, min_overlap_ratio=0.50, padding=4):
            continue
        unmatched_comics.append(comic_region)

    blocks: list[dict] = []
    used_comic_indices: set[int] = set()

    for pp_region in pp_blocks:
        attached_comics = []
        for index, comic_region in enumerate(unmatched_comics):
            if index in used_comic_indices:
                continue
            overlap_pp, overlap_comic = overlap_over_area(pp_region.bbox, comic_region.bbox)
            if overlap_comic >= 0.50 or center_in_bbox(bbox_center(comic_region.bbox), pp_region.bbox, padding=2):
                attached_comics.append(comic_region)
                used_comic_indices.add(index)

        evidence_regions = _combine_text_regions(attached_comics, image_shape) if attached_comics else [pp_region]
        ocr_bbox = _choose_outside_ocr_bbox(pp_region.bbox, evidence_regions, image_shape)
        source_direction = (
            "vertical"
            if (pp_region.bbox[3] - pp_region.bbox[1]) >= ((pp_region.bbox[2] - pp_region.bbox[0]) * 1.15)
            else "horizontal"
        )
        blocks.append(
            {
                "outside_source": "pp",
                "container_bbox": pp_region.bbox,
                "text_region": pp_region,
                "text_regions": evidence_regions,
                "ocr_bbox": ocr_bbox,
                "render_bbox": pp_region.bbox,
                "inpaint_bbox": pp_region.bbox,
                "reading_order": pp_region.reading_order,
                "source_direction": source_direction,
            }
        )

    pp_container_bboxes = [block["container_bbox"] for block in blocks]
    fallback_candidates: list[TextRegion] = []
    for index, comic_region in enumerate(unmatched_comics):
        if index in used_comic_indices:
            continue
        bbox = comic_region.bbox
        if comic_region.confidence < 0.55:
            continue
        if _is_tiny_bbox(bbox, min_width=12, min_height=8, min_area=120):
            continue
        if any(
            overlap_over_area(existing_bbox, bbox)[1] >= 0.50
            or center_in_bbox(bbox_center(bbox), existing_bbox, padding=2)
            for existing_bbox in pp_container_bboxes
        ):
            continue
        fallback_candidates.append(comic_region)

    fallback_candidates = dedupe_text_regions_koharu_style(
        fallback_candidates,
        image_shape=image_shape,
        iou_threshold=0.35,
        overlap_threshold=0.80,
        strict_overlap_threshold=0.90,
        prefer_comic=True,
    )

    filtered_fallback_candidates: list[TextRegion] = []
    for candidate in fallback_candidates:
        contained = False
        for other in fallback_candidates:
            if other is candidate:
                continue
            overlap_candidate, _ = overlap_over_area(candidate.bbox, other.bbox)
            if overlap_candidate >= 0.90 and bbox_area(candidate.bbox) <= bbox_area(other.bbox):
                contained = True
                break
        if not contained:
            filtered_fallback_candidates.append(candidate)

    for comic_region in sort_manga_reading_order(filtered_fallback_candidates, order="ltr"):
        source_direction = (
            "vertical"
            if (comic_region.bbox[3] - comic_region.bbox[1]) >= ((comic_region.bbox[2] - comic_region.bbox[0]) * 1.15)
            else "horizontal"
        )
        blocks.append(
            {
                "outside_source": "comic_fallback",
                "container_bbox": comic_region.bbox,
                "text_region": comic_region,
                "text_regions": [comic_region],
                "ocr_bbox": comic_region.bbox,
                "render_bbox": comic_region.bbox,
                "inpaint_bbox": comic_region.bbox,
                "reading_order": comic_region.reading_order,
                "source_direction": source_direction,
            }
        )

    return (
        _sort_container_items(blocks),
        {
            "pp_outside_blocks": sum(1 for block in blocks if block.get("outside_source") == "pp"),
            "comic_fallback_outside_blocks": sum(1 for block in blocks if block.get("outside_source") == "comic_fallback"),
        },
    )


def _attach_outside_to_bubble(bubble_item, outside_item, image_shape):
    combined_regions = _combine_text_regions(
        _collect_text_regions(bubble_item) + _collect_text_regions(outside_item),
        image_shape,
    )
    bubble_item["text_regions"] = combined_regions
    bubble_item["reading_order"] = min(
        [
            value
            for value in (
                bubble_item.get("reading_order"),
                outside_item.get("reading_order"),
            )
            if value is not None
        ],
        default=bubble_item.get("reading_order"),
    )
    bubble_item["render_bbox"] = _bbox_union(
        bubble_item.get("render_bbox"),
        outside_item.get("render_bbox"),
    )
    bubble_item["ocr_bbox"] = _bbox_union(
        bubble_item.get("ocr_bbox"),
        outside_item.get("ocr_bbox"),
    )
    bubble_item["inpaint_bbox"] = _bbox_union(
        bubble_item.get("inpaint_bbox"),
        outside_item.get("inpaint_bbox"),
    )
    bubble_item["inpaint_bboxes"] = _dedupe_bboxes(
        list(bubble_item.get("inpaint_bboxes") or [])
        + list(outside_item.get("inpaint_bboxes") or [])
        + [bubble_item.get("render_bbox"), bubble_item.get("ocr_bbox")],
        image_shape,
    )
    bubble_item["inpaint_fallback_used"] = False
    bubble_item["fallback_reason"] = None
    return bubble_item


def _should_attach_outside_to_bubble(outside_item, bubble_item):
    outside_bbox = outside_item.get("container_bbox") or outside_item.get("render_bbox") or outside_item.get("ocr_bbox")
    bubble_target = bubble_item.get("render_bbox") or bubble_item.get("ocr_bbox") or bubble_item.get("coords")
    bubble_regions = _collect_text_regions(bubble_item)
    if bubble_regions:
        bubble_target = (
            min(region.bbox[0] for region in bubble_regions),
            min(region.bbox[1] for region in bubble_regions),
            max(region.bbox[2] for region in bubble_regions),
            max(region.bbox[3] for region in bubble_regions),
        )

    if outside_bbox is None or bubble_target is None:
        return False

    overlap_outside, overlap_bubble = overlap_over_area(outside_bbox, bubble_target)
    return (
        overlap_outside >= 0.80
        or (center_in_bbox(bbox_center(outside_bbox), bubble_target, padding=2) and overlap_outside >= 0.55)
        or (overlap_outside >= 0.65 and overlap_bubble >= 0.45)
    )


def consolidate_render_items(render_items, image_shape, logger: Callable[[str], None] | None = None):
    copied_items = [deepcopy(item) for item in render_items]
    bubble_items = [item for item in copied_items if item.get("kind") == "bubble"]
    outside_items = [item for item in copied_items if item.get("kind") == "outside_text"]

    filtered_outside: list[dict] = []
    for outside_item in outside_items:
        if _is_tiny_bbox(
            outside_item.get("container_bbox") or outside_item.get("render_bbox") or outside_item.get("ocr_bbox") or (0, 0, 0, 0),
            min_width=8,
            min_height=6,
            min_area=64,
        ):
            _log(logger, f"Dropped tiny outside_text item {outside_item.get('render_bbox')}")
            continue

        attached = False
        for bubble_item in bubble_items:
            if _should_attach_outside_to_bubble(outside_item, bubble_item):
                _attach_outside_to_bubble(bubble_item, outside_item, image_shape)
                attached = True
                _log(
                    logger,
                    f"Attached outside_text {outside_item.get('render_bbox')} to bubble {bubble_item.get('coords')}",
                )
                break
        if not attached:
            filtered_outside.append(outside_item)

    deduped_outside: list[dict] = []
    for candidate in _sort_container_items(filtered_outside):
        match_index = None
        for index, existing in enumerate(deduped_outside):
            overlap_existing, overlap_candidate = overlap_over_area(
                existing.get("container_bbox") or existing.get("render_bbox"),
                candidate.get("container_bbox") or candidate.get("render_bbox"),
            )
            if overlap_existing >= 0.90 or overlap_candidate >= 0.90:
                match_index = index
                break

        if match_index is None:
            deduped_outside.append(candidate)
            continue

        existing = deduped_outside[match_index]
        primary = _choose_primary_block(existing, candidate)
        secondary = candidate if primary is existing else existing
        primary["text_regions"] = _combine_text_regions(
            _collect_text_regions(primary) + _collect_text_regions(secondary),
            image_shape,
        )
        primary["ocr_bbox"] = _choose_outside_ocr_bbox(
            primary.get("container_bbox") or primary.get("render_bbox"),
            primary["text_regions"],
            image_shape,
        )
        primary["reading_order"] = min(
            [
                value
                for value in (
                    primary.get("reading_order"),
                    secondary.get("reading_order"),
                )
                if value is not None
            ],
            default=primary.get("reading_order"),
        )
        primary["inpaint_bboxes"] = _dedupe_bboxes(
            list(primary.get("inpaint_bboxes") or [])
            + list(secondary.get("inpaint_bboxes") or [])
            + [primary.get("container_bbox"), primary.get("ocr_bbox")],
            image_shape,
        )
        deduped_outside[match_index] = primary
        _log(
            logger,
            f"Dropped duplicate outside_text {candidate.get('render_bbox')} into {primary.get('render_bbox')}",
        )

    combined_items = bubble_items + deduped_outside
    for item in combined_items:
        if item.get("text_regions"):
            item["text_regions"] = sort_manga_reading_order(item["text_regions"], order="ltr")

    return [
        item
        for _, item in sorted(
            enumerate(combined_items),
            key=lambda entry: (
                entry[1].get("reading_order") if entry[1].get("reading_order") is not None else 10**9,
                (entry[1].get("render_bbox") or entry[1].get("coords") or (0, 0, 0, 0))[1],
                (entry[1].get("render_bbox") or entry[1].get("coords") or (0, 0, 0, 0))[0],
                entry[0],
            ),
        )
    ]


def _normalize_ocr_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", (text or "").upper(), flags=re.UNICODE)


def _ocr_similarity(existing_text: str, candidate_text: str) -> bool:
    existing_normalized = _normalize_ocr_text(existing_text)
    candidate_normalized = _normalize_ocr_text(candidate_text)
    if not existing_normalized or not candidate_normalized:
        return False
    if existing_normalized == candidate_normalized:
        return True

    shorter, longer = sorted(
        (existing_normalized, candidate_normalized),
        key=len,
    )
    return len(shorter) >= 6 and shorter in longer and (len(shorter) / max(len(longer), 1)) >= 0.75


def _ocr_item_quality(item, text: str):
    normalized = _normalize_ocr_text(text)
    return (
        len(normalized),
        len(text or ""),
        len(_collect_text_regions(item)),
        bbox_area(item.get("render_bbox") or item.get("ocr_bbox") or item.get("coords") or (0, 0, 0, 0)),
    )


def dedupe_ocr_items_by_text_and_geometry(render_items, texts, logger: Callable[[str], None] | None = None):
    if len(render_items) != len(texts):
        raise ValueError("render_items and texts must have the same length")

    filtered_items: list[dict] = []
    filtered_texts: list[str] = []

    for item, text in zip(render_items, texts):
        match_index = None
        for index, existing_item in enumerate(filtered_items):
            render_existing = existing_item.get("render_bbox") or existing_item.get("ocr_bbox") or existing_item.get("coords")
            render_candidate = item.get("render_bbox") or item.get("ocr_bbox") or item.get("coords")
            overlap_existing, overlap_candidate = overlap_over_area(render_existing, render_candidate)
            if (
                _ocr_similarity(filtered_texts[index], text)
                and (
                    overlap_existing >= 0.90
                    or overlap_candidate >= 0.90
                    or bbox_iou(render_existing, render_candidate) >= 0.60
                )
            ):
                match_index = index
                break

        if match_index is None:
            filtered_items.append(item)
            filtered_texts.append(text)
            continue

        if _ocr_item_quality(item, text) > _ocr_item_quality(filtered_items[match_index], filtered_texts[match_index]):
            kept_bbox = filtered_items[match_index].get("render_bbox")
            filtered_items[match_index] = item
            filtered_texts[match_index] = text
            _log(
                logger,
                f"Dropped OCR duplicate {kept_bbox} in favor of {item.get('render_bbox')}",
            )
        else:
            _log(
                logger,
                f"Dropped OCR duplicate {item.get('render_bbox')} kept {filtered_items[match_index].get('render_bbox')}",
            )

    return filtered_items, filtered_texts


__all__ = [
    "build_outside_text_blocks",
    "consolidate_render_items",
    "dedupe_ocr_items_by_text_and_geometry",
]
