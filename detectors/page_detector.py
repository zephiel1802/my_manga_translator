from __future__ import annotations

from .base import PageDetectionResult, TextRegion
from .comic_text_detector import get_comic_text_detector
from .matching import assign_text_regions_to_bubbles
from .pp_doclayout_v3 import (
    get_pp_doclayout_v3_detector,
    layout_regions_to_text_regions,
)
from .runtime_utils import merge_duplicate_bubble_regions
from .selection import dedupe_text_regions_koharu_style, sort_manga_reading_order
from .yolov8_seg_bubble import get_yolov8_seg_bubble_detector


def merge_text_region_candidates(
    pp_text_regions,
    comic_text_regions,
    *,
    image_shape,
    iou_threshold: float = 0.35,
) -> list[TextRegion]:
    merged_regions = dedupe_text_regions_koharu_style(
        list(pp_text_regions) + list(comic_text_regions),
        image_shape=image_shape,
        iou_threshold=iou_threshold,
        prefer_comic=True,
    )
    return sort_manga_reading_order(merged_regions, order="ltr")


def detect_page_regions_layout_first(
    image,
    *,
    layout_detector=None,
    bubble_detector=None,
    text_detector=None,
) -> PageDetectionResult:
    active_layout_detector = (
        layout_detector
        if layout_detector is not None
        else get_pp_doclayout_v3_detector()
    )
    active_bubble_detector = (
        bubble_detector
        if bubble_detector is not None
        else get_yolov8_seg_bubble_detector()
    )
    active_text_detector = (
        text_detector if text_detector is not None else get_comic_text_detector()
    )

    layout_regions = active_layout_detector.detect_layout_regions(image)
    pp_text_regions = layout_regions_to_text_regions(
        layout_regions,
        image.shape,
    )
    raw_bubbles = active_bubble_detector.detect_segmented_bubble_regions(image)
    bubbles = merge_duplicate_bubble_regions(
        raw_bubbles,
        image_shape=image.shape,
    )
    comic_text_regions = active_text_detector.detect_text_regions(image)
    merged_text_regions = merge_text_region_candidates(
        pp_text_regions,
        comic_text_regions,
        image_shape=image.shape,
    )
    matched_text_regions = assign_text_regions_to_bubbles(
        merged_text_regions,
        bubbles,
    )

    return PageDetectionResult(
        bubbles=bubbles,
        text_regions=matched_text_regions,
        layout_regions=layout_regions,
        method="pp_doclayout_v3_text_source+yolov8_seg_bubble+comic_text_detector",
        stats={
            "raw_bubbles": getattr(active_bubble_detector, "last_raw_bubble_count", len(raw_bubbles)),
            "merged_bubbles": getattr(active_bubble_detector, "last_merged_bubble_count", len(bubbles)),
            "raw_pp_text_regions": len(pp_text_regions),
            "raw_comic_text_regions": len(comic_text_regions),
            "merged_text_regions": len(merged_text_regions),
            "matched_text_regions": len(matched_text_regions),
        },
    )


__all__ = [
    "detect_page_regions_layout_first",
    "merge_text_region_candidates",
]
