from __future__ import annotations

from typing import Sequence

from .base import TextRegion


def _clamp_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: Sequence[int],
) -> tuple[int, int, int, int]:
    height = int(image_shape[0])
    width = int(image_shape[1])
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), width))
    y1 = max(0, min(int(y1), height))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def _polygon_to_bbox(polygon) -> tuple[int, int, int, int] | None:
    points = []
    try:
        for point in polygon:
            if len(point) < 2:
                continue
            points.append((int(point[0]), int(point[1])))
    except TypeError:
        return None

    if not points:
        return None

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def text_region_line_polygons_as_bboxes(
    text_region: TextRegion,
) -> list[tuple[int, int, int, int]]:
    if not text_region.line_polygons:
        return []

    polygon_bboxes: list[tuple[int, int, int, int]] = []
    if (
        isinstance(text_region.line_polygons, (list, tuple))
        and text_region.line_polygons
        and isinstance(text_region.line_polygons[0], (list, tuple))
        and len(text_region.line_polygons[0]) >= 2
        and isinstance(text_region.line_polygons[0][0], (int, float))
    ):
        raw_polygons = [text_region.line_polygons]
    else:
        raw_polygons = list(text_region.line_polygons)

    for polygon in raw_polygons:
        bbox = _polygon_to_bbox(polygon)
        if bbox is not None and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            polygon_bboxes.append(bbox)
    return polygon_bboxes


def expanded_text_block_crop_bounds(
    image_shape: Sequence[int],
    text_region: TextRegion,
) -> tuple[int, int, int, int]:
    bbox = _clamp_bbox(text_region.bbox, image_shape)
    width = max(1, bbox[2] - bbox[0])
    height = max(1, bbox[3] - bbox[1])

    polygon_bboxes = text_region_line_polygons_as_bboxes(text_region)
    use_refinement = (
        text_region.detector == "comic_text_detector"
        or bool(polygon_bboxes)
    )
    if not use_refinement:
        return bbox

    bounds = [bbox]
    bounds.extend(polygon_bboxes)
    min_x = min(candidate[0] for candidate in bounds)
    min_y = min(candidate[1] for candidate in bounds)
    max_x = max(candidate[2] for candidate in bounds)
    max_y = max(candidate[3] for candidate in bounds)

    font_size = float(
        text_region.detected_font_size_px
        if text_region.detected_font_size_px is not None
        else min(width, height)
    )
    base_pad = max(font_size * 0.08, 2.0)
    direction = text_region.source_direction or (
        "vertical" if height >= (width * 1.15) else "horizontal"
    )
    if direction == "vertical":
        pad_x = max(font_size * 0.18, base_pad)
        pad_y = max(font_size * 0.12, base_pad)
    else:
        pad_x = max(font_size * 0.12, base_pad)
        pad_y = max(font_size * 0.18, base_pad)

    return _clamp_bbox(
        (
            int(round(min_x - pad_x)),
            int(round(min_y - pad_y)),
            int(round(max_x + pad_x)),
            int(round(max_y + pad_y)),
        ),
        image_shape,
    )


__all__ = [
    "expanded_text_block_crop_bounds",
    "text_region_line_polygons_as_bboxes",
]
