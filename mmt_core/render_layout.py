"""Render-stage bbox and writing-mode helpers."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from text_rendering import contains_cjk


def clamp_bbox_to_image(
    bbox: Any,
    image_shape: Sequence[int],
) -> tuple[int, int, int, int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None

    height = int(image_shape[0])
    width = int(image_shape[1])
    x1, y1, x2, y2 = [int(value) for value in bbox[:4]]

    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def bbox_area(bbox: tuple[int, int, int, int] | None) -> int:
    if bbox is None:
        return 0
    return max(0, int(bbox[2]) - int(bbox[0])) * max(0, int(bbox[3]) - int(bbox[1]))


def bbox_to_list(bbox: Sequence[int | float] | None) -> list[int] | None:
    if bbox is None:
        return None
    return [int(value) for value in bbox[:4]]


def choose_render_bbox(
    *,
    kind: str,
    image_shape: Sequence[int],
    translation_bbox: Any,
    translation_ocr_bbox: Any,
    ocr_bbox: Any,
    ocr_item_bbox: Any,
) -> tuple[int, int, int, int] | None:
    """Choose a conservative render box.

    Bubble items can safely use the larger detector bubble box when available.
    Other item kinds prefer the OCR-refined box.
    """

    preferred_ocr_box = clamp_bbox_to_image(
        translation_ocr_bbox if translation_ocr_bbox is not None else ocr_bbox,
        image_shape,
    )
    base_box = clamp_bbox_to_image(
        translation_bbox if translation_bbox is not None else ocr_item_bbox,
        image_shape,
    )

    if kind == "bubble":
        if base_box is not None:
            return base_box
        return preferred_ocr_box

    return preferred_ocr_box or base_box


def choose_writing_mode(
    text: str,
    bbox: tuple[int, int, int, int],
    *,
    source_direction: str | None,
    auto_direction: bool,
    vertical_cjk: bool,
) -> str:
    if not vertical_cjk:
        return "horizontal"

    normalized_source_direction = str(source_direction or "").strip().lower()
    has_cjk = contains_cjk(text)
    width = max(1, int(bbox[2]) - int(bbox[0]))
    height = max(1, int(bbox[3]) - int(bbox[1]))
    box_is_tall = height >= int(width * 1.1)

    if not auto_direction:
        if normalized_source_direction.startswith("vertical") and has_cjk:
            return "vertical_rl"
        return "horizontal"

    if has_cjk and normalized_source_direction.startswith("vertical"):
        return "vertical_rl"
    if has_cjk and box_is_tall:
        return "vertical_rl"
    return "horizontal"


def iter_vertical_tokens(text: str) -> list[str]:
    """Tokenize text for best-effort vertical layout.

    This groups short punctuation pairs like ``!?`` so they do not look
    especially awkward when stacked as individual glyph cells.
    """

    text_without_newlines = "".join(character for character in text if character != "\n")
    if not text_without_newlines:
        return []

    pairable = {"!", "?", "！", "？", "‼", "⁉"}
    tokens: list[str] = []
    cursor = 0
    while cursor < len(text_without_newlines):
        current = text_without_newlines[cursor]
        if cursor + 1 < len(text_without_newlines):
            following = text_without_newlines[cursor + 1]
            if current in pairable and following in pairable:
                tokens.append(current + following)
                cursor += 2
                continue
        tokens.append(current)
        cursor += 1
    return tokens


def choose_best_font_size(
    candidates: Iterable[int],
    *,
    default_value: int,
) -> int:
    for candidate in candidates:
        return int(candidate)
    return int(default_value)


__all__ = [
    "bbox_area",
    "bbox_to_list",
    "choose_best_font_size",
    "choose_render_bbox",
    "choose_writing_mode",
    "clamp_bbox_to_image",
    "iter_vertical_tokens",
]
