from __future__ import annotations

from typing import Callable, Sequence

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None

from ocr_crop_utils import ocr_crop_to_pil_rgb
from render_item_utils import dedupe_ocr_items_by_text_and_geometry


def run_page_scoped_ocr(
    all_pages_data: dict,
    page_order: Sequence[str],
    mocr,
    logger: Callable[[str], None] | None = None,
    page_callback: Callable[[int, int, str, int], None] | None = None,
):
    total_pages = len(page_order)
    total_text_items = 0

    if Image is None:
        raise RuntimeError("Pillow is required for page-scoped OCR crop conversion.")

    for page_index, page_name in enumerate(page_order, start=1):
        page_data = all_pages_data[page_name]
        render_items = list(page_data.get("render_items") or [])
        item_count = len(render_items)

        if page_callback is not None:
            page_callback(page_index, total_pages, page_name, item_count)
        if logger is not None:
            logger(f"[OCR Page {page_index}/{total_pages}] {page_name}: {item_count} text items")

        if not render_items:
            page_data["texts"] = []
            if hasattr(mocr, "reset_session"):
                mocr.reset_session()
            continue

        page_ocr_images = [
            ocr_crop_to_pil_rgb(render_item["ocr_crop"])
            for render_item in render_items
        ]

        if hasattr(mocr, "process_batch"):
            page_texts = mocr.process_batch(page_ocr_images)
        else:
            page_texts = [mocr(image) for image in page_ocr_images]

        filtered_render_items, filtered_texts = dedupe_ocr_items_by_text_and_geometry(
            render_items,
            page_texts,
            logger=logger,
        )
        page_data["render_items"] = filtered_render_items
        page_data["texts"] = filtered_texts
        total_text_items += len(filtered_texts)

        if hasattr(mocr, "reset_session"):
            mocr.reset_session()

    return total_text_items


def build_translation_inputs_from_pages(
    all_pages_data: dict,
    page_order: Sequence[str],
) -> tuple[list[str], list[tuple[str, int]]]:
    ordered_texts: list[str] = []
    translation_mapping: list[tuple[str, int]] = []

    for page_name in page_order:
        page_texts = list(all_pages_data[page_name].get("texts") or [])
        for item_index, text in enumerate(page_texts):
            ordered_texts.append(text)
            translation_mapping.append((page_name, item_index))

    return ordered_texts, translation_mapping


__all__ = [
    "build_translation_inputs_from_pages",
    "run_page_scoped_ocr",
]
