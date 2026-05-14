"""Core helpers for the desktop GUI pipeline integration."""

from __future__ import annotations

from .detection_io import (
    detection_json_path,
    load_detection_json,
    mask_dir_for_page,
    save_detection_result,
)
from .detection_stage import run_detection_for_image
from .llama_server import LlamaServerManager, LlamaServerStatus
from .inpaint_io import (
    bubble_mask_path,
    inpaint_image_path,
    inpaint_json_path,
    inpaint_preview_mask_path,
    load_inpaint_json,
    save_inpaint_json,
    summarize_inpaint_json,
    text_mask_path,
)
from .inpaint_stage import (
    get_lama_model_manager,
    load_lama_model,
    prepare_inpaint_mask_for_page,
    run_inpaint_for_page,
    run_inpaint_for_pages,
    unload_lama_model,
)
from .ocr_io import (
    load_ocr_json,
    normalize_ocr_item,
    ocr_crop_dir_for_page,
    ocr_json_path,
    save_ocr_payload,
    save_ocr_items_result,
    summarize_ocr_items,
)
from .ocr_stage import prepare_ocr_items_for_image, run_ocr_for_page
from .render_io import (
    load_render_json,
    render_image_path,
    render_json_path,
    render_sprite_dir,
    save_render_json,
    summarize_render_json,
)
from .render_models import RenderConfig
from .render_stage import (
    build_render_items_from_translation,
    prepare_render_for_page,
    run_render_for_page,
    run_render_for_pages,
)
from .render_styles import list_project_fonts, parse_color_value, resolve_font_path
from .paddleocr_vl_client import (
    DEFAULT_PROMPT,
    PaddleOCRVLClient,
    PaddleOCRVLClientError,
)
from .translation_io import (
    initialize_translation_from_ocr,
    load_translation_json,
    normalize_translation_item,
    save_translation_json,
    summarize_translation_json,
    translation_json_path,
)
from .translation_models import (
    LANGUAGE_CHOICES,
    STYLE_PROMPTS,
    TRANSLATOR_CHOICES,
    TranslationConfig,
    normalize_translator_name,
)
from .translation_stage import (
    initialize_translation_for_page,
    run_translation_for_page,
    run_translation_for_pages,
)

__all__ = [
    "detection_json_path",
    "load_ocr_json",
    "load_translation_json",
    "load_detection_json",
    "DEFAULT_PROMPT",
    "LANGUAGE_CHOICES",
    "LlamaServerManager",
    "LlamaServerStatus",
    "bubble_mask_path",
    "get_lama_model_manager",
    "inpaint_image_path",
    "inpaint_json_path",
    "inpaint_preview_mask_path",
    "load_inpaint_json",
    "load_lama_model",
    "mask_dir_for_page",
    "normalize_ocr_item",
    "normalize_translation_item",
    "normalize_translator_name",
    "ocr_crop_dir_for_page",
    "ocr_json_path",
    "PaddleOCRVLClient",
    "PaddleOCRVLClientError",
    "RenderConfig",
    "render_image_path",
    "render_json_path",
    "render_sprite_dir",
    "resolve_font_path",
    "list_project_fonts",
    "parse_color_value",
    "prepare_ocr_items_for_image",
    "prepare_inpaint_mask_for_page",
    "prepare_render_for_page",
    "run_inpaint_for_page",
    "run_inpaint_for_pages",
    "run_render_for_page",
    "run_render_for_pages",
    "STYLE_PROMPTS",
    "TRANSLATOR_CHOICES",
    "TranslationConfig",
    "translation_json_path",
    "initialize_translation_from_ocr",
    "initialize_translation_for_page",
    "run_translation_for_page",
    "run_translation_for_pages",
    "run_ocr_for_page",
    "run_detection_for_image",
    "build_render_items_from_translation",
    "save_ocr_payload",
    "save_inpaint_json",
    "save_ocr_items_result",
    "save_render_json",
    "save_translation_json",
    "save_detection_result",
    "load_render_json",
    "summarize_inpaint_json",
    "summarize_ocr_items",
    "summarize_render_json",
    "summarize_translation_json",
    "text_mask_path",
    "unload_lama_model",
]
