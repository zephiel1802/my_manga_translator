"""Desktop compatibility adapter for the old Flask OCR preprocessing chain."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None


DEFAULT_MIN_SHORT_SIDE = 10
DEFAULT_MAX_LONG_SIDE = 1000
DEFAULT_IMAGE_PAD = 12
DEFAULT_MAX_UPSCALE_FACTOR = 4.0


def preprocess_ocr_provider_image(image: Any) -> Image.Image:
    """Mirror the old Flask OCR crop conversion and Paddle preprocessing chain."""

    pil_rgb = _legacy_crop_to_pil_rgb(image)
    normalized = _legacy_paddle_normalize_image(pil_rgb)
    return _legacy_paddle_preprocess_ocr_image(normalized)


def save_ocr_provider_image(image: Any, output_path: Path | str) -> Path:
    """Save one provider image using the desktop legacy-compatibility adapter."""

    if Image is None:
        raise RuntimeError("Pillow is required for OCR provider image preprocessing.")

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    preprocessed_image = preprocess_ocr_provider_image(image)
    preprocessed_image.save(path, format="PNG")
    return path


def _legacy_crop_to_pil_rgb(image: Any) -> Image.Image:
    """Reproduce the old Flask OCR crop conversion into a PIL RGB image."""

    if Image is None:
        raise RuntimeError("Pillow is required for OCR provider image preprocessing.")

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if np is not None and isinstance(image, np.ndarray):
        array = image
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 2:
            return Image.fromarray(array).convert("RGB")
        if array.ndim == 3 and array.shape[2] == 3:
            return Image.fromarray(array[:, :, ::-1]).convert("RGB")
        if array.ndim == 3 and array.shape[2] == 4:
            return Image.fromarray(array[:, :, [2, 1, 0, 3]]).convert("RGB")
        if array.ndim != 3:
            raise ValueError(f"Unsupported OCR crop shape: {array.shape}")
        raise ValueError(f"Unsupported OCR crop channel count: {array.shape[2]}")

    raise ValueError("OCR crop must be a PIL image or numpy ndarray.")


def _legacy_paddle_normalize_image(image: Any) -> Image.Image:
    """Reproduce PaddleOCRVLOCR._normalize_image for the desktop adapter."""

    if Image is None:
        raise RuntimeError("Pillow is required for OCR provider image preprocessing.")

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if np is not None and isinstance(image, np.ndarray):
        array = image
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 2:
            return Image.fromarray(array).convert("RGB")
        if array.ndim == 3 and array.shape[2] == 3:
            return Image.fromarray(array[:, :, ::-1]).convert("RGB")
        if array.ndim == 3 and array.shape[2] == 4:
            return Image.fromarray(array[:, :, [2, 1, 0, 3]]).convert("RGB")
        raise RuntimeError(
            f"PaddleOCR-VL received unsupported numpy crop shape: {array.shape}"
        )

    raise RuntimeError("PaddleOCR-VL expects a PIL.Image or numpy array crop.")


def _legacy_paddle_preprocess_ocr_image(pil_image: Image.Image) -> Image.Image:
    """Reproduce PaddleOCRVLOCR._preprocess_ocr_image for desktop OCR."""

    image = pil_image.convert("RGB")

    if DEFAULT_IMAGE_PAD > 0:
        padded = Image.new(
            "RGB",
            (image.width + DEFAULT_IMAGE_PAD * 2, image.height + DEFAULT_IMAGE_PAD * 2),
            "white",
        )
        padded.paste(image, (DEFAULT_IMAGE_PAD, DEFAULT_IMAGE_PAD))
        image = padded

    width, height = image.size
    if width <= 0 or height <= 0:
        return image

    short_side = min(width, height)
    long_side = max(width, height)
    scale = 1.0

    if DEFAULT_MIN_SHORT_SIDE > 0 and short_side < DEFAULT_MIN_SHORT_SIDE:
        desired_scale = DEFAULT_MIN_SHORT_SIDE / short_side
        scale = max(scale, min(desired_scale, DEFAULT_MAX_UPSCALE_FACTOR))

    if DEFAULT_MAX_LONG_SIDE > 0 and long_side * scale > DEFAULT_MAX_LONG_SIDE:
        scale = min(scale, DEFAULT_MAX_LONG_SIDE / long_side)

    if abs(scale - 1.0) > 0.01 and (scale < 1.0 or scale > 1.01):
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return image


__all__ = [
    "DEFAULT_IMAGE_PAD",
    "DEFAULT_MAX_LONG_SIDE",
    "DEFAULT_MAX_UPSCALE_FACTOR",
    "DEFAULT_MIN_SHORT_SIDE",
    "preprocess_ocr_provider_image",
    "save_ocr_provider_image",
]
