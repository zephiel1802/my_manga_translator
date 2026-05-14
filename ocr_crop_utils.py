from __future__ import annotations

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None


def ocr_crop_to_pil_rgb(crop):
    if Image is None:
        raise RuntimeError("Pillow is required for OCR crop conversion.")

    if isinstance(crop, Image.Image):
        return crop.convert("RGB")

    if np is None or not isinstance(crop, np.ndarray):
        raise ValueError("OCR crop must be a PIL image or numpy ndarray.")

    array = crop
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.ndim == 2:
        return Image.fromarray(array).convert("RGB")

    if array.ndim != 3:
        raise ValueError(f"Unsupported OCR crop shape: {array.shape}")

    channels = array.shape[2]
    if channels == 3:
        rgb = array[:, :, ::-1]
        return Image.fromarray(rgb).convert("RGB")

    if channels == 4:
        rgba = array[:, :, [2, 1, 0, 3]]
        return Image.fromarray(rgba).convert("RGB")

    raise ValueError(f"Unsupported OCR crop channel count: {channels}")


__all__ = ["ocr_crop_to_pil_rgb"]
