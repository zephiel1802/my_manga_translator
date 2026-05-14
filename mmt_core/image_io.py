"""Shared image loading and path helpers for GUI-facing pipeline stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def ensure_path(path_like: Path | str) -> Path:
    """Normalize a path-like input into a resolved ``Path``."""

    return Path(path_like).expanduser().resolve()


def project_relative_path(project_root: Path | str, path_like: Path | str) -> str:
    """Return a stable POSIX-style path relative to a project root when possible."""

    root_path = ensure_path(project_root)
    target_path = ensure_path(path_like)

    try:
        return target_path.relative_to(root_path).as_posix()
    except ValueError:
        return target_path.name


def load_image_bgr(image_path: Path | str) -> Any:
    """Load an image from disk as an OpenCV BGR array."""

    path = ensure_path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Source image does not exist: {path}")

    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required to load project images.") from exc

    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("NumPy is required to load project images.") from exc

    encoded_bytes = np.fromfile(str(path), dtype=np.uint8)
    if encoded_bytes.size == 0:
        raise ValueError(f"Source image is empty or unreadable: {path}")

    image = cv2.imdecode(encoded_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")

    return image


def load_image_grayscale(image_path: Path | str) -> Any:
    """Load an image from disk as a single-channel grayscale array."""

    path = ensure_path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Source image does not exist: {path}")

    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required to load project images.") from exc

    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("NumPy is required to load project images.") from exc

    encoded_bytes = np.fromfile(str(path), dtype=np.uint8)
    if encoded_bytes.size == 0:
        raise ValueError(f"Source image is empty or unreadable: {path}")

    image = cv2.imdecode(encoded_bytes, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")

    return image


def save_png_image(image_array: Any, output_path: Path | str) -> Path:
    """Safely save an array-like image to PNG on disk."""

    path = ensure_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required to save cached image files.") from exc

    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("NumPy is required to save cached image files.") from exc

    array = np.asarray(image_array)
    if array.ndim not in (2, 3):
        raise ValueError(f"Unsupported image array shape for PNG save: {array.shape}")

    if array.dtype == np.bool_:
        array = array.astype(np.uint8) * 255
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    success, encoded = cv2.imencode(".png", array)
    if not success:
        raise ValueError(f"Failed to encode PNG image for: {path}")

    encoded.tofile(str(path))
    return path
