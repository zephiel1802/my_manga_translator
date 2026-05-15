"""Resident detection engine for preloaded detector ownership."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from detectors import (
    detect_page_regions_layout_first,
    get_comic_text_detector,
    get_pp_doclayout_v3_detector,
    get_yolov8_seg_bubble_detector,
)


Logger = Callable[[str], None] | None
StatusCallback = Callable[[str], None] | None


@dataclass(slots=True)
class DetectionEngine:
    """Owns the detector instances used by the desktop studio runtime."""

    bubble_detector: Any | None = None
    layout_detector: Any | None = None
    text_detector: Any | None = None

    def preload(self, *, logger: Logger = None, status_callback: StatusCallback = None) -> None:
        _status(status_callback, "Loading YOLO bubble detector...")
        _log(logger, "Loading YOLO bubble detector...")
        self.bubble_detector = get_yolov8_seg_bubble_detector()
        if hasattr(self.bubble_detector, "load"):
            self.bubble_detector.load()

        _status(status_callback, "Loading PPLayout detector...")
        _log(logger, "Loading PPLayout detector...")
        self.layout_detector = get_pp_doclayout_v3_detector()
        if hasattr(self.layout_detector, "load"):
            self.layout_detector.load()

        _status(status_callback, "Loading comic/text detector...")
        _log(logger, "Loading comic/text detector...")
        self.text_detector = get_comic_text_detector()
        if hasattr(self.text_detector, "load"):
            self.text_detector.load()

    def is_ready(self) -> bool:
        return self.bubble_detector is not None and self.layout_detector is not None and self.text_detector is not None

    def clear(self) -> None:
        self.bubble_detector = None
        self.layout_detector = None
        self.text_detector = None

    def detect_image(self, image: Any, *, logger: Logger = None):
        if not self.is_ready():
            raise RuntimeError("Detection models are not loaded. Restart the Detection service.")
        _log(logger, "Running resident detection inference...")
        return detect_page_regions_layout_first(
            image,
            layout_detector=self.layout_detector,
            bubble_detector=self.bubble_detector,
            text_detector=self.text_detector,
        )


def _log(logger: Logger, message: str) -> None:
    if logger is not None:
        logger(str(message or ""))


def _status(callback: StatusCallback, message: str) -> None:
    if callback is not None:
        callback(str(message or ""))


__all__ = ["DetectionEngine"]
