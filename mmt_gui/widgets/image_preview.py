"""Image preview widget with fit-to-window scaling, mask overlays, and detection boxes."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)


class ImagePreviewWidget(QGraphicsView):
    """Displays a page image and optional detection overlays."""

    def __init__(self, parent: QGraphicsView | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._mask_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._scene.addItem(self._mask_item)
        self._overlay_items: list[QGraphicsRectItem] = []
        self.setScene(self._scene)

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setBackgroundBrush(self.palette().brush(self.backgroundRole()))

    def clear_image(self) -> None:
        self.clear_overlays()
        self.clear_mask_overlay()
        self._pixmap_item.setPixmap(QPixmap())
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.resetTransform()

    def clear_overlays(self) -> None:
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items.clear()

    def clear_mask_overlay(self) -> None:
        self._mask_item.setPixmap(QPixmap())
        self._mask_item.setOpacity(0.0)
        self._mask_item.setVisible(False)

    def set_image(self, image_path: Path | str | None) -> bool:
        if image_path is None:
            self.clear_image()
            return False

        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.clear_image()
            return False

        self.clear_overlays()
        self.clear_mask_overlay()
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fit_to_view()
        return True

    def set_mask_overlay(self, mask_path: Path | str | None) -> bool:
        if mask_path is None:
            self.clear_mask_overlay()
            return False

        pixmap = QPixmap(str(mask_path))
        if pixmap.isNull():
            self.clear_mask_overlay()
            return False

        self._mask_item.setPixmap(pixmap)
        self._mask_item.setPos(0, 0)
        self._mask_item.setOpacity(0.35)
        self._mask_item.setZValue(5)
        self._mask_item.setVisible(True)
        return True

    def set_detection_overlay(self, detection_data: dict[str, Any] | None) -> None:
        self.clear_overlays()

        if not detection_data:
            return

        self._add_rectangles(
            detection_data.get("layout_regions", []),
            color=QColor(70, 180, 255),
            line_style=Qt.PenStyle.DashLine,
            label_prefix="Layout",
        )
        self._add_rectangles(
            detection_data.get("bubbles", []),
            color=QColor(80, 220, 120),
            line_style=Qt.PenStyle.SolidLine,
            label_prefix="Bubble",
        )
        self._add_rectangles(
            detection_data.get("text_regions", []),
            color=QColor(255, 196, 61),
            line_style=Qt.PenStyle.DotLine,
            label_prefix="Text",
        )

    def fit_to_view(self) -> None:
        pixmap = self._pixmap_item.pixmap()
        if pixmap.isNull():
            return

        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self.fit_to_view()

    def _add_rectangles(
        self,
        regions: Iterable[dict[str, Any]],
        *,
        color: QColor,
        line_style: Qt.PenStyle,
        label_prefix: str,
    ) -> None:
        for region in regions:
            bbox = region.get("bbox")
            rect = _rect_from_bbox(bbox)
            if rect is None:
                continue

            pen = QPen(color)
            pen.setStyle(line_style)
            pen.setWidth(2)
            pen.setCosmetic(True)

            rect_item = QGraphicsRectItem(rect)
            rect_item.setPen(pen)
            rect_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            rect_item.setZValue(10)
            rect_item.setToolTip(_tooltip_for_region(region, label_prefix))
            self._scene.addItem(rect_item)
            self._overlay_items.append(rect_item)


def _rect_from_bbox(bbox: Any) -> QRectF | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None

    try:
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    except (TypeError, ValueError):
        return None

    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return None

    return QRectF(x1, y1, width, height)


def _tooltip_for_region(region: dict[str, Any], label_prefix: str) -> str:
    region_id = region.get("id", "?")
    detector = region.get("detector", "unknown")
    confidence = region.get("confidence")
    if confidence is None:
        return f"{label_prefix} #{region_id}\nDetector: {detector}"
    return f"{label_prefix} #{region_id}\nDetector: {detector}\nConfidence: {float(confidence):.3f}"
