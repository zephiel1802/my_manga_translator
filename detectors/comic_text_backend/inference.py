from __future__ import annotations

from statistics import median
from typing import Any

import cv2
import numpy as np
import torch

from .basemodel import TextDetBase
from .yolov5_utils import non_max_suppression


def letterbox(
    image: np.ndarray,
    new_shape=(1024, 1024),
    color=(0, 0, 0),
    scaleup=True,
):
    shape = image.shape[:2]
    if not isinstance(new_shape, tuple):
        new_shape = (new_shape, new_shape)

    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        ratio = min(ratio, 1.0)

    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw = int(dw)
    dh = int(dh)

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    image = cv2.copyMakeBorder(image, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=color)
    return image, (ratio, ratio), (dw, dh)


def preprocess_image(
    image: np.ndarray,
    input_size=(1024, 1024),
    device: str = "cpu",
) -> tuple[torch.Tensor, tuple[int, int]]:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_in, _, (dw, dh) = letterbox(image_rgb, new_shape=input_size)
    image_in = image_in.transpose((2, 0, 1))[::-1]
    image_in = np.ascontiguousarray(image_in)[None].astype(np.float32) / 255.0
    tensor = torch.from_numpy(image_in).to(device)
    return tensor, (dw, dh)


def _resize_map_to_original(
    pred_map: np.ndarray,
    original_shape: tuple[int, int],
    padding: tuple[int, int],
) -> np.ndarray:
    dw, dh = padding
    if dh > 0:
        pred_map = pred_map[: pred_map.shape[0] - dh, :]
    if dw > 0:
        pred_map = pred_map[:, : pred_map.shape[1] - dw]
    original_height, original_width = original_shape
    return cv2.resize(pred_map, (original_width, original_height), interpolation=cv2.INTER_LINEAR)


def _mean_contour_score(prob_map: np.ndarray, contour: np.ndarray) -> float:
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return 0.0
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = contour.copy()
    shifted[:, 0, 0] -= x
    shifted[:, 0, 1] -= y
    cv2.fillPoly(mask, [shifted], 1)
    region = prob_map[y : y + h, x : x + w]
    return float(cv2.mean(region, mask)[0])


def _bbox_polygon(bbox: list[int] | tuple[int, int, int, int]) -> list[list[int]]:
    x1, y1, x2, y2 = [int(value) for value in bbox[:4]]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _bbox_area(bbox: list[int] | tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_center(bbox: list[int] | tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _center_in_bbox(center: tuple[float, float], bbox, padding: int = 0) -> bool:
    x, y = center
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    pad = float(padding)
    return (x1 - pad) <= x <= (x2 + pad) and (y1 - pad) <= y <= (y2 + pad)


def _bbox_intersection_area(a, b) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in a[:4]]
    bx1, by1, bx2, by2 = [float(value) for value in b[:4]]
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    return width * height


def _overlap_over_area(a, b) -> tuple[float, float]:
    intersection = _bbox_intersection_area(a, b)
    area_a = _bbox_area(a)
    area_b = _bbox_area(b)
    return (
        float(intersection / area_a) if area_a > 0 else 0.0,
        float(intersection / area_b) if area_b > 0 else 0.0,
    )


def _infer_source_direction(bbox) -> str:
    width = max(1, int(bbox[2]) - int(bbox[0]))
    height = max(1, int(bbox[3]) - int(bbox[1]))
    return "vertical" if height >= (width * 1.15) else "horizontal"


def _infer_font_size_from_lines(line_bboxes, fallback_bbox, direction: str) -> float:
    if line_bboxes:
        if direction == "vertical":
            values = [
                max(1, int(line_bbox[2]) - int(line_bbox[0]))
                for line_bbox in line_bboxes
            ]
        else:
            values = [
                max(1, int(line_bbox[3]) - int(line_bbox[1]))
                for line_bbox in line_bboxes
            ]
        return float(max(1.0, median(values)))

    return float(
        max(
            1,
            min(
                max(1, int(fallback_bbox[2]) - int(fallback_bbox[0])),
                max(1, int(fallback_bbox[3]) - int(fallback_bbox[1])),
            ),
        )
    )


def _union_bbox(a, b) -> list[int]:
    return [
        int(min(a[0], b[0])),
        int(min(a[1], b[1])),
        int(max(a[2], b[2])),
        int(max(a[3], b[3])),
    ]


def _should_merge_line_into_group(line_bbox, group_bbox, *, median_height: float, median_width: float) -> bool:
    overlap_group_x, overlap_line_x = _overlap_over_area(
        (group_bbox[0], 0, group_bbox[2], 1),
        (line_bbox[0], 0, line_bbox[2], 1),
    )
    overlap_group_y, overlap_line_y = _overlap_over_area(
        (0, group_bbox[1], 1, group_bbox[3]),
        (0, line_bbox[1], 1, line_bbox[3]),
    )

    vertical_gap = max(
        0,
        max(int(line_bbox[1]) - int(group_bbox[3]), int(group_bbox[1]) - int(line_bbox[3])),
    )
    horizontal_gap = max(
        0,
        max(int(line_bbox[0]) - int(group_bbox[2]), int(group_bbox[0]) - int(line_bbox[2])),
    )
    horizontal_stack = (
        (overlap_group_x >= 0.25 or overlap_line_x >= 0.25)
        and vertical_gap <= max(int(round(median_height * 1.5)), 10)
    )
    vertical_stack = (
        (overlap_group_y >= 0.30 or overlap_line_y >= 0.30)
        and horizontal_gap <= max(int(round(median_width * 1.2)), 10)
    )
    return horizontal_stack or vertical_stack


def _extract_line_regions(
    line_pred: torch.Tensor,
    original_shape: tuple[int, int],
    padding: tuple[int, int],
    *,
    text_map_thresh: float,
    confidence_threshold: float,
    min_text_area: float,
) -> list[dict[str, Any]]:
    line_prob_map = line_pred[0, 0].detach().float().cpu().numpy()
    line_prob_map = _resize_map_to_original(line_prob_map, original_shape, padding)

    binary_map = (line_prob_map > text_map_thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    raw_regions: list[dict[str, Any]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_text_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 1 or h <= 1:
            continue

        score = _mean_contour_score(line_prob_map, contour)
        if score < confidence_threshold:
            continue

        raw_regions.append(
            {
                "bbox": [x, y, x + w, y + h],
                "confidence": score,
                "line_polygons": [_bbox_polygon((x, y, x + w, y + h))],
                "line_bboxes": [[x, y, x + w, y + h]],
                "source_direction": _infer_source_direction((x, y, x + w, y + h)),
                "detected_font_size_px": float(max(1, min(w, h))),
            }
        )

    raw_regions.sort(key=lambda region: (region["bbox"][1], region["bbox"][0]))
    for index, region in enumerate(raw_regions):
        region["reading_order"] = index

    return raw_regions


def _extract_block_regions(
    block_pred: torch.Tensor,
    original_shape: tuple[int, int],
    input_shape: tuple[int, int],
    padding: tuple[int, int],
    *,
    confidence_threshold: float,
    nms_threshold: float,
) -> list[dict[str, Any]]:
    detections = non_max_suppression(
        block_pred,
        conf_thres=confidence_threshold,
        iou_thres=nms_threshold,
    )[0]
    if detections.device != torch.device("cpu"):
        detections = detections.detach().cpu()
    detections = detections.numpy()

    original_height, original_width = original_shape
    input_height, input_width = input_shape
    dw, dh = padding
    scale_x = original_width / float(input_width - dw)
    scale_y = original_height / float(input_height - dh)

    raw_regions: list[dict[str, Any]] = []
    for index, detection in enumerate(detections):
        x1, y1, x2, y2, score, _ = detection[:6]
        x1 = max(0, min(int(x1 * scale_x), original_width))
        y1 = max(0, min(int(y1 * scale_y), original_height))
        x2 = max(0, min(int(x2 * scale_x), original_width))
        y2 = max(0, min(int(y2 * scale_y), original_height))
        if x2 <= x1 or y2 <= y1:
            continue
        raw_regions.append(
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": float(score),
                "reading_order": index,
            }
        )

    raw_regions.sort(key=lambda region: (region["bbox"][1], region["bbox"][0]))
    for index, region in enumerate(raw_regions):
        region["reading_order"] = index

    return raw_regions


def attach_line_regions_to_blocks(
    block_regions: list[dict[str, Any]],
    line_regions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not block_regions:
        return []

    attachments: list[list[dict[str, Any]]] = [[] for _ in block_regions]
    for line_region in line_regions:
        line_bbox = line_region["bbox"]
        line_area = max(_bbox_area(line_bbox), 1.0)
        line_center = _bbox_center(line_bbox)
        best_index = None
        best_score = -1.0

        for index, block_region in enumerate(block_regions):
            block_bbox = block_region["bbox"]
            overlap = _bbox_intersection_area(block_bbox, line_bbox) / line_area
            center_match = _center_in_bbox(line_center, block_bbox, padding=2)
            if overlap < 0.50 and not center_match:
                continue

            score = overlap + (0.5 if center_match else 0.0)
            if score > best_score:
                best_score = score
                best_index = index

        if best_index is not None:
            attachments[best_index].append(line_region)

    enriched_blocks: list[dict[str, Any]] = []
    for index, block_region in enumerate(block_regions):
        block_bbox = [int(value) for value in block_region["bbox"][:4]]
        attached_lines = sorted(
            attachments[index],
            key=lambda region: (
                region.get("reading_order", 10**9),
                region["bbox"][1],
                region["bbox"][0],
            ),
        )
        line_polygons: list[list[list[int]]] = []
        line_bboxes: list[list[int]] = []
        for line_region in attached_lines:
            line_bboxes.append([int(value) for value in line_region["bbox"][:4]])
            raw_polygons = line_region.get("line_polygons") or [_bbox_polygon(line_region["bbox"])]
            for polygon in raw_polygons:
                line_polygons.append(
                    [[int(point[0]), int(point[1])] for point in polygon]
                )

        direction = _infer_source_direction(block_bbox)
        enriched_blocks.append(
            {
                **block_region,
                "bbox": block_bbox,
                "reading_order": block_region.get("reading_order", index),
                "line_polygons": line_polygons,
                "line_bboxes": line_bboxes,
                "source_direction": direction,
                "detected_font_size_px": _infer_font_size_from_lines(
                    line_bboxes,
                    block_bbox,
                    direction,
                ),
                "detector": "comic_text_detector",
            }
        )

    enriched_blocks.sort(key=lambda region: (region["bbox"][1], region["bbox"][0]))
    for index, region in enumerate(enriched_blocks):
        region["reading_order"] = index
    return enriched_blocks


def group_line_regions_into_blocks(
    line_regions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not line_regions:
        return []

    ordered_lines = sorted(
        line_regions,
        key=lambda region: (
            region.get("reading_order", 10**9),
            region["bbox"][1],
            region["bbox"][0],
        ),
    )
    line_heights = [
        max(1, int(region["bbox"][3]) - int(region["bbox"][1]))
        for region in ordered_lines
    ]
    line_widths = [
        max(1, int(region["bbox"][2]) - int(region["bbox"][0]))
        for region in ordered_lines
    ]
    median_height = float(max(1.0, median(line_heights)))
    median_width = float(max(1.0, median(line_widths)))

    groups: list[dict[str, Any]] = []
    for line_region in ordered_lines:
        line_bbox = [int(value) for value in line_region["bbox"][:4]]
        match_index = None
        for index, group in enumerate(groups):
            if _should_merge_line_into_group(
                line_bbox,
                group["bbox"],
                median_height=median_height,
                median_width=median_width,
            ):
                match_index = index
                break

        if match_index is None:
            groups.append(
                {
                    "bbox": line_bbox,
                    "confidence": float(line_region.get("confidence", 1.0)),
                    "lines": [line_region],
                }
            )
            continue

        group = groups[match_index]
        group["bbox"] = _union_bbox(group["bbox"], line_bbox)
        group["confidence"] = max(group["confidence"], float(line_region.get("confidence", 1.0)))
        group["lines"].append(line_region)

    block_regions: list[dict[str, Any]] = []
    for index, group in enumerate(
        sorted(groups, key=lambda group_item: (group_item["bbox"][1], group_item["bbox"][0]))
    ):
        line_bboxes = [
            [int(value) for value in line_region["bbox"][:4]]
            for line_region in sorted(
                group["lines"],
                key=lambda line_region: (
                    line_region.get("reading_order", 10**9),
                    line_region["bbox"][1],
                    line_region["bbox"][0],
                ),
            )
        ]
        line_polygons: list[list[list[int]]] = []
        for line_region in group["lines"]:
            raw_polygons = line_region.get("line_polygons") or [_bbox_polygon(line_region["bbox"])]
            for polygon in raw_polygons:
                line_polygons.append(
                    [[int(point[0]), int(point[1])] for point in polygon]
                )

        direction = _infer_source_direction(group["bbox"])
        block_regions.append(
            {
                "bbox": group["bbox"],
                "confidence": float(group["confidence"]),
                "reading_order": index,
                "line_polygons": line_polygons,
                "line_bboxes": line_bboxes,
                "source_direction": direction,
                "detected_font_size_px": _infer_font_size_from_lines(
                    line_bboxes,
                    group["bbox"],
                    direction,
                ),
                "detector": "comic_text_detector",
                "grouped_from_lines": True,
            }
        )

    return block_regions


class PyTorchComicTextDetectorBackend:
    def __init__(
        self,
        *,
        yolo_weights_path: str,
        unet_weights_path: str,
        dbnet_weights_path: str,
        device: str = "cpu",
        input_size: int = 1024,
        confidence_threshold: float = 0.3,
        nms_threshold: float = 0.35,
        text_map_threshold: float = 0.3,
        min_text_area: float = 9.0,
        act: str = "leaky",
    ):
        self.device = device
        self.input_size = (input_size, input_size)
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self.text_map_threshold = float(text_map_threshold)
        self.min_text_area = float(min_text_area)
        self.last_block_region_count = 0
        self.last_line_region_count = 0
        self.last_grouped_from_lines_count = 0

        self.model = TextDetBase(
            yolo_weights_path=yolo_weights_path,
            unet_weights_path=unet_weights_path,
            dbnet_weights_path=dbnet_weights_path,
            device=device,
            act=act,
        ).eval()

    @torch.no_grad()
    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        image_tensor, padding = preprocess_image(
            image,
            input_size=self.input_size,
            device=self.device,
        )
        block_pred, _, line_pred = self.model(image_tensor)

        original_shape = image.shape[:2]
        block_regions = _extract_block_regions(
            block_pred,
            original_shape,
            self.input_size,
            padding,
            confidence_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
        )
        line_regions = _extract_line_regions(
            line_pred,
            original_shape,
            padding,
            text_map_thresh=self.text_map_threshold,
            confidence_threshold=self.confidence_threshold,
            min_text_area=self.min_text_area,
        )
        self.last_block_region_count = len(block_regions)
        self.last_line_region_count = len(line_regions)
        self.last_grouped_from_lines_count = 0

        if block_regions:
            return attach_line_regions_to_blocks(block_regions, line_regions)

        if line_regions:
            grouped_blocks = group_line_regions_into_blocks(line_regions)
            self.last_grouped_from_lines_count = len(grouped_blocks)
            return grouped_blocks

        return []

__all__ = [
    "PyTorchComicTextDetectorBackend",
    "attach_line_regions_to_blocks",
    "group_line_regions_into_blocks",
]
