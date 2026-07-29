"""Resolution, tiled-inference, and prediction-merging helpers."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class Tile:
    """A rectangular crop in full-image coordinates."""

    x: int
    y: int
    width: int
    height: int


def generate_tiles(
    width: int, height: int, tile_size: int, overlap: float = 0.2
) -> list[Tile]:
    """Cover an image with overlapping square crops, including edge-aligned crops."""
    if width <= 0 or height <= 0 or tile_size <= 0:
        raise ValueError("width, height, and tile_size must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    step = max(1, int(tile_size * (1 - overlap)))
    final_x = max(0, width - tile_size)
    final_y = max(0, height - tile_size)
    xs = list(range(0, final_x + 1, step)) or [0]
    ys = list(range(0, final_y + 1, step)) or [0]
    if xs[-1] != final_x:
        xs.append(final_x)
    if ys[-1] != final_y:
        ys.append(final_y)
    return [
        Tile(x, y, min(tile_size, width - x), min(tile_size, height - y))
        for y in sorted(set(ys))
        for x in sorted(set(xs))
    ]


def _xywh_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[0] + box[2], boxes[:, 0] + boxes[:, 2])
    y2 = np.minimum(box[1] + box[3], boxes[:, 1] + boxes[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    union = box[2] * box[3] + boxes[:, 2] * boxes[:, 3] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def class_aware_nms(
    predictions: list[dict[str, Any]], iou_threshold: float = 0.5
) -> list[dict[str, Any]]:
    """Apply deterministic class-aware NMS to COCO-style predictions."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")
    kept: list[dict[str, Any]] = []
    categories = sorted({int(item["category_id"]) for item in predictions})
    for category_id in categories:
        group = [item for item in predictions if int(item["category_id"]) == category_id]
        group.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        while group:
            current = group.pop(0)
            kept.append(current)
            if not group:
                break
            boxes = np.asarray([item["bbox"] for item in group], dtype=float)
            overlaps = _xywh_iou(np.asarray(current["bbox"], dtype=float), boxes)
            group = [item for item, overlap in zip(group, overlaps) if overlap <= iou_threshold]
    return sorted(kept, key=lambda item: float(item.get("score", 0.0)), reverse=True)


def merge_tiled_predictions(
    tiled_predictions: Iterable[tuple[Tile, list[dict[str, Any]]]],
    image_width: int,
    image_height: int,
    image_id: int,
    nms_iou: float = 0.5,
    max_detections: int | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Map tile-local boxes to the full image and merge overlaps.

    Returns the merged COCO predictions and merging latency in milliseconds, so
    tiled inference can report its full end-to-end cost rather than forward time only.
    """
    started = time.perf_counter()
    global_predictions: list[dict[str, Any]] = []
    for tile, predictions in tiled_predictions:
        for prediction in predictions:
            x, y, width, height = map(float, prediction["bbox"])
            x = min(max(x + tile.x, 0.0), float(image_width))
            y = min(max(y + tile.y, 0.0), float(image_height))
            width = min(max(width, 0.0), max(0.0, image_width - x))
            height = min(max(height, 0.0), max(0.0, image_height - y))
            if width <= 0 or height <= 0:
                continue
            item = dict(prediction)
            item.update({"image_id": int(image_id), "bbox": [x, y, width, height]})
            global_predictions.append(item)
    merged = class_aware_nms(global_predictions, nms_iou)
    if max_detections is not None:
        merged = merged[: int(max_detections)]
    return merged, (time.perf_counter() - started) * 1000.0
