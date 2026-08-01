"""Sliced inference coordinate restoration and class-aware merging."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from src.data.tiling import TileWindow, tile_windows


def restore_prediction(
    prediction: Mapping[str, Any],
    tile: TileWindow,
    *,
    image_id: int,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    x, y, width, height = (float(value) for value in prediction["bbox"])
    left = min(max(x + tile.x, 0.0), float(image_width))
    top = min(max(y + tile.y, 0.0), float(image_height))
    right = min(max(x + width + tile.x, left), float(image_width))
    bottom = min(max(y + height + tile.y, top), float(image_height))
    return {
        **dict(prediction),
        "image_id": image_id,
        "bbox": [left, top, right - left, bottom - top],
        "slice_origin": [tile.x, tile.y],
        "slice_size": [tile.width, tile.height],
    }


def _iou(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    ax, ay, aw, ah = (float(value) for value in first["bbox"])
    bx, by, bw, bh = (float(value) for value in second["bbox"])
    intersection_width = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    intersection_height = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    intersection = intersection_width * intersection_height
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def merge_predictions(
    predictions: Iterable[Mapping[str, Any]], *, iou_threshold: float
) -> list[dict[str, Any]]:
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")
    ordered = sorted(
        (dict(prediction) for prediction in predictions),
        key=lambda value: (-float(value.get("score", 0.0)), int(value["category_id"])),
    )
    kept: list[dict[str, Any]] = []
    for candidate in ordered:
        duplicate = any(
            int(candidate["image_id"]) == int(selected["image_id"])
            and int(candidate["category_id"]) == int(selected["category_id"])
            and _iou(candidate, selected) > iou_threshold
            for selected in kept
        )
        if not duplicate:
            kept.append(candidate)
    return kept


def sliced_output_paths(root: str | Path, run_id: str) -> dict[str, Path]:
    base = Path(root)
    return {
        "predictions": base / "predictions" / "sliced" / run_id / "predictions.json",
        "metrics": base / "evaluation" / "sliced" / run_id / "metrics.json",
    }


def run_sliced_inference(
    *,
    image_id: int,
    image_width: int,
    image_height: int,
    tile_size: int,
    overlap: int,
    iou_threshold: float,
    predict_slice: Callable[[TileWindow], Iterable[Mapping[str, Any]]],
) -> dict[str, Any]:
    restored: list[dict[str, Any]] = []
    model_seconds = 0.0
    windows = tile_windows(
        image_width, image_height, tile_size=tile_size, overlap=overlap
    )
    started = time.perf_counter()
    for tile in windows:
        slice_started = time.perf_counter()
        local = tuple(predict_slice(tile))
        model_seconds += time.perf_counter() - slice_started
        restored.extend(
            restore_prediction(
                prediction,
                tile,
                image_id=image_id,
                image_width=image_width,
                image_height=image_height,
            )
            for prediction in local
        )
    merge_started = time.perf_counter()
    merged = merge_predictions(restored, iou_threshold=iou_threshold)
    merge_seconds = time.perf_counter() - merge_started
    return {
        "schema_version": 1,
        "inference_mode": "sliced",
        "image_id": image_id,
        "predictions": merged,
        "latency": {
            "slice_count": len(windows),
            "slice_model_seconds": model_seconds,
            "merge_seconds": merge_seconds,
            "total_seconds": time.perf_counter() - started,
        },
    }
