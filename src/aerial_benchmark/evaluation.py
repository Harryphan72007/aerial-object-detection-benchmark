from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any

METRIC_NAMES = (
    "map",
    "map_50",
    "map_75",
    "map_small",
    "map_medium",
    "map_large",
)


def evaluate_coco(ground_truth: str | Path, predictions: str | Path) -> dict[str, float]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("Install the 'eval' extra to run COCO evaluation") from exc

    with contextlib.redirect_stdout(io.StringIO()):
        coco_ground_truth = COCO(str(ground_truth))
        coco_predictions = coco_ground_truth.loadRes(str(predictions))
        evaluator = COCOeval(coco_ground_truth, coco_predictions, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return dict(zip(METRIC_NAMES, (float(value) for value in evaluator.stats[:6]), strict=True))


def validate_metrics(metrics: dict[str, Any]) -> None:
    missing = [name for name in METRIC_NAMES if name not in metrics]
    if missing:
        raise ValueError(f"Missing evaluation metrics: {', '.join(missing)}")
    for name in METRIC_NAMES:
        value = metrics[name]
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be a numeric value in [0, 1]")
