"""MMEngine metric plugin for custom aerial-object size ranges."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

try:
    from mmengine.evaluator import BaseMetric
    from mmdet.registry import METRICS
except ImportError:  # Lightweight tests do not install MMDetection.
    BaseMetric = object  # type: ignore[assignment,misc]
    METRICS = None

from src.evaluation.coco_evaluator import evaluate_coco
from src.evaluation.policy import MAX_DETECTIONS_PER_IMAGE


def _value(sample: Any, key: str, default: Any = None) -> Any:
    if isinstance(sample, dict):
        return sample.get(key, default)
    return getattr(sample, key, default)


def flatten_image_results(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten one-result-per-image metric records for COCO evaluation."""

    detections: list[dict[str, Any]] = []
    for result in results:
        rows = result.get("detections", [])
        if not isinstance(rows, list):
            raise TypeError("AerialCocoMetric detections must be a list")
        detections.extend(rows)
    return detections


if METRICS is not None:

    @METRICS.register_module()
    class AerialCocoMetric(BaseMetric):
        """Compute APtiny/ARtiny from MMDetection predictions.

        Category labels emitted by MMDetection are zero-based; COCO category IDs in
        the converted benchmark are contiguous and one-based.
        """

        default_prefix = "aerial_coco"

        def __init__(
            self,
            ann_file: str,
            collect_device: str = "cpu",
            prefix: str | None = None,
        ) -> None:
            super().__init__(collect_device=collect_device, prefix=prefix)
            self.ann_file = str(Path(ann_file))

        def process(
            self, data_batch: Any, data_samples: Sequence[Any]
        ) -> None:
            for sample in data_samples:
                metainfo = _value(sample, "metainfo", {}) or {}
                image_id = int(
                    metainfo.get("img_id", metainfo.get("image_id", -1))
                )
                predictions = _value(sample, "pred_instances")
                if predictions is None:
                    boxes, scores, labels = (), (), ()
                else:
                    predictions = predictions.to("cpu")
                    boxes = predictions.bboxes.numpy()
                    scores = predictions.scores.numpy()
                    labels = predictions.labels.numpy()
                rows: list[dict[str, Any]] = []
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = map(float, box)
                    rows.append(
                        {
                            "image_id": image_id,
                            "category_id": int(label) + 1,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(score),
                        }
                    )
                self.results.append({"image_id": image_id, "detections": rows})

        def compute_metrics(
            self, results: list[dict[str, Any]]
        ) -> dict[str, float]:
            metrics = evaluate_coco(
                self.ann_file,
                flatten_image_results(results),
                max_detections=[1, 10, 100, 300, MAX_DETECTIONS_PER_IMAGE],
            )
            return {
                "APtiny": float(metrics["APtiny"]),
                "ARtiny": float(metrics["ARtiny"]),
                "APsmall_custom": float(metrics["APsmall"]),
                "APmedium_custom": float(metrics["APmedium"]),
                "APlarge_custom": float(metrics["APlarge"]),
            }
