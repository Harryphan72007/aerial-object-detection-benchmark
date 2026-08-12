"""MMEngine metric plugin for custom aerial-object size ranges."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

try:
    from mmengine.evaluator import BaseMetric
    from mmdet.registry import METRICS
except ImportError as error:  # Lightweight tests do not install MMDetection.
    BaseMetric = object  # type: ignore[assignment,misc]
    METRICS = None
    IMPORT_ERROR: ImportError | None = error
else:
    IMPORT_ERROR = None

from src.evaluation.coco_evaluator import evaluate_coco
from src.evaluation.policy import MAX_DETECTIONS_PER_IMAGE

METRIC_NAME = "AerialCocoMetric"
METRIC_MODULE = "src.evaluation.mmdet_aerial_metric"


def _value(sample: Any, key: str, default: Any = None) -> Any:
    if isinstance(sample, dict):
        return sample.get(key, default)
    return getattr(sample, key, default)


def _to_array(field: Any) -> Any:
    """Detach one prediction field from its device without assuming a backend."""

    if hasattr(field, "cpu"):
        field = field.cpu()
    if hasattr(field, "numpy"):
        field = field.numpy()
    return field


def image_id_from_sample(sample: Any) -> int:
    """Read the COCO image id from one MMDetection data sample.

    ``Evaluator.process`` hands metrics ``DetDataSample.to_dict()`` output, which
    flattens metainfo onto the top level, so ``img_id`` is a plain key rather
    than a nested ``metainfo`` entry. Callers that pass the data element itself
    still work through the nested lookup.
    """

    holders = [sample, _value(sample, "metainfo", {}) or {}]
    for holder in holders:
        for key in ("img_id", "image_id"):
            found = _value(holder, key)
            if found is not None:
                return int(found)
    raise KeyError(
        f"{METRIC_NAME} received a data sample without img_id; MMDetection "
        "validation samples must carry the COCO image id"
    )


def detection_rows(sample: Any, image_id: int) -> list[dict[str, Any]]:
    """Convert one data sample's predictions into COCO detection rows.

    Category labels emitted by MMDetection are zero-based; COCO category IDs in
    the converted benchmark are contiguous and one-based.
    """

    predictions = _value(sample, "pred_instances")
    if predictions is None:
        return []
    fields = []
    for name in ("bboxes", "scores", "labels"):
        field = _value(predictions, name)
        if field is None:
            raise KeyError(
                f"{METRIC_NAME} received pred_instances without {name}"
            )
        fields.append(_to_array(field))
    boxes, scores, labels = fields
    rows: list[dict[str, Any]] = []
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = (float(value) for value in box)
        rows.append(
            {
                "image_id": image_id,
                "category_id": int(label) + 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            }
        )
    return rows


def image_record(sample: Any) -> dict[str, Any]:
    """Bundle one image's detections so empty images survive result collection."""

    image_id = image_id_from_sample(sample)
    return {"image_id": image_id, "detections": detection_rows(sample, image_id)}


def flatten_image_results(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten one-result-per-image metric records for COCO evaluation."""

    detections: list[dict[str, Any]] = []
    for result in results:
        rows = result.get("detections", [])
        if not isinstance(rows, list):
            raise TypeError("AerialCocoMetric detections must be a list")
        detections.extend(rows)
    return detections


def ensure_registered() -> None:
    """Fail loudly when the metric is not usable in this interpreter.

    A silent registry miss surfaces much later as an opaque MMEngine
    ``KeyError`` while the validation loop is built, so the training backend
    checks the registration up front instead.
    """

    if METRICS is None:
        raise RuntimeError(
            f"{METRIC_NAME} needs mmdet and mmengine, but importing "
            "mmdet.registry failed in this interpreter"
        ) from IMPORT_ERROR
    if METRICS.get(METRIC_NAME) is None:
        raise RuntimeError(
            f"{METRIC_NAME} did not register in the mmdet metric registry; "
            f"import {METRIC_MODULE} before building the MMEngine Runner"
        )


if METRICS is not None:

    @METRICS.register_module()
    class AerialCocoMetric(BaseMetric):
        """Compute APtiny/ARtiny from MMDetection predictions."""

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
                self.results.append(image_record(sample))

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
