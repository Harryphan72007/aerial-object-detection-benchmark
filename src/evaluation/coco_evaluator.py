"""COCO evaluation including custom aerial size ranges."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

CUSTOM_AREA_RANGES = {
    "all": [0, 1e10],
    "tiny": [0, 16**2],
    "small": [16**2, 32**2],
    "medium": [32**2, 96**2],
    "large": [96**2, 1e10],
}


def _summarize_precision(
    coco_eval: Any,
    area_label: str,
    max_dets: int = 100,
    iou: float | None = None,
) -> float:
    precision = coco_eval.eval["precision"]
    if iou is not None:
        indices = np.where(np.isclose(coco_eval.params.iouThrs, iou))[0]
        if indices.size == 0:
            return float("nan")
        precision = precision[indices]
    area_index = coco_eval.params.areaRngLbl.index(area_label)
    max_index = coco_eval.params.maxDets.index(max_dets)
    values = precision[:, :, :, area_index, max_index]
    values = values[values > -1]
    return float(values.mean()) if values.size else float("nan")


def _summarize_recall(
    coco_eval: Any, area_label: str, max_dets: int = 100
) -> float:
    recall = coco_eval.eval["recall"]
    area_index = coco_eval.params.areaRngLbl.index(area_label)
    max_index = coco_eval.params.maxDets.index(max_dets)
    values = recall[:, :, area_index, max_index]
    values = values[values > -1]
    return float(values.mean()) if values.size else float("nan")


def _category_ap(
    precision: np.ndarray,
    category_index: int,
    area_index: int,
    max_index: int,
    iou_index: int | None = None,
) -> float:
    values = (
        precision[:, :, category_index, area_index, max_index]
        if iou_index is None
        else precision[iou_index, :, category_index, area_index, max_index]
    )
    values = values[values > -1]
    return float(values.mean()) if values.size else float("nan")


def evaluate_coco(
    ground_truth: str | Path,
    predictions: str | Path | list[dict[str, Any]],
    max_detections: list[int] | None = None,
) -> dict[str, Any]:
    """Evaluate COCO predictions with standard and custom aerial size bins."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(ground_truth))
    prediction_data = (
        json.loads(Path(predictions).read_text(encoding="utf-8"))
        if isinstance(predictions, (str, Path))
        else predictions
    )
    if not prediction_data:
        category_names = [item["name"] for item in coco_gt.dataset["categories"]]
        return {
            "mAP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "APtiny": 0.0,
            "APsmall": 0.0,
            "APmedium": 0.0,
            "APlarge": 0.0,
            "ARtiny": 0.0,
            "ARsmall": 0.0,
            "ARmedium": 0.0,
            "ARlarge": 0.0,
            "prediction_count": 0,
            "per_class": {
                name: {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
                for name in category_names
            },
        }

    coco_dt = coco_gt.loadRes(prediction_data)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.maxDets = sorted(
        set(max_detections or [1, 10, 100, 300, 500])
    )
    evaluator.params.areaRng = list(CUSTOM_AREA_RANGES.values())
    evaluator.params.areaRngLbl = list(CUSTOM_AREA_RANGES)
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()

    largest = max(evaluator.params.maxDets)
    metrics: dict[str, Any] = {
        "mAP": _summarize_precision(evaluator, "all", largest),
        "AP50": _summarize_precision(evaluator, "all", largest, 0.5),
        "AP75": _summarize_precision(evaluator, "all", largest, 0.75),
        "APtiny": _summarize_precision(evaluator, "tiny", largest),
        "APsmall": _summarize_precision(evaluator, "small", largest),
        "APmedium": _summarize_precision(evaluator, "medium", largest),
        "APlarge": _summarize_precision(evaluator, "large", largest),
        "ARtiny": _summarize_recall(evaluator, "tiny", largest),
        "ARsmall": _summarize_recall(evaluator, "small", largest),
        "ARmedium": _summarize_recall(evaluator, "medium", largest),
        "ARlarge": _summarize_recall(evaluator, "large", largest),
        "prediction_count": len(prediction_data),
    }
    for detections in evaluator.params.maxDets:
        metrics[f"AR{detections}"] = _summarize_recall(
            evaluator, "all", detections
        )

    category_names = {
        int(category["id"]): str(category["name"])
        for category in coco_gt.dataset["categories"]
    }
    precision = evaluator.eval["precision"]
    area_index = evaluator.params.areaRngLbl.index("all")
    max_index = evaluator.params.maxDets.index(largest)
    iou50 = int(np.where(np.isclose(evaluator.params.iouThrs, 0.5))[0][0])
    iou75 = int(np.where(np.isclose(evaluator.params.iouThrs, 0.75))[0][0])
    per_class: dict[str, dict[str, float]] = {}
    for category_index, category_id in enumerate(evaluator.params.catIds):
        per_class[category_names[int(category_id)]] = {
            "AP": _category_ap(
                precision, category_index, area_index, max_index
            ),
            "AP50": _category_ap(
                precision, category_index, area_index, max_index, iou50
            ),
            "AP75": _category_ap(
                precision, category_index, area_index, max_index, iou75
            ),
        }
    metrics["per_class"] = per_class
    return metrics


def paired_bootstrap_map_difference(
    ground_truth: str | Path,
    predictions_a: str | Path | list[dict[str, Any]],
    predictions_b: str | Path | list[dict[str, Any]],
    samples: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Estimate a paired image-bootstrap confidence interval for an mAP difference.

    Images are sampled with replacement. Repeated images receive new image and
    annotation IDs, and both models use the identical resample, preserving the
    paired comparison. This can be expensive; use fewer samples for smoke tests.
    """
    import tempfile

    if samples <= 0:
        raise ValueError("samples must be positive")
    ground_truth_data = json.loads(Path(ground_truth).read_text(encoding="utf-8"))

    def load_predictions(value: str | Path | list[dict[str, Any]]) -> list[dict[str, Any]]:
        return (
            json.loads(Path(value).read_text(encoding="utf-8"))
            if isinstance(value, (str, Path))
            else value
        )

    prediction_sets = [load_predictions(predictions_a), load_predictions(predictions_b)]
    images = ground_truth_data["images"]
    image_ids = [int(image["id"]) for image in images]
    image_lookup = {int(image["id"]): image for image in images}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in image_ids}
    predictions_by_model: list[dict[int, list[dict[str, Any]]]] = []
    for annotation in ground_truth_data["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    for prediction_set in prediction_sets:
        grouped: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in image_ids}
        for prediction in prediction_set:
            if int(prediction["image_id"]) in grouped:
                grouped[int(prediction["image_id"])].append(prediction)
        predictions_by_model.append(grouped)

    rng = np.random.default_rng(seed)
    differences: list[float] = []
    with tempfile.TemporaryDirectory(prefix="paired_map_bootstrap_") as temporary:
        directory = Path(temporary)
        for sample_index in range(samples):
            chosen = rng.choice(image_ids, size=len(image_ids), replace=True)
            sampled_gt = {
                "info": ground_truth_data.get("info", {}),
                "licenses": ground_truth_data.get("licenses", []),
                "categories": ground_truth_data["categories"],
                "images": [],
                "annotations": [],
            }
            sampled_predictions = [[], []]
            annotation_id = 1
            for new_image_id, original_image_id in enumerate(chosen, start=1):
                original_image_id = int(original_image_id)
                image = dict(image_lookup[original_image_id])
                image["id"] = new_image_id
                sampled_gt["images"].append(image)
                for annotation in annotations_by_image[original_image_id]:
                    item = dict(annotation)
                    item.update({"id": annotation_id, "image_id": new_image_id})
                    annotation_id += 1
                    sampled_gt["annotations"].append(item)
                for model_index, grouped in enumerate(predictions_by_model):
                    for prediction in grouped[original_image_id]:
                        item = dict(prediction)
                        item["image_id"] = new_image_id
                        sampled_predictions[model_index].append(item)
            gt_path = directory / f"gt_{sample_index}.json"
            gt_path.write_text(json.dumps(sampled_gt), encoding="utf-8")
            map_a = evaluate_coco(gt_path, sampled_predictions[0])["mAP"]
            map_b = evaluate_coco(gt_path, sampled_predictions[1])["mAP"]
            differences.append(float(map_a) - float(map_b))
    values = np.asarray(differences, dtype=float)
    return {
        "mean_difference": float(values.mean()),
        "standard_deviation": float(values.std(ddof=1)) if samples > 1 else 0.0,
        "ci95_lower": float(np.quantile(values, 0.025)),
        "ci95_upper": float(np.quantile(values, 0.975)),
        "probability_a_better": float(np.mean(values > 0)),
        "samples": float(samples),
    }
