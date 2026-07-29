"""TIDE-like error decomposition and VisDrone attribute/density slices."""
from __future__ import annotations

import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.evaluation.coco_evaluator import evaluate_coco
from src.evaluation.detection_metrics import greedy_match, iou_matrix, xywh_to_xyxy


def decompose_errors(
    ground_truth_file: str | Path,
    prediction_file: str | Path,
    match_iou: float = 0.5,
    localization_iou: float = 0.1,
) -> dict[str, Any]:
    """Separate correct, class, localization, duplicate, background, and misses."""
    ground_truth = json.loads(
        Path(ground_truth_file).read_text(encoding="utf-8")
    )
    predictions = json.loads(
        Path(prediction_file).read_text(encoding="utf-8")
    )
    ground_truth_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    predictions_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for annotation in ground_truth["annotations"]:
        ground_truth_by_image[int(annotation["image_id"])].append(annotation)
    for prediction in predictions:
        predictions_by_image[int(prediction["image_id"])].append(prediction)

    for image in ground_truth["images"]:
        image_id = int(image["id"])
        image_ground_truth = ground_truth_by_image[image_id]
        image_predictions = predictions_by_image[image_id]
        matches = greedy_match(
            image_ground_truth, image_predictions, match_iou, True
        )
        matched = {
            int(match["gt_index"])
            for match in matches
            if match["is_tp"]
        }
        for match in matches:
            if match["is_tp"]:
                counts["correct"] += 1
                continue
            prediction_index = int(match["prediction_index"])
            if not image_ground_truth:
                counts["background_false_positive"] += 1
                continue
            prediction_box = np.array(
                [xywh_to_xyxy(image_predictions[prediction_index]["bbox"])]
            )
            ground_truth_boxes = np.array(
                [xywh_to_xyxy(item["bbox"]) for item in image_ground_truth]
            )
            overlaps = iou_matrix(prediction_box, ground_truth_boxes)[0]
            best_index = int(np.argmax(overlaps))
            best_iou = float(overlaps[best_index])
            same_class = int(
                image_predictions[prediction_index]["category_id"]
            ) == int(image_ground_truth[best_index]["category_id"])
            if best_iou >= match_iou and same_class and best_index in matched:
                counts["duplicate_detection"] += 1
            elif best_iou >= match_iou and not same_class:
                counts["classification"] += 1
            elif best_iou >= localization_iou and same_class:
                counts["localization"] += 1
            elif best_iou >= localization_iou:
                counts["classification_and_localization"] += 1
            else:
                counts["background_false_positive"] += 1
        counts["missed_ground_truth"] += len(image_ground_truth) - len(matched)
    return {"error_counts": dict(counts)}


def attribute_slices(
    ground_truth_file: str | Path,
) -> dict[str, dict[str, list[int]]]:
    """Return annotation IDs grouped by raw VisDrone attributes."""
    data = json.loads(Path(ground_truth_file).read_text(encoding="utf-8"))
    output: dict[str, dict[str, list[int]]] = {
        "occlusion": defaultdict(list),
        "truncation": defaultdict(list),
    }
    for annotation in data["annotations"]:
        attributes = annotation.get("attributes", {})
        output["occlusion"][str(attributes.get("occlusion", "unknown"))].append(
            int(annotation["id"])
        )
        output["truncation"][str(attributes.get("truncation", "unknown"))].append(
            int(annotation["id"])
        )
    return {key: dict(value) for key, value in output.items()}


def _evaluate_annotation_slice(
    ground_truth: dict[str, Any],
    predictions: list[dict[str, Any]],
    selector: Callable[[dict[str, Any]], bool],
    directory: Path,
    name: str,
) -> dict[str, Any]:
    selected_annotations = [
        annotation for annotation in ground_truth["annotations"] if selector(annotation)
    ]
    selected_image_ids = {
        int(annotation["image_id"]) for annotation in selected_annotations
    }
    if not selected_annotations:
        return {"annotation_count": 0, "image_count": 0, "status": "empty"}
    payload = dict(ground_truth)
    payload["images"] = [
        image
        for image in ground_truth["images"]
        if int(image["id"]) in selected_image_ids
    ]
    # Non-slice objects in selected images are marked ignored so predictions on
    # them are not incorrectly counted as background false positives.
    payload["annotations"] = []
    for annotation in ground_truth["annotations"]:
        if int(annotation["image_id"]) not in selected_image_ids:
            continue
        item = dict(annotation)
        selected = selector(annotation)
        item["ignore"] = 0 if selected else 1
        item["iscrowd"] = int(item.get("iscrowd", 0)) if selected else 1
        payload["annotations"].append(item)
    prediction_subset = [
        prediction
        for prediction in predictions
        if int(prediction["image_id"]) in selected_image_ids
    ]
    ground_truth_path = directory / f"{name}_gt.json"
    prediction_path = directory / f"{name}_pred.json"
    ground_truth_path.write_text(json.dumps(payload), encoding="utf-8")
    prediction_path.write_text(json.dumps(prediction_subset), encoding="utf-8")
    metrics = evaluate_coco(ground_truth_path, prediction_path)
    metrics.update(
        {
            "annotation_count": len(selected_annotations),
            "image_count": len(selected_image_ids),
            "status": "completed",
        }
    )
    return metrics


def _evaluate_image_slice(
    ground_truth: dict[str, Any],
    predictions: list[dict[str, Any]],
    image_ids: set[int],
    directory: Path,
    name: str,
) -> dict[str, Any]:
    if not image_ids:
        return {"annotation_count": 0, "image_count": 0, "status": "empty"}
    payload = dict(ground_truth)
    payload["images"] = [
        image for image in ground_truth["images"] if int(image["id"]) in image_ids
    ]
    payload["annotations"] = [
        annotation
        for annotation in ground_truth["annotations"]
        if int(annotation["image_id"]) in image_ids
    ]
    prediction_subset = [
        prediction
        for prediction in predictions
        if int(prediction["image_id"]) in image_ids
    ]
    ground_truth_path = directory / f"{name}_gt.json"
    prediction_path = directory / f"{name}_pred.json"
    ground_truth_path.write_text(json.dumps(payload), encoding="utf-8")
    prediction_path.write_text(json.dumps(prediction_subset), encoding="utf-8")
    metrics = evaluate_coco(ground_truth_path, prediction_path)
    metrics.update(
        {
            "annotation_count": len(payload["annotations"]),
            "image_count": len(payload["images"]),
            "status": "completed",
        }
    )
    return metrics


def evaluate_visdrone_slices(
    ground_truth_file: str | Path, prediction_file: str | Path
) -> dict[str, Any]:
    """Evaluate occlusion, truncation, and density using documented slice rules."""
    ground_truth = json.loads(
        Path(ground_truth_file).read_text(encoding="utf-8")
    )
    predictions = json.loads(
        Path(prediction_file).read_text(encoding="utf-8")
    )
    attribute_names = {
        "occlusion": {0: "none", 1: "partial", 2: "heavy"},
        "truncation": {0: "none", 1: "partial", 2: "heavy"},
    }
    counts_by_image: Counter[int] = Counter(
        int(annotation["image_id"]) for annotation in ground_truth["annotations"]
    )
    count_values = np.asarray(
        [counts_by_image[int(image["id"])] for image in ground_truth["images"]],
        dtype=float,
    )
    quartiles = np.quantile(count_values, [0.25, 0.5, 0.75]).tolist()

    with tempfile.TemporaryDirectory(prefix="visdrone_slices_") as temporary:
        directory = Path(temporary)
        output: dict[str, Any] = {
            "slice_definition": (
                "Attribute slices keep selected-image context and mark non-slice "
                "ground truth ignored. Density slices use training/evaluation-set "
                "object-count quartiles."
            ),
            "density_thresholds": {
                "q25": float(quartiles[0]),
                "q50": float(quartiles[1]),
                "q75": float(quartiles[2]),
            },
        }
        for attribute, labels in attribute_names.items():
            output[attribute] = {}
            for raw_value, label in labels.items():
                output[attribute][label] = _evaluate_annotation_slice(
                    ground_truth,
                    predictions,
                    lambda annotation, a=attribute, value=raw_value: int(
                        annotation.get("attributes", {}).get(a, -1)
                    )
                    == value,
                    directory,
                    f"{attribute}_{label}",
                )
        density_groups = {
            "low": {
                int(image["id"])
                for image in ground_truth["images"]
                if counts_by_image[int(image["id"])] <= quartiles[0]
            },
            "medium": {
                int(image["id"])
                for image in ground_truth["images"]
                if quartiles[0]
                < counts_by_image[int(image["id"])]
                <= quartiles[1]
            },
            "high": {
                int(image["id"])
                for image in ground_truth["images"]
                if quartiles[1]
                < counts_by_image[int(image["id"])]
                <= quartiles[2]
            },
            "extremely_dense": {
                int(image["id"])
                for image in ground_truth["images"]
                if counts_by_image[int(image["id"])] > quartiles[2]
            },
        }
        output["density"] = {
            label: _evaluate_image_slice(
                ground_truth, predictions, image_ids, directory, f"density_{label}"
            )
            for label, image_ids in density_groups.items()
        }
    return output


def per_image_error_statistics(
    ground_truth_file: str | Path, prediction_file: str | Path, iou_threshold: float = 0.5
) -> list[dict[str, Any]]:
    """Compute image-level TP/FP/FN and tiny-miss counts for qualitative selection."""
    ground_truth = json.loads(Path(ground_truth_file).read_text(encoding="utf-8"))
    predictions = json.loads(Path(prediction_file).read_text(encoding="utf-8"))
    gt_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    pred_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in ground_truth["annotations"]:
        gt_by_image[int(annotation["image_id"])].append(annotation)
    for prediction in predictions:
        pred_by_image[int(prediction["image_id"])].append(prediction)
    rows: list[dict[str, Any]] = []
    for image in ground_truth["images"]:
        image_id = int(image["id"])
        annotations = gt_by_image[image_id]
        image_predictions = pred_by_image[image_id]
        matches = greedy_match(annotations, image_predictions, iou_threshold, True)
        true_positives = sum(bool(match["is_tp"]) for match in matches)
        matched_gt = {int(match["gt_index"]) for match in matches if match["is_tp"]}
        false_positives = len(image_predictions) - true_positives
        false_negatives = len(annotations) - true_positives
        tiny_missed = sum(
            index not in matched_gt and float(annotation["bbox"][2]) * float(annotation["bbox"][3]) < 16**2
            for index, annotation in enumerate(annotations)
        )
        precision = true_positives / max(1, true_positives + false_positives)
        recall = true_positives / max(1, true_positives + false_negatives)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        rows.append({
            "image_id": image_id,
            "file_name": image.get("file_name", ""),
            "ground_truth_count": len(annotations),
            "prediction_count": len(image_predictions),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "tiny_missed": int(tiny_missed),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return rows


def select_qualitative_images(
    primary_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]] | None = None,
    top_k: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """Select best, worst, tiny-miss, false-positive, and disagreement examples."""
    result = {
        "best_performing": sorted(primary_rows, key=lambda row: (row["f1"], row["ground_truth_count"]), reverse=True)[:top_k],
        "worst_performing": sorted(primary_rows, key=lambda row: (row["f1"], -row["ground_truth_count"]))[:top_k],
        "most_missed_tiny_objects": sorted(primary_rows, key=lambda row: row["tiny_missed"], reverse=True)[:top_k],
        "most_false_positives": sorted(primary_rows, key=lambda row: row["false_positives"], reverse=True)[:top_k],
    }
    if comparison_rows is not None:
        comparison = {int(row["image_id"]): row for row in comparison_rows}
        differences = []
        for row in primary_rows:
            other = comparison.get(int(row["image_id"]))
            if other is None:
                continue
            item = dict(row)
            item["comparison_f1"] = float(other["f1"])
            item["absolute_f1_difference"] = abs(float(row["f1"]) - float(other["f1"]))
            differences.append(item)
        result["largest_model_differences"] = sorted(
            differences, key=lambda row: row["absolute_f1_difference"], reverse=True
        )[:top_k]
    return result
