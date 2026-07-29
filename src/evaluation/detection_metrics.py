"""Detection matching, per-class metrics, confidence curves, and localization errors."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def xywh_to_xyxy(box: list[float]) -> np.ndarray:
    x, y, width, height = map(float, box)
    return np.array([x, y, x + width, y + height], dtype=float)


def iou_matrix(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if len(first) == 0 or len(second) == 0:
        return np.zeros((len(first), len(second)))
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    width_height = np.clip(bottom_right - top_left, 0, None)
    intersection = width_height[..., 0] * width_height[..., 1]
    first_area = (first[:, 2] - first[:, 0]) * (first[:, 3] - first[:, 1])
    second_area = (second[:, 2] - second[:, 0]) * (second[:, 3] - second[:, 1])
    return intersection / (
        first_area[:, None] + second_area[None, :] - intersection + 1e-12
    )


def greedy_match(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float = 0.5,
    class_aware: bool = True,
) -> list[dict[str, Any]]:
    """Greedily match predictions by descending confidence, one GT at most once."""
    ground_truth_boxes = (
        np.array([xywh_to_xyxy(item["bbox"]) for item in ground_truth])
        if ground_truth
        else np.zeros((0, 4))
    )
    prediction_boxes = (
        np.array([xywh_to_xyxy(item["bbox"]) for item in predictions])
        if predictions
        else np.zeros((0, 4))
    )
    overlaps = iou_matrix(prediction_boxes, ground_truth_boxes)
    order = np.argsort([-float(item["score"]) for item in predictions])
    used: set[int] = set()
    rows: list[dict[str, Any]] = []
    for prediction_index in order:
        candidates = (
            np.argsort(-overlaps[prediction_index]) if ground_truth else []
        )
        match = None
        for ground_truth_index in candidates:
            ground_truth_index = int(ground_truth_index)
            if ground_truth_index in used:
                continue
            if class_aware and int(
                predictions[prediction_index]["category_id"]
            ) != int(ground_truth[ground_truth_index]["category_id"]):
                continue
            if overlaps[prediction_index, ground_truth_index] >= iou_threshold:
                match = ground_truth_index
                break
        if match is not None:
            used.add(match)
        rows.append(
            {
                "prediction_index": int(prediction_index),
                "gt_index": match,
                "score": float(predictions[prediction_index]["score"]),
                "pred_category": int(
                    predictions[prediction_index]["category_id"]
                ),
                "gt_category": int(ground_truth[match]["category_id"])
                if match is not None
                else None,
                "iou": float(overlaps[prediction_index, match])
                if match is not None
                else 0.0,
                "is_tp": match is not None,
            }
        )
    return rows


def _size_group(area: float) -> str:
    if area < 16**2:
        return "tiny"
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def _optimal_threshold(
    scored_outcomes: list[tuple[float, bool]], total_ground_truth: int
) -> tuple[float, float]:
    if not scored_outcomes:
        return 0.0, 0.0
    best_threshold = 0.0
    best_f1 = 0.0
    for threshold in np.linspace(0, 1, 201):
        selected = [item for item in scored_outcomes if item[0] >= threshold]
        true_positives = sum(outcome for _, outcome in selected)
        false_positives = len(selected) - true_positives
        false_negatives = max(0, total_ground_truth - true_positives)
        precision = true_positives / (true_positives + false_positives + 1e-12)
        recall = true_positives / (true_positives + false_negatives + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def detailed_metrics(
    ground_truth_file: str | Path,
    prediction_file: str | Path,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Calculate thresholded, localization, confusion, and custom size metrics."""
    ground_truth_data = json.loads(
        Path(ground_truth_file).read_text(encoding="utf-8")
    )
    predictions = json.loads(
        Path(prediction_file).read_text(encoding="utf-8")
    )
    ground_truth_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    predictions_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in ground_truth_data["annotations"]:
        ground_truth_by_image[int(annotation["image_id"])].append(annotation)
    for prediction in predictions:
        predictions_by_image[int(prediction["image_id"])].append(prediction)

    categories = {
        int(category["id"]): str(category["name"])
        for category in ground_truth_data["categories"]
    }
    totals: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "scores": [],
            "ious": [],
            "outcomes": [],
            "gt_count": 0,
        }
    )
    size_totals = {
        name: {"gt": 0, "matched": 0} for name in ("tiny", "small", "medium", "large")
    }
    all_ious: list[float] = []
    localization: list[dict[str, float]] = []
    false_positive_count = 0
    labels = [*categories, 0]
    confusion = {true: {pred: 0 for pred in labels} for true in labels}

    for image in ground_truth_data["images"]:
        image_id = int(image["id"])
        ground_truth = ground_truth_by_image[image_id]
        image_predictions = predictions_by_image[image_id]
        for item in ground_truth:
            category = int(item["category_id"])
            totals[category]["gt_count"] += 1
            size_totals[_size_group(float(item.get("area", item["bbox"][2] * item["bbox"][3])))]["gt"] += 1

        matches = greedy_match(
            ground_truth, image_predictions, iou_threshold, True
        )
        matched_ground_truth: set[int] = set()
        for match in matches:
            category = match["pred_category"]
            totals[category]["scores"].append(match["score"])
            totals[category]["outcomes"].append(
                (match["score"], bool(match["is_tp"]))
            )
            if match["is_tp"]:
                totals[category]["tp"] += 1
                matched_ground_truth.add(int(match["gt_index"]))
                totals[category]["ious"].append(match["iou"])
                all_ious.append(match["iou"])
                selected_gt = ground_truth[int(match["gt_index"])]
                size_totals[
                    _size_group(
                        float(
                            selected_gt.get(
                                "area",
                                selected_gt["bbox"][2] * selected_gt["bbox"][3],
                            )
                        )
                    )
                ]["matched"] += 1
                predicted_box = xywh_to_xyxy(
                    image_predictions[match["prediction_index"]]["bbox"]
                )
                true_box = xywh_to_xyxy(selected_gt["bbox"])
                predicted_center = (predicted_box[:2] + predicted_box[2:]) / 2
                true_center = (true_box[:2] + true_box[2:]) / 2
                center_error = float(
                    np.linalg.norm(predicted_center - true_center)
                )
                diagonal = np.linalg.norm([image["width"], image["height"]])
                localization.append(
                    {
                        "iou": match["iou"],
                        "center_error": center_error,
                        "normalized_center_error": center_error / max(diagonal, 1e-12),
                        "width_error": float(
                            (predicted_box[2] - predicted_box[0])
                            - (true_box[2] - true_box[0])
                        ),
                        "height_error": float(
                            (predicted_box[3] - predicted_box[1])
                            - (true_box[3] - true_box[1])
                        ),
                        "area_ratio": float(
                            (
                                (predicted_box[2] - predicted_box[0])
                                * (predicted_box[3] - predicted_box[1])
                            )
                            / (
                                (true_box[2] - true_box[0])
                                * (true_box[3] - true_box[1])
                                + 1e-12
                            )
                        ),
                    }
                )
            else:
                totals[category]["fp"] += 1
                false_positive_count += 1
        for ground_truth_index, item in enumerate(ground_truth):
            if ground_truth_index not in matched_ground_truth:
                totals[int(item["category_id"])]["fn"] += 1

        # Class-agnostic matching creates a true-vs-predicted confusion table.
        agnostic_matches = greedy_match(
            ground_truth, image_predictions, iou_threshold, False
        )
        agnostic_matched: set[int] = set()
        for match in agnostic_matches:
            if match["is_tp"]:
                true_category = int(match["gt_category"])
                predicted_category = int(match["pred_category"])
                confusion[true_category][predicted_category] += 1
                agnostic_matched.add(int(match["gt_index"]))
            else:
                confusion[0][int(match["pred_category"])] += 1
        for ground_truth_index, item in enumerate(ground_truth):
            if ground_truth_index not in agnostic_matched:
                confusion[int(item["category_id"])][0] += 1

    per_class: dict[str, dict[str, Any]] = {}
    for category_id, name in categories.items():
        values = totals[category_id]
        precision = values["tp"] / (values["tp"] + values["fp"] + 1e-12)
        recall = values["tp"] / (values["tp"] + values["fn"] + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        threshold, best_f1 = _optimal_threshold(
            values["outcomes"], values["gt_count"]
        )
        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "F1": float(f1),
            "optimal_confidence_threshold": threshold,
            "optimal_F1": best_f1,
            "true_positives": values["tp"],
            "false_positives": values["fp"],
            "false_negatives": values["fn"],
            "average_confidence": float(np.mean(values["scores"]))
            if values["scores"]
            else 0.0,
            "mean_matched_iou": float(np.mean(values["ious"]))
            if values["ious"]
            else 0.0,
        }

    normalized_confusion: dict[str, dict[str, float]] = {}
    confusion_named: dict[str, dict[str, int]] = {}
    name_for = {0: "background", **categories}
    for true_category, row in confusion.items():
        total = sum(row.values())
        confusion_named[name_for[true_category]] = {
            name_for[predicted]: int(count) for predicted, count in row.items()
        }
        normalized_confusion[name_for[true_category]] = {
            name_for[predicted]: float(count / total) if total else 0.0
            for predicted, count in row.items()
        }

    size_metrics = {
        name: {
            "ground_truth": values["gt"],
            "matched": values["matched"],
            "recall": values["matched"] / max(values["gt"], 1),
            "miss_rate": (values["gt"] - values["matched"])
            / max(values["gt"], 1),
        }
        for name, values in size_totals.items()
    }
    localization_summary: dict[str, float] = {}
    for key in (
        "center_error",
        "normalized_center_error",
        "width_error",
        "height_error",
        "area_ratio",
    ):
        values = [item[key] for item in localization]
        localization_summary[f"mean_{key}"] = (
            float(np.mean(values)) if values else 0.0
        )
        localization_summary[f"median_{key}"] = (
            float(np.median(values)) if values else 0.0
        )

    image_count = max(1, len(ground_truth_data["images"]))
    return {
        "per_class_detailed": per_class,
        "mean_matched_iou": float(np.mean(all_ious)) if all_ious else 0.0,
        "median_matched_iou": float(np.median(all_ious)) if all_ious else 0.0,
        "matched_iou_distribution": all_ious,
        "localization_errors": localization,
        "localization_summary": localization_summary,
        "custom_size_metrics": size_metrics,
        "tiny_miss_rate": size_metrics["tiny"]["miss_rate"],
        "false_positives_per_image": false_positive_count / image_count,
        "detections_per_image": len(predictions) / image_count,
        "ground_truth_objects_per_image": len(ground_truth_data["annotations"])
        / image_count,
        "confusion_matrix": confusion_named,
        "normalized_confusion_matrix": normalized_confusion,
        "background_false_positive_count": sum(confusion[0].values()),
        "missed_detection_count": sum(row[0] for key, row in confusion.items() if key != 0),
    }


def confidence_curves(
    ground_truth_file: str | Path,
    prediction_file: str | Path,
    iou_threshold: float = 0.5,
    steps: int = 101,
) -> dict[str, list[float]]:
    """Calculate global precision, recall, and F1 versus confidence."""
    ground_truth = json.loads(Path(ground_truth_file).read_text(encoding="utf-8"))
    predictions = json.loads(Path(prediction_file).read_text(encoding="utf-8"))
    thresholds = np.linspace(0, 1, steps)
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    ground_truth_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    predictions_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in ground_truth["annotations"]:
        ground_truth_by_image[int(annotation["image_id"])].append(annotation)
    for prediction in predictions:
        predictions_by_image[int(prediction["image_id"])].append(prediction)
    total_ground_truth = len(ground_truth["annotations"])
    for threshold in thresholds:
        true_positives = 0
        false_positives = 0
        for image in ground_truth["images"]:
            image_id = int(image["id"])
            selected = [
                prediction
                for prediction in predictions_by_image[image_id]
                if float(prediction["score"]) >= threshold
            ]
            matches = greedy_match(
                ground_truth_by_image[image_id],
                selected,
                iou_threshold,
                True,
            )
            true_positives += sum(bool(match["is_tp"]) for match in matches)
            false_positives += sum(
                not bool(match["is_tp"]) for match in matches
            )
        precision = true_positives / (
            true_positives + false_positives + 1e-12
        )
        recall = true_positives / (total_ground_truth + 1e-12)
        precision_values.append(float(precision))
        recall_values.append(float(recall))
        f1_values.append(float(2 * precision * recall / (precision + recall + 1e-12)))
    return {
        "threshold": thresholds.tolist(),
        "precision": precision_values,
        "recall": recall_values,
        "f1": f1_values,
    }
