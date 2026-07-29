"""Detection calibration after explicit one-to-one IoU matching."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.detection_metrics import greedy_match


def _calibration_bins(
    scores: list[float], outcomes: list[float], bins: int
) -> dict[str, Any]:
    score_array = np.asarray(scores, dtype=float)
    outcome_array = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    expected_error = 0.0
    maximum_error = 0.0
    for index in range(bins):
        upper = edges[index + 1]
        mask = (score_array >= edges[index]) & (
            score_array < (upper if index < bins - 1 else upper + 1e-9)
        )
        if not mask.any():
            rows.append(
                {
                    "lower": float(edges[index]),
                    "upper": float(upper),
                    "count": 0,
                    "confidence": None,
                    "accuracy": None,
                }
            )
            continue
        confidence = float(score_array[mask].mean())
        accuracy = float(outcome_array[mask].mean())
        gap = abs(confidence - accuracy)
        expected_error += float(mask.mean()) * gap
        maximum_error = max(maximum_error, gap)
        rows.append(
            {
                "lower": float(edges[index]),
                "upper": float(upper),
                "count": int(mask.sum()),
                "confidence": confidence,
                "accuracy": accuracy,
            }
        )
    return {
        "ECE": float(expected_error),
        "MCE": float(maximum_error),
        "brier_style_detection_score": float(
            np.mean((score_array - outcome_array) ** 2)
        )
        if len(score_array)
        else 0.0,
        "bins": rows,
        "sample_count": len(scores),
    }


def detection_calibration(
    ground_truth_file: str | Path,
    prediction_file: str | Path,
    iou_threshold: float = 0.5,
    bins: int = 15,
) -> dict[str, Any]:
    """Calibrate detection confidence against class-aware matched precision."""
    ground_truth = json.loads(
        Path(ground_truth_file).read_text(encoding="utf-8")
    )
    predictions = json.loads(
        Path(prediction_file).read_text(encoding="utf-8")
    )
    ground_truth_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    predictions_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in ground_truth["annotations"]:
        ground_truth_by_image[int(annotation["image_id"])].append(annotation)
    for prediction in predictions:
        predictions_by_image[int(prediction["image_id"])].append(prediction)
    categories = {
        int(category["id"]): str(category["name"])
        for category in ground_truth["categories"]
    }
    scores: list[float] = []
    outcomes: list[float] = []
    class_groups: dict[str, tuple[list[float], list[float]]] = defaultdict(
        lambda: ([], [])
    )
    size_groups: dict[str, tuple[list[float], list[float]]] = defaultdict(
        lambda: ([], [])
    )
    for image in ground_truth["images"]:
        image_id = int(image["id"])
        image_ground_truth = ground_truth_by_image[image_id]
        matches = greedy_match(
            image_ground_truth,
            predictions_by_image[image_id],
            iou_threshold,
            True,
        )
        for match in matches:
            score = float(match["score"])
            outcome = 1.0 if match["is_tp"] else 0.0
            scores.append(score)
            outcomes.append(outcome)
            class_name = categories[int(match["pred_category"])]
            class_groups[class_name][0].append(score)
            class_groups[class_name][1].append(outcome)
            if match["is_tp"]:
                annotation = image_ground_truth[int(match["gt_index"])]
                area = float(
                    annotation.get(
                        "area", annotation["bbox"][2] * annotation["bbox"][3]
                    )
                )
                size_name = (
                    "tiny"
                    if area < 256
                    else "small"
                    if area < 1024
                    else "medium"
                    if area < 9216
                    else "large"
                )
            else:
                size_name = "background"
            size_groups[size_name][0].append(score)
            size_groups[size_name][1].append(outcome)
    result = _calibration_bins(scores, outcomes, bins)
    result["matching_definition"] = (
        f"greedy one-to-one, class-aware, IoU >= {iou_threshold}; "
        "a detection is correct only when matched to an unused same-class GT"
    )
    result["by_class"] = {
        name: _calibration_bins(group_scores, group_outcomes, bins)
        for name, (group_scores, group_outcomes) in class_groups.items()
    }
    result["by_size"] = {
        name: _calibration_bins(group_scores, group_outcomes, bins)
        for name, (group_scores, group_outcomes) in size_groups.items()
    }
    return result
