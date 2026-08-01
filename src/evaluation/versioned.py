"""Adapters and v2 metric envelopes for legacy and versioned predictions."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

from src.evaluation.detection_metrics import detailed_metrics


def read_prediction_artifact(
    path: str | Path,
    *,
    legacy_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, list):
        if legacy_identity is None:
            raise ValueError("legacy predictions require explicit identity metadata")
        return {
            "schema_version": 0,
            "artifact_type": "legacy_coco_predictions",
            "identity": dict(legacy_identity),
            "benchmark_track": "controlled",
            "inference_mode": "full",
            "weight_variant": "raw",
            "predictions": value,
        }
    if not isinstance(value, dict) or not isinstance(value.get("predictions"), list):
        raise ValueError("unsupported prediction artifact")
    if value.get("inference_mode") == "sliced":
        identity = value.get("identity") or legacy_identity
        if identity is None:
            raise ValueError("sliced predictions require identity metadata")
        return {
            **value,
            "identity": dict(identity),
            "benchmark_track": value.get("benchmark_track", "performance"),
            "weight_variant": value.get("weight_variant", "raw"),
        }
    if value.get("artifact_type") != "coco_predictions":
        raise ValueError("unknown versioned prediction artifact type")
    return {
        **value,
        "benchmark_track": value.get("benchmark_track", "controlled"),
        "inference_mode": value.get("inference_mode", "full"),
        "weight_variant": value.get("weight_variant", "raw"),
    }


def evaluate_prediction_artifact(
    ground_truth_path: str | Path,
    prediction_path: str | Path,
    *,
    legacy_identity: Mapping[str, Any] | None = None,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    artifact = read_prediction_artifact(
        prediction_path, legacy_identity=legacy_identity
    )
    with tempfile.TemporaryDirectory(prefix="visdrone-evaluator-v2-") as temporary:
        normalized = Path(temporary) / "predictions.json"
        normalized.write_text(
            json.dumps(artifact["predictions"]), encoding="utf-8"
        )
        metrics = detailed_metrics(
            ground_truth_path, normalized, iou_threshold=iou_threshold
        )
    return {
        "schema_version": 2,
        "artifact_type": "evaluation_metrics",
        "identity": artifact["identity"],
        "benchmark_track": artifact["benchmark_track"],
        "inference_mode": artifact["inference_mode"],
        "weight_variant": artifact["weight_variant"],
        "metrics": metrics,
    }


def assert_metric_parity(
    first: Any, second: Any, *, tolerance: float = 1e-12, path: str = "metrics"
) -> None:
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        if set(first) != set(second):
            raise AssertionError(f"{path} keys differ")
        for key in first:
            assert_metric_parity(
                first[key], second[key], tolerance=tolerance, path=f"{path}.{key}"
            )
        return
    if isinstance(first, list) and isinstance(second, list):
        if len(first) != len(second):
            raise AssertionError(f"{path} lengths differ")
        for index, (left, right) in enumerate(zip(first, second)):
            assert_metric_parity(
                left, right, tolerance=tolerance, path=f"{path}[{index}]"
            )
        return
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        if abs(float(first) - float(second)) > tolerance:
            raise AssertionError(f"{path} differs: {first} != {second}")
        return
    if first != second:
        raise AssertionError(f"{path} differs: {first!r} != {second!r}")
