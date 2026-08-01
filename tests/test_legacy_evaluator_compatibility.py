from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.detection_metrics import detailed_metrics
from src.paths import ProjectPaths
from src.utils.serialization import read_json, write_json, write_yaml
from src.workflows.comparison import compare_completed_models

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "tests" / "fixtures" / "legacy_artifacts"
CASES = ROOT / "tests" / "fixtures" / "legacy_evaluator" / "comparison_cases.json"


def _materialize_cases(paths: ProjectPaths) -> None:
    fixtures = read_json(CASES)
    registry: dict[str, object] = {"schema_version": 1, "runs": {}}
    for case in fixtures["runs"]:
        manifest = dict(case["manifest"])
        run_id = manifest["run_id"]
        model_id = manifest["model_id"]
        run_dir = paths.final_checkpoints / model_id / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest["run_dir"] = str(run_dir)
        write_yaml(run_dir / "training_config.yaml", case["config"])
        write_json(paths.evaluation / f"{run_id}__res640__metrics.json", case["metrics"])
        write_json(paths.evaluation / f"{run_id}__profile.json", case["profile"])
        registry["runs"][run_id] = manifest
    write_json(paths.checkpoint_registry, registry)


def test_legacy_detailed_evaluator_metrics_are_frozen() -> None:
    metrics = detailed_metrics(LEGACY / "ground_truth.json", LEGACY / "predictions.json")
    assert metrics["per_class_detailed"] == {
        "person": {
            "true_positives": 1,
            "false_positives": 1,
            "false_negatives": 0,
            "precision": 0.49999999999975,
            "recall": 0.9999999999989999,
            "F1": 0.6666666666657777,
            "mean_matched_iou": 0.999999999999995,
            "optimal_confidence_threshold": 0.20500000000000002,
            "optimal_F1": 0.9999999999984999,
            "average_confidence": 0.575,
        },
        "vehicle": {
            "true_positives": 1,
            "false_positives": 0,
            "false_negatives": 0,
            "precision": 0.9999999999989999,
            "recall": 0.9999999999989999,
            "F1": 0.9999999999984999,
            "mean_matched_iou": 0.9999999999999966,
            "optimal_confidence_threshold": 0.0,
            "optimal_F1": 0.9999999999984999,
            "average_confidence": 0.9,
        }
    }
    assert metrics["false_positives_per_image"] == 0.5
    assert metrics["tiny_miss_rate"] == 0.0


def test_two_legacy_models_produce_deterministic_rows_and_exclude_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setattr(
        pd.DataFrame,
        "to_markdown",
        lambda self, index=False: self.to_csv(index=index),
    )
    paths = ProjectPaths.from_value(tmp_path).create()
    _materialize_cases(paths)
    first = compare_completed_models(paths.root, tmp_path / "first")
    second = compare_completed_models(paths.root, tmp_path / "second")
    assert first["models"] == second["models"]
    complete = {row["Model"]: row for row in first["models"] if row["status"] == "COMPLETE"}
    assert complete["faster_rcnn_resnet50"]["run_id"].endswith("120000__seed42")
    assert complete["faster_rcnn_resnet50"]["mAP50-95"] == 0.5
    assert complete["rtdetrv2_l"]["mAP50-95"] == 0.6
    assert all("__smoke__" not in row.get("run_id", "") for row in first["models"])
    assert any("__smoke__" in row["run_id"] for row in first["rejected"])
    assert read_json(tmp_path / "first" / "comparison.json")["models"] == first["models"]
