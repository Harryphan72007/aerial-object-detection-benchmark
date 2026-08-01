from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.detection_metrics import detailed_metrics
from src.evaluation.versioned import assert_metric_parity, evaluate_prediction_artifact
from src.utils.serialization import read_json
from src.workflows.versioned_comparison import (
    build_comparison_tables,
    write_comparison_tables,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "legacy_artifacts"
IDENTITY = {
    "run_id": "fixture-run",
    "model_id": "rtdetrv2_l",
    "architecture_family": "End-to-end Transformer",
    "dataset_track": "2class",
    "input_resolution": 640,
    "seed": 42,
}


def test_old_and_versioned_prediction_paths_have_metric_parity(tmp_path: Path) -> None:
    legacy_metrics = detailed_metrics(
        FIXTURE / "ground_truth.json", FIXTURE / "predictions.json"
    )
    legacy_v2 = evaluate_prediction_artifact(
        FIXTURE / "ground_truth.json",
        FIXTURE / "predictions.json",
        legacy_identity=IDENTITY,
    )
    versioned_path = tmp_path / "predictions.v1.json"
    versioned_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "coco_predictions",
                "identity": IDENTITY,
                "predictions": read_json(FIXTURE / "predictions.json"),
            }
        ),
        encoding="utf-8",
    )
    versioned_v2 = evaluate_prediction_artifact(
        FIXTURE / "ground_truth.json", versioned_path
    )
    assert_metric_parity(legacy_metrics, legacy_v2["metrics"])
    assert_metric_parity(legacy_v2["metrics"], versioned_v2["metrics"])


def test_track_and_inference_tables_are_written_separately(tmp_path: Path) -> None:
    artifacts = []
    for index, (track, mode) in enumerate(
        (("controlled", "full"), ("performance", "full"), ("performance", "sliced"), ("performance", "ensemble"))
    ):
        artifacts.append(
            {
                "schema_version": 2,
                "identity": {**IDENTITY, "run_id": f"run-{index}"},
                "benchmark_track": track,
                "inference_mode": mode,
                "weight_variant": "raw",
                "metrics": {"mAP": index / 10},
            }
        )
    tables = build_comparison_tables(artifacts)
    assert [row["run_id"] for row in tables["controlled"]] == ["run-0"]
    assert {row["run_id"] for row in tables["performance"]} == {"run-1", "run-2", "run-3"}
    assert {row["run_id"] for row in tables["full"]} == {"run-0", "run-1"}
    assert [row["run_id"] for row in tables["sliced"]] == ["run-2"]
    assert [row["run_id"] for row in tables["ensemble"]] == ["run-3"]
    outputs = write_comparison_tables(tmp_path, tables)
    assert len(outputs) == 10
    assert read_json(outputs["controlled_json"])["table"] == "controlled"
