from __future__ import annotations

import json
from pathlib import Path

from src.artifacts import (
    ArtifactIdentity,
    load_checkpoint_artifact,
    load_metric_artifact,
    load_prediction_artifact,
    write_checkpoint_artifact,
    write_metric_artifact,
    write_prediction_artifact,
)
from src.evaluation.detection_metrics import detailed_metrics
from src.utils.serialization import read_json, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "legacy_artifacts"


def _identity() -> ArtifactIdentity:
    return ArtifactIdentity(
        run_id="faster_rcnn_resnet50__2class__640__20260729_120000__seed42",
        model_id="faster_rcnn_resnet50",
        architecture_family="CNN",
        dataset_track="2class",
        evaluation_resolution=640,
        seed=42,
    )


def test_metric_dual_write_preserves_flat_legacy_values(tmp_path: Path) -> None:
    legacy_fixture = read_json(FIXTURE / "evaluation_metrics.json")
    identity_keys = set(_identity().to_dict())
    metrics = {key: value for key, value in legacy_fixture.items() if key not in identity_keys}
    paths = write_metric_artifact(
        tmp_path / "metrics.v1.json",
        _identity(),
        metrics,
        legacy_destination=tmp_path / "legacy" / "metrics.json",
    )
    assert load_metric_artifact(paths["versioned"]) == read_json(paths["legacy"])
    assert read_json(paths["legacy"]) == legacy_fixture
    assert read_json(paths["versioned"])["schema_version"] == 1


def test_prediction_dual_write_has_equal_old_and_new_evaluator_values(
    tmp_path: Path,
) -> None:
    predictions = read_json(FIXTURE / "predictions.json")
    paths = write_prediction_artifact(
        tmp_path / "predictions.v1.json",
        _identity(),
        predictions,
        legacy_destination=tmp_path / "legacy" / "predictions.json",
    )
    new_reader_view = tmp_path / "new_reader_predictions.json"
    write_json(new_reader_view, load_prediction_artifact(paths["versioned"]))
    new_metrics = detailed_metrics(FIXTURE / "ground_truth.json", new_reader_view)
    legacy_metrics = detailed_metrics(FIXTURE / "ground_truth.json", paths["legacy"])
    assert new_metrics == legacy_metrics
    assert read_json(paths["legacy"]) == predictions


def test_checkpoint_dual_write_preserves_bytes_and_frozen_alias(tmp_path: Path) -> None:
    source = tmp_path / "source.pth"
    source.write_bytes(b"synthetic-checkpoint-state")
    paths = write_checkpoint_artifact(
        source,
        tmp_path / "versioned",
        _identity(),
        role="best_map",
        legacy_run_dir=tmp_path / "legacy",
        state_keys=["optimizer", "model", "epoch"],
    )
    metadata = load_checkpoint_artifact(paths["metadata"])
    assert paths["legacy"].name == "best_map.pth"
    assert paths["checkpoint"].read_bytes() == paths["legacy"].read_bytes() == source.read_bytes()
    assert metadata["schema_version"] == 1
    assert metadata["state_keys"] == ["epoch", "model", "optimizer"]
    assert metadata["checkpoint_sha256"] == sha256_file(source)


def test_portable_schemas_are_strict_and_versioned() -> None:
    schema_root = ROOT / "schemas" / "artifacts"
    for relative in (
        "checkpoint/v1.json",
        "metric/v1.json",
        "prediction/v1.json",
        "shared/identity_v1.json",
    ):
        schema = json.loads((schema_root / relative).read_text())
        assert schema["additionalProperties"] is False
