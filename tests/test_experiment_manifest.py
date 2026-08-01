from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.manifests import (
    ManifestValidationError,
    create_experiment_manifest,
    finalize_experiment_manifest,
    load_experiment_manifest,
    validate_experiment_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def _create(path: Path) -> dict[str, object]:
    return create_experiment_manifest(
        path,
        experiment_id="round-trip",
        model_id="faster_rcnn_resnet50",
        dataset_track="2class",
        execution_mode="smoke",
        code={"repository": "repo", "revision": "1" * 40},
        config={"path": "config.yaml", "sha256": "2" * 64},
        dataset={"name": "VisDrone2019-DET", "version": "2019", "hashes": {"val": "3" * 64}},
        environment={"python_version": "3.11", "dependencies_sha256": "4" * 64},
        hardware={"device_type": "cpu", "device_name": "test-cpu"},
        seed=42,
        output_path="artifacts/smoke/round-trip",
        created_at="2026-08-01T00:00:00+00:00",
    )


def test_success_manifest_create_finalize_reload_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "experiment_manifest.json"
    created = _create(path)
    assert load_experiment_manifest(path) == created
    completed = finalize_experiment_manifest(
        path,
        status="completed",
        result={"mAP": 0.25},
        completed_at="2026-08-01T00:01:00+00:00",
    )
    assert load_experiment_manifest(path) == completed
    assert completed["status"] == "completed"
    assert completed["result"] == {"mAP": 0.25}


def test_failure_manifest_create_finalize_reload_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "experiment_manifest.json"
    _create(path)
    failed = finalize_experiment_manifest(
        path,
        status="failed",
        failure="synthetic interruption",
        completed_at="2026-08-01T00:00:05+00:00",
    )
    assert load_experiment_manifest(path) == failed
    assert failed["failure"] == "synthetic interruption"


def test_terminal_manifest_cannot_be_finalized_twice(tmp_path: Path) -> None:
    path = tmp_path / "experiment_manifest.json"
    _create(path)
    finalize_experiment_manifest(path, status="completed", result={})
    with pytest.raises(ManifestValidationError, match="only a running"):
        finalize_experiment_manifest(path, status="failed", failure="late")


def test_failure_requires_a_reason(tmp_path: Path) -> None:
    path = tmp_path / "experiment_manifest.json"
    _create(path)
    with pytest.raises(ManifestValidationError, match="requires completed_at and failure"):
        finalize_experiment_manifest(path, status="failed")
    assert load_experiment_manifest(path)["status"] == "running"


def test_checked_in_examples_validate_and_match_portable_schema() -> None:
    examples = ROOT / "schemas" / "experiment_manifest" / "examples"
    for path in sorted(examples.glob("*.json")):
        assert validate_experiment_manifest(json.loads(path.read_text()))
    schema = json.loads(
        (ROOT / "schemas" / "experiment_manifest" / "v1.json").read_text()
    )
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])


def test_new_manifest_name_does_not_match_legacy_evaluator_glob(tmp_path: Path) -> None:
    run_dir = tmp_path / "checkpoints" / "model" / "run"
    run_dir.mkdir(parents=True)
    _create(run_dir / "experiment_manifest.json")
    assert list(tmp_path.glob("checkpoints/*/*/run_manifest.json")) == []
