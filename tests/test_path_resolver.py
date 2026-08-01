from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation.detection_metrics import detailed_metrics
from src.pathing import ArtifactPathResolver, RunPathIdentity, resolve_legacy_paths
from src.paths import ProjectPaths


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "legacy_artifacts"


def test_legacy_resolver_matches_every_existing_project_path(tmp_path: Path) -> None:
    old = ProjectPaths.from_value(tmp_path)
    new = resolve_legacy_paths(tmp_path)
    properties = (
        "datasets",
        "visdrone",
        "archives",
        "raw",
        "processed",
        "dataset_manifests",
        "lr_search_manifests",
        "checkpoints",
        "lr_search_checkpoints",
        "final_checkpoints",
        "registry_dir",
        "checkpoint_registry",
        "runs_csv",
        "predictions",
        "evaluation",
        "reports",
        "cache",
        "logs",
        "profiling",
        "exports",
        "result_bundles",
        "pretrained",
    )
    assert type(new) is ProjectPaths
    for name in properties:
        assert getattr(new, name) == getattr(old, name), name
    assert new.coco("2class") == old.coco("2class")
    assert new.run_dir("model", "run") == old.run_dir("model", "run")


def _identity(track: str, mode: str, run_id: str) -> RunPathIdentity:
    return RunPathIdentity(track, mode, "rtdetrv2_l", "baseline", 42, run_id)


def test_smoke_controlled_performance_and_inference_modes_are_isolated(tmp_path: Path) -> None:
    resolver = ArtifactPathResolver(tmp_path)
    values = {
        resolver.run_path("checkpoints", _identity("smoke", "smoke", "smoke-1")),
        resolver.run_path("checkpoints", _identity("controlled", "full", "control-1")),
        resolver.run_path("checkpoints", _identity("performance", "full", "perf-1")),
        resolver.run_path("predictions", _identity("performance", "sliced", "slice-1")),
        resolver.run_path("predictions", _identity("performance", "ensemble", "ens-1")),
    }
    assert len(values) == 5
    assert all(str(path).startswith(str(tmp_path.resolve())) for path in values)


def test_invalid_track_mode_and_unsafe_segments_are_rejected() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        _identity("controlled", "sliced", "bad")
    with pytest.raises(ValueError, match="unsafe"):
        RunPathIdentity("smoke", "smoke", "../model", "baseline", 42, "run")


def test_existing_evaluator_still_reads_legacy_fixture() -> None:
    result = detailed_metrics(FIXTURES / "ground_truth.json", FIXTURES / "predictions.json")
    assert result["per_class_detailed"]["person"]["true_positives"] == 1
