"""The report stage must actually evaluate, aggregate, and report.

Notebooks 30/31 previously bootstrapped and then stopped, so the last stage of
the benchmark produced nothing while appearing to run. These tests pin the
wiring: what is pending, what gets launched, what is written, and that
publishing never happens implicitly.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.hpo.final_workflow import IMAGE_SIZE
from src.hpo.workflow import HPO_PROTOCOL_ID
from src.paths import ProjectPaths
from src.utils.serialization import read_json, write_json
from src.workflows import evaluation_runner, reporting
from src.workflows.evaluation_runner import evaluate_pending_runs, pending_evaluations

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "faster_rcnn_resnet50"


def _registered_run(
    paths: ProjectPaths, model_id: str = MODEL_ID, *, seed: int = 42
) -> dict:
    run_id = f"{model_id}__2class__{IMAGE_SIZE}__20260729_120000__seed{seed}"
    manifest = {
        "run_id": run_id,
        "run_dir": str(paths.final_checkpoints / model_id / run_id),
        "model_id": model_id,
        "dataset_track": "2class",
        "status": "completed",
        "protocol_id": HPO_PROTOCOL_ID,
        "run_kind": "final_complete_official_train",
        "baseline_or_tuned": "tuned",
        "seed": seed,
        "input_resolution": IMAGE_SIZE,
        "created_at": "2026-07-29T12:00:00Z",
    }
    registry = (
        read_json(paths.checkpoint_registry)
        if paths.checkpoint_registry.is_file()
        else {"schema_version": 1, "runs": {}}
    )
    registry["runs"][run_id] = manifest
    write_json(paths.checkpoint_registry, registry)
    return manifest


def _metrics(paths: ProjectPaths, run_id: str, *, seed: int = 42) -> None:
    write_json(
        paths.evaluation / f"{run_id}__res{IMAGE_SIZE}__metrics.json",
        {
            "run_id": run_id,
            "model_id": MODEL_ID,
            "dataset_track": "2class",
            "evaluation_resolution": IMAGE_SIZE,
            "seed": seed,
            "mAP": 0.5,
        },
    )


def test_pending_evaluations_skips_runs_that_already_have_metrics(tmp_path):
    paths = ProjectPaths.from_value(tmp_path).create()
    manifest = _registered_run(paths)
    assert [run["run_id"] for run in pending_evaluations(paths.root, "2class")] == [
        manifest["run_id"]
    ]
    _metrics(paths, manifest["run_id"])
    assert pending_evaluations(paths.root, "2class") == []


def test_pending_evaluations_honours_the_model_filter(tmp_path):
    paths = ProjectPaths.from_value(tmp_path).create()
    _registered_run(paths, MODEL_ID)
    _registered_run(paths, "rtdetrv2_l")
    selected = pending_evaluations(paths.root, "2class", model_ids=[MODEL_ID])
    assert [run["model_id"] for run in selected] == [MODEL_ID]


def test_evaluation_launches_one_module_per_pending_run(tmp_path, monkeypatch):
    """Evaluation must run in the model runtime, not the notebook kernel."""
    paths = ProjectPaths.from_value(tmp_path).create()
    manifest = _registered_run(paths)
    launched: list[tuple] = []

    monkeypatch.setattr(
        evaluation_runner, "ensure_model_environment", lambda *a, **k: {"status": "READY"}
    )
    monkeypatch.setattr(
        evaluation_runner,
        "run_module_in_model_runtime",
        lambda repo, module, *arguments, **kwargs: launched.append(
            (module, arguments)
        ),
    )
    result = evaluate_pending_runs(ROOT, paths.root, "2class", skip_profile=True)

    assert [module for module, _ in launched] == ["scripts.evaluate"]
    arguments = launched[0][1]
    assert "--run-id" in arguments and manifest["run_id"] in arguments
    assert "--skip-profile" in arguments
    assert result["evaluated"] == [
        {"model_id": MODEL_ID, "run_id": manifest["run_id"]}
    ]
    # The subprocess was faked, so the metrics file still does not exist: the
    # report must say so rather than imply the run was evaluated.
    assert result["still_missing"] == [
        {"model_id": MODEL_ID, "run_id": manifest["run_id"]}
    ]


def test_report_aggregates_metrics_and_does_not_publish_by_default(tmp_path, monkeypatch):
    paths = ProjectPaths.from_value(tmp_path).create()
    manifest = _registered_run(paths)
    _metrics(paths, manifest["run_id"])
    monkeypatch.setattr(
        reporting, "publish_results", lambda *a, **k: pytest.fail("published")
    )

    result = reporting.build_benchmark_report(
        ROOT, paths.root, "2class", evaluate_missing=False
    )

    assert result["publish_status"] == "NOT_PUBLISHED"
    # One model cannot be compared against three that have not run; the report
    # must still produce everything it can.
    assert result["comparison"]["status"] == "UNAVAILABLE"
    assert result["published"] == []
    assert result["evaluated_metric_files"] == 1
    assert result["evaluation"]["still_missing"] == []
    assert (paths.reports / "final_results.csv").is_file()
    assert (paths.reports / "final_results.json").is_file()
    assert Path(result["aggregate"]["output"]).is_file()


def test_report_refuses_a_publish_request_that_is_still_a_dry_run(tmp_path):
    paths = ProjectPaths.from_value(tmp_path).create()
    with pytest.raises(ValueError, match="publish=True and dry_run=False"):
        reporting.build_benchmark_report(
            ROOT, paths.root, "2class", evaluate_missing=False, publish=True
        )
