"""One model's benchmark day runs as one ordered, re-entrant sequence.

The pipeline replaced six manual launches with one call, so the ordering that
used to live in an operator's head now has to live in a test: the dataset comes
before anything reads it, the adapter gate comes before anything expensive, and
a preview never downloads, provisions, or trains.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.paths import ProjectPaths
from src.utils.serialization import write_json
from src.workflows import model_pipeline
from src.workflows.adapter_gate import (
    SMOKE_CHECK_ORDER,
    AdapterGateError,
    adapter_fingerprint,
    build_smoke_record,
    smoke_check_result,
    smoke_record_path,
)
from src.workflows.model_pipeline import PIPELINE_STAGES, run_model_pipeline

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "faster_rcnn_resnet50"


def _ready_dataset(paths: ProjectPaths) -> None:
    """A dataset large enough for the held-out split fractions to be non-empty.

    Train and validation file names must not overlap: the manifest validator
    treats a shared name as the two official splits contaminating each other.
    """
    for split, count in (("train", 40), ("val", 20)):
        write_json(
            paths.coco("2class") / "annotations" / f"instances_{split}.json",
            {
                "images": [
                    {
                        "id": index,
                        "file_name": f"{split}_{index:04d}.jpg",
                        "width": 32,
                        "height": 32,
                    }
                    for index in range(count)
                ],
                "annotations": [],
                "categories": [
                    {"id": 1, "name": "person"},
                    {"id": 2, "name": "vehicle"},
                ],
            },
        )
        paths.images(split).mkdir(parents=True, exist_ok=True)


def _ready_gate(paths: ProjectPaths) -> None:
    write_json(
        smoke_record_path(paths.root, MODEL_ID, "2class"),
        build_smoke_record(
            MODEL_ID,
            adapter_fingerprint(MODEL_ID, ROOT),
            [smoke_check_result(name, passed=True) for name in SMOKE_CHECK_ORDER],
            gpu=str(adapter_fingerprint(MODEL_ID, ROOT)["gpu"]),
            dataset_track="2class",
            image_size=640,
        ),
    )


def _stage(result: dict, name: str) -> dict:
    return next(stage for stage in result["stages"] if stage["stage"] == name)


def test_preview_stops_at_an_unprepared_dataset_without_downloading(tmp_path, monkeypatch):
    monkeypatch.setattr(
        model_pipeline, "prepare_visdrone", lambda *a, **k: pytest.fail("downloaded")
    )
    result = run_model_pipeline(ROOT, tmp_path, MODEL_ID)

    assert result["preview"] is True
    assert _stage(result, "dataset")["status"] == "PENDING"
    assert result["completed_stages"] == ["dataset"]
    assert result["next_stage"] == "environment"


def test_preview_never_provisions_an_environment_or_runs_the_gate(tmp_path, monkeypatch):
    paths = ProjectPaths.from_value(tmp_path).create()
    _ready_dataset(paths)
    monkeypatch.setattr(
        model_pipeline,
        "ensure_model_environment",
        lambda *a, **k: pytest.fail("provisioned"),
    )
    monkeypatch.setattr(
        model_pipeline, "run_adapter_gate", lambda *a, **k: pytest.fail("ran the gate")
    )

    result = run_model_pipeline(ROOT, tmp_path, MODEL_ID)

    assert _stage(result, "environment")["status"] == "SKIPPED_PREVIEW"
    gate = _stage(result, "adapter_gate")
    assert gate["status"] == "PENDING"
    assert gate["blockers"] == ["no adapter smoke record exists"]
    # HPO previews its contract; finetuning has no tuned config yet on a first
    # pass, which is a pending stage rather than a failure.
    assert _stage(result, "hpo")["status"] == "PREVIEW"
    assert _stage(result, "finetune")["status"] == "PENDING"


def test_preview_reaches_evaluation_once_hpo_has_exported_a_config(tmp_path, monkeypatch):
    paths = ProjectPaths.from_value(tmp_path).create()
    _ready_dataset(paths)
    _ready_gate(paths)
    monkeypatch.setattr(
        model_pipeline.FinalExperimentWorkflow,
        "run",
        lambda self, **kwargs: {"preview": True},
    )

    result = run_model_pipeline(ROOT, tmp_path, MODEL_ID)

    assert _stage(result, "adapter_gate")["status"] == "READY"
    assert [stage["stage"] for stage in result["stages"]] == list(PIPELINE_STAGES)
    assert result["next_stage"] is None
    assert _stage(result, "evaluation")["detail"]["pending"] == []


def test_a_failed_gate_stops_the_run_before_hpo(tmp_path, monkeypatch):
    """The gate is auto-run now, which must not make it easier to bypass."""
    paths = ProjectPaths.from_value(tmp_path).create()
    _ready_dataset(paths)
    order: list[str] = []
    monkeypatch.setattr(
        model_pipeline,
        "ensure_model_environment",
        lambda *a, **k: order.append("environment") or {"status": "READY"},
    )
    monkeypatch.setattr(
        model_pipeline,
        "run_adapter_gate",
        lambda *a, **k: order.append("gate") or {"status": "FAILED_ADAPTER"},
    )
    monkeypatch.setattr(
        model_pipeline.TwoStageRandomHPO,
        "run",
        lambda self, **kwargs: order.append("hpo"),
    )

    with pytest.raises(AdapterGateError):
        run_model_pipeline(ROOT, tmp_path, MODEL_ID, start=True)

    assert order == ["environment", "gate"]


def test_the_gate_is_reused_rather_than_rerun_when_it_still_matches(tmp_path, monkeypatch):
    paths = ProjectPaths.from_value(tmp_path).create()
    _ready_dataset(paths)
    _ready_gate(paths)
    order: list[str] = []
    monkeypatch.setattr(
        model_pipeline,
        "ensure_model_environment",
        lambda *a, **k: order.append("environment") or {"status": "READY"},
    )
    monkeypatch.setattr(
        model_pipeline, "run_adapter_gate", lambda *a, **k: pytest.fail("reran the gate")
    )
    monkeypatch.setattr(
        model_pipeline.TwoStageRandomHPO,
        "run",
        lambda self, **kwargs: order.append("hpo") or {"stage": "HPO"},
    )
    monkeypatch.setattr(
        model_pipeline.FinalExperimentWorkflow,
        "run",
        lambda self, **kwargs: order.append("finetune") or {"stage": "FINAL"},
    )
    monkeypatch.setattr(
        model_pipeline,
        "evaluate_pending_runs",
        lambda *a, **k: order.append("evaluation") or {"evaluated": []},
    )

    result = run_model_pipeline(ROOT, tmp_path, MODEL_ID, start=True)

    assert order == ["environment", "hpo", "finetune", "evaluation"]
    assert _stage(result, "adapter_gate")["action"] == "reused"
    assert result["preview"] is False


def test_an_unknown_model_id_fails_before_any_stage(tmp_path):
    with pytest.raises(Exception):
        run_model_pipeline(ROOT, tmp_path, "not_a_model")


def test_an_unsupported_dataset_track_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unsupported dataset track"):
        run_model_pipeline(ROOT, tmp_path, MODEL_ID, "3class")
