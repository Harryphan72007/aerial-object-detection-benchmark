"""The live HPO and final-training workflows enforce the GPU smoke gate.

Storing an adapter fingerprint in a manifest documents what ran; it does not
prevent a run from starting with a broken adapter. These tests cover the
enforcement itself: no READY record, no expensive stage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.hpo.final_workflow import FinalExperimentWorkflow
from src.hpo.workflow import TwoStageRandomHPO
from src.utils.serialization import write_json
from src.workflows.adapter_gate import (
    SMOKE_CHECK_ORDER,
    AdapterGateError,
    adapter_fingerprint,
    build_smoke_record,
    smoke_check_result,
    smoke_record_path,
)
from src.workflows.dataset_setup import DatasetTrackNotPreparedError

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "faster_rcnn_resnet50"


def _write_ready_record(drive_root: Path, **overrides: object) -> Path:
    record = build_smoke_record(
        MODEL_ID,
        adapter_fingerprint(MODEL_ID, ROOT),
        [smoke_check_result(name, passed=True) for name in SMOKE_CHECK_ORDER],
        gpu="Test GPU",
        dataset_track=str(overrides.pop("dataset_track", "2class")),
        image_size=int(overrides.pop("image_size", 640)),
    )
    record.update(overrides)
    path = smoke_record_path(drive_root, MODEL_ID, str(record["dataset_track"]))
    write_json(path, record)
    return path


class _StubStudy:
    """Enough of an Optuna study to reach the post-gate control flow."""

    trials: list[object] = []


def _hpo(drive_root: Path) -> TwoStageRandomHPO:
    return TwoStageRandomHPO(ROOT, drive_root, MODEL_ID, "2class")


def _final(drive_root: Path) -> FinalExperimentWorkflow:
    return FinalExperimentWorkflow(ROOT, drive_root, MODEL_ID, "2class")


def test_both_workflows_require_the_gate_by_default(tmp_path: Path) -> None:
    assert _hpo(tmp_path).require_adapter_gate is True
    assert _final(tmp_path).require_adapter_gate is True


@pytest.mark.parametrize("factory", [_hpo, _final])
def test_missing_record_blocks_and_names_the_gate_command(factory, tmp_path) -> None:
    with pytest.raises(AdapterGateError) as raised:
        factory(tmp_path).assert_adapter_gate()

    message = str(raised.value)
    assert "no adapter smoke record exists" in message
    assert "python -m scripts.gpu_adapter_smoke" in message
    assert MODEL_ID in message


@pytest.mark.parametrize("factory", [_hpo, _final])
def test_matching_ready_record_authorizes_the_run(factory, tmp_path) -> None:
    _write_ready_record(tmp_path)

    record = factory(tmp_path).assert_adapter_gate()

    assert record["status"] == "READY"
    assert record["model_id"] == MODEL_ID


def test_a_failed_record_never_authorizes_a_run(tmp_path: Path) -> None:
    checks = [smoke_check_result(name, passed=True) for name in SMOKE_CHECK_ORDER]
    checks[4] = smoke_check_result(
        "forward_backward_finite_loss", passed=False, error=RuntimeError("CUDA error")
    )
    record = build_smoke_record(
        MODEL_ID,
        adapter_fingerprint(MODEL_ID, ROOT),
        checks,
        gpu="Test GPU",
        dataset_track="2class",
        image_size=640,
    )
    write_json(smoke_record_path(tmp_path, MODEL_ID, "2class"), record)

    with pytest.raises(AdapterGateError, match="FAILED_ADAPTER"):
        _hpo(tmp_path).assert_adapter_gate()


def test_a_partial_record_promoted_to_ready_is_rejected(tmp_path: Path) -> None:
    """Hand-editing status to READY does not survive the signature check."""
    record = build_smoke_record(
        MODEL_ID,
        adapter_fingerprint(MODEL_ID, ROOT),
        [smoke_check_result(name, passed=True) for name in SMOKE_CHECK_ORDER[:2]],
        gpu="Test GPU",
        dataset_track="2class",
        image_size=640,
    )
    forged = {**record, "status": "READY", "reasons": []}
    write_json(smoke_record_path(tmp_path, MODEL_ID, "2class"), forged)

    with pytest.raises(AdapterGateError) as raised:
        _hpo(tmp_path).assert_adapter_gate()

    message = str(raised.value)
    assert "missing required checks" in message
    assert "signature" in message


def test_a_record_for_another_track_or_resolution_is_rejected(tmp_path: Path) -> None:
    _write_ready_record(tmp_path, dataset_track="2class", image_size=1024)

    with pytest.raises(AdapterGateError, match="image size 1024"):
        _hpo(tmp_path).assert_adapter_gate()


def test_a_record_from_another_environment_is_rejected(tmp_path: Path) -> None:
    record = build_smoke_record(
        MODEL_ID,
        {**adapter_fingerprint(MODEL_ID, ROOT), "gpu": "Some other GPU"},
        [smoke_check_result(name, passed=True) for name in SMOKE_CHECK_ORDER],
        gpu="Some other GPU",
        dataset_track="2class",
        image_size=640,
    )
    write_json(smoke_record_path(tmp_path, MODEL_ID, "2class"), record)

    with pytest.raises(AdapterGateError, match="gpu:"):
        _hpo(tmp_path).assert_adapter_gate()


def test_the_gate_runs_before_any_expensive_work_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate must precede study creation and manifest preparation."""
    order: list[str] = []
    workflow = _hpo(tmp_path)
    monkeypatch.setattr(
        workflow, "prepare_manifests", lambda: order.append("manifests") or {"hashes": {}}
    )
    monkeypatch.setattr(
        workflow,
        "assert_adapter_gate",
        lambda: order.append("gate"),
    )
    monkeypatch.setattr(
        workflow, "_study", lambda metadata: order.append("study") or _StubStudy()
    )
    monkeypatch.setattr(
        workflow, "_run_phase", lambda *_args: order.append("phase")
    )
    annotations = workflow.paths.coco("2class") / "annotations"
    annotations.mkdir(parents=True, exist_ok=True)
    for name in ("instances_train.json", "instances_val.json"):
        (annotations / name).write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError):  # Phase A produces no trials in this stub
        workflow.run(start_expensive_stage=True)

    assert order.index("gate") < order.index("study")


def test_an_unprepared_track_fails_before_the_gate(tmp_path: Path) -> None:
    """Selecting 10class without preparing it names the exact remedy."""
    workflow = TwoStageRandomHPO(
        ROOT, tmp_path, MODEL_ID, "10class", require_adapter_gate=False
    )

    with pytest.raises(DatasetTrackNotPreparedError) as raised:
        workflow.run(start_expensive_stage=False)

    assert "PREPARE_10CLASS_TRACK = True" in str(raised.value)
