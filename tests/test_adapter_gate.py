from pathlib import Path

import pytest

from src.workflows.adapter_gate import (
    FINGERPRINT_FIELDS,
    adapter_fingerprint,
    adapter_gate_decision,
    print_gate_decision,
)

MODEL_ID = "rtdetrv2_l"


def _fingerprint() -> dict[str, object]:
    return {
        "adapter_schema_version": 2,
        "git_commit": "abc123",
        "model_id": MODEL_ID,
        "framework": "transformers",
        "python_version": "3.11.0",
        "pytorch_version": "2.1.0",
        "cuda_version": "11.8",
        "gpu": "Test GPU",
        "dependency_lock_hash": "lock-a",
    }


def test_adapter_fingerprint_contains_complete_compatibility_contract() -> None:
    fingerprint = adapter_fingerprint(
        MODEL_ID, Path(__file__).resolve().parents[1]
    )
    assert set(FINGERPRINT_FIELDS) == set(fingerprint)
    assert fingerprint["model_id"] == MODEL_ID
    assert fingerprint["framework"] == "transformers"
    assert len(str(fingerprint["dependency_lock_hash"])) == 64


@pytest.mark.parametrize(
    ("gate", "decision"),
    [
        ({"status": "READY", "fingerprint": _fingerprint()}, "reuse"),
        ({"status": "FAILED_ADAPTER", "fingerprint": _fingerprint()}, "blocked"),
        ({"status": "FAILED_ADAPTER"}, "retry"),
        ({}, "run"),
    ],
)
def test_gate_decisions_for_compatible_and_legacy_records(
    gate: dict[str, object], decision: str
) -> None:
    observed, reasons = adapter_gate_decision(gate, _fingerprint())
    assert observed == decision
    assert reasons


def test_ready_gate_is_invalidated_and_failed_gate_retried_after_source_change() -> None:
    current = _fingerprint()
    current["git_commit"] = "repaired"
    ready, ready_reasons = adapter_gate_decision(
        {"status": "READY", "fingerprint": _fingerprint()}, current
    )
    failed, failed_reasons = adapter_gate_decision(
        {"status": "FAILED_ADAPTER", "fingerprint": _fingerprint()}, current
    )
    assert ready == "invalidate"
    assert failed == "retry"
    assert ready_reasons == failed_reasons
    assert "git_commit" in ready_reasons[0]


def test_failed_environment_gate_is_retryable_with_same_fingerprint() -> None:
    decision, reasons = adapter_gate_decision(
        {"status": "FAILED_ENVIRONMENT", "fingerprint": _fingerprint()},
        _fingerprint(),
    )
    assert decision == "retry"
    assert "transactional runtime state" in reasons[0]


@pytest.mark.parametrize(
    ("decision", "prefix"),
    [
        ("reuse", "ADAPTER GATE REUSED:"),
        ("invalidate", "ADAPTER GATE INVALIDATED:"),
        ("retry", "ADAPTER GATE RETRIED:"),
    ],
)
def test_gate_decision_prints_exact_action(
    decision: str, prefix: str, capsys
) -> None:
    print_gate_decision(decision, ["reason"])
    assert capsys.readouterr().out == f"{prefix} reason\n"


def test_gate_decision_never_mutates_checkpoint_tree(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "last.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    current = _fingerprint()
    current["dependency_lock_hash"] = "lock-b"

    decision, _ = adapter_gate_decision(
        {"status": "FAILED_ADAPTER", "fingerprint": _fingerprint()}, current
    )

    assert decision == "retry"
    assert checkpoint.read_bytes() == b"checkpoint"


# --- PR-08: GPU adapter smoke-gate assertion logic (CPU-testable) -------------

from src.workflows.adapter_gate import (  # noqa: E402
    AdapterGateError,
    assert_checkpoint_roundtrip,
    assert_detection_head_class_count,
    assert_feature_map_contract,
    assert_finite_loss,
    assert_pretrained_load_complete,
    assert_wellformed_predictions,
    build_smoke_record,
    SMOKE_CHECK_ORDER,
)


def test_partial_weight_load_is_fatal():
    assert_pretrained_load_complete([], [])  # complete load is fine
    with pytest.raises(AdapterGateError, match="missing="):
        assert_pretrained_load_complete(["backbone.stage4.weight"], [])
    with pytest.raises(AdapterGateError, match="unexpected="):
        assert_pretrained_load_complete([], ["head.coco80.bias"])


def test_feature_map_contract_checks_channels_stride_and_nchw():
    # FPN at 640px: 256 channels, strides 4/8/16/32 -> 160/80/40/20 spatial.
    fmaps = [
        {"shape": (2, 256, 160, 160), "stride": 4},
        {"shape": (2, 256, 80, 80), "stride": 8},
        {"shape": (2, 256, 40, 40), "stride": 16},
        {"shape": (2, 256, 20, 20), "stride": 32},
    ]
    expected = [(256, 4), (256, 8), (256, 16), (256, 32)]
    assert_feature_map_contract(fmaps, expected, image_size=640)

    wrong_channels = [dict(fmaps[0], shape=(2, 512, 160, 160))] + fmaps[1:]
    with pytest.raises(AdapterGateError, match="channels"):
        assert_feature_map_contract(wrong_channels, expected, image_size=640)

    # 224px feature map slipped in where 640 was configured.
    wrong_spatial = [dict(fmaps[0], shape=(2, 256, 56, 56))] + fmaps[1:]
    with pytest.raises(AdapterGateError, match="spatial"):
        assert_feature_map_contract(wrong_spatial, expected, image_size=640)

    not_nchw = [{"shape": (256, 160, 160), "stride": 4}]
    with pytest.raises(AdapterGateError, match="not NCHW"):
        assert_feature_map_contract(not_nchw, [(256, 4)], image_size=640)


def test_detection_head_rejects_coco_residue():
    assert_detection_head_class_count(2, 2)
    with pytest.raises(AdapterGateError, match="COCO-80 residue"):
        assert_detection_head_class_count(80, 2)


def test_finite_loss_guard():
    assert assert_finite_loss(1.5) == 1.5
    with pytest.raises(AdapterGateError, match="non-finite"):
        assert_finite_loss(float("nan"))
    with pytest.raises(AdapterGateError, match="non-finite"):
        assert_finite_loss(float("inf"))


def test_wellformed_predictions_guard():
    good = [{"boxes": [[1, 1, 5, 5]], "scores": [0.9], "labels": [1]}]
    assert_wellformed_predictions(good, num_classes=2)
    degenerate = [{"boxes": [[5, 5, 1, 1]], "scores": [0.9], "labels": [1]}]
    with pytest.raises(AdapterGateError, match="degenerate box"):
        assert_wellformed_predictions(degenerate, num_classes=2)
    bad_score = [{"boxes": [[1, 1, 5, 5]], "scores": [1.4], "labels": [1]}]
    with pytest.raises(AdapterGateError, match=r"score outside"):
        assert_wellformed_predictions(bad_score, num_classes=2)
    bad_label = [{"boxes": [[1, 1, 5, 5]], "scores": [0.5], "labels": [7]}]
    with pytest.raises(AdapterGateError, match="outside"):
        assert_wellformed_predictions(bad_label, num_classes=2)


def test_checkpoint_roundtrip_guard():
    before = {"a": "h1", "b": "h2"}
    assert_checkpoint_roundtrip(before, dict(before))
    with pytest.raises(AdapterGateError, match="changed across save/load"):
        assert_checkpoint_roundtrip(before, {"a": "h1", "b": "CHANGED"})
    with pytest.raises(AdapterGateError, match="names changed"):
        assert_checkpoint_roundtrip(before, {"a": "h1"})


def _passing_checks():
    return [{"name": name, "passed": True} for name in SMOKE_CHECK_ORDER]


def test_build_smoke_record_signs_and_orders_checks():
    record = build_smoke_record(
        MODEL_ID, _fingerprint(), _passing_checks(), gpu="Tesla T4"
    )
    assert record["status"] == "READY"
    assert record["artifact_kind"] == "gpu_adapter_smoke"
    assert len(record["signature"]) == 64
    # A single failed check flips the record to FAILED_ADAPTER.
    failed = _passing_checks()
    failed[4]["passed"] = False
    assert build_smoke_record(MODEL_ID, _fingerprint(), failed, gpu="T4")[
        "status"
    ] == "FAILED_ADAPTER"


def test_build_smoke_record_requires_full_ordered_check_set():
    with pytest.raises(AdapterGateError, match="in order"):
        build_smoke_record(
            MODEL_ID, _fingerprint(), _passing_checks()[:3], gpu="T4"
        )
