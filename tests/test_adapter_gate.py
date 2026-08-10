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

from src.models.adapter_validation import (  # noqa: E402
    AdapterValidationContract,
    CheckpointLoadResult,
)
from src.workflows.adapter_gate import (  # noqa: E402
    AdapterGateError,
    SMOKE_CHECK_ORDER,
    SMOKE_CONTRACT_VERSION,
    assert_checkpoint_roundtrip,
    assert_detection_head_class_count,
    assert_feature_map_contract,
    assert_finite_loss,
    assert_pretrained_load_complete,
    assert_wellformed_predictions,
    build_smoke_record,
    require_ready_adapter_gate,
    smoke_check_result,
    smoke_gate_blockers,
    smoke_record_path,
)


def test_partial_weight_load_is_fatal():
    complete = CheckpointLoadResult(
        state="loaded", source="local.pth", value_coverage=1.0, minimum_value_coverage=0.95
    )
    assert assert_pretrained_load_complete(complete) is complete
    with pytest.raises(AdapterGateError, match="missing="):
        assert_pretrained_load_complete(
            CheckpointLoadResult(
                state="incomplete", missing_keys=("backbone.stage4.weight",)
            )
        )
    with pytest.raises(AdapterGateError, match="unexpected="):
        assert_pretrained_load_complete(
            CheckpointLoadResult(state="incomplete", unexpected_keys=("head.coco80.bias",))
        )
    with pytest.raises(AdapterGateError, match="missing"):
        assert_pretrained_load_complete(
            CheckpointLoadResult(state="missing", source="/absent.pth")
        )


def test_absent_checkpoint_result_cannot_read_as_success():
    """The historical defect: a never-assigned load result passing silently."""
    with pytest.raises(AdapterGateError, match="cannot be treated as a successful load"):
        assert_pretrained_load_complete(None)
    with pytest.raises(AdapterGateError, match="cannot be treated as a successful load"):
        assert_pretrained_load_complete({"missing_keys": [], "unexpected_keys": []})


def test_checkpoint_state_must_be_one_of_the_four_declared_states():
    with pytest.raises(ValueError, match="unknown checkpoint state"):
        CheckpointLoadResult(state="probably_fine")


def test_every_supported_adapter_implements_the_validation_contract():
    """A family adapter that omits a check cannot even be constructed."""
    from src.models.registry import MODEL_CONFIGS, create_adapter

    for model_id in MODEL_CONFIGS:
        adapter = create_adapter(model_id, device="cpu", repo_root=Path(__file__).parents[1])
        assert isinstance(adapter, AdapterValidationContract), model_id
    required = {
        "build_for_validation",
        "checkpoint_load_result",
        "feature_maps",
        "head_class_count",
        "forward_backward",
        "checkpoint_roundtrip",
    }
    assert required <= set(AdapterValidationContract.__abstractmethods__)


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


def test_wellformed_predictions_use_one_based_coco_categories():
    """Adapters export COCO category ids, so 0 and num_classes+1 are defects."""
    assert_wellformed_predictions(
        [{"boxes": [[1, 1, 5, 5]], "scores": [0.5], "labels": [2]}], num_classes=2
    )
    with pytest.raises(AdapterGateError, match="one-based"):
        assert_wellformed_predictions(
            [{"boxes": [[1, 1, 5, 5]], "scores": [0.5], "labels": [0]}], num_classes=2
        )


def _passing_checks():
    return [smoke_check_result(name, passed=True) for name in SMOKE_CHECK_ORDER]


def _record(checks=None, **overrides):
    return build_smoke_record(
        MODEL_ID,
        overrides.pop("fingerprint", _fingerprint()),
        _passing_checks() if checks is None else checks,
        gpu=overrides.pop("gpu", "Tesla T4"),
        dataset_track=overrides.pop("dataset_track", "2class"),
        image_size=overrides.pop("image_size", 640),
        **overrides,
    )


def test_build_smoke_record_signs_and_orders_checks():
    record = _record()
    assert record["status"] == "READY"
    assert record["artifact_kind"] == "gpu_adapter_smoke"
    assert record["smoke_contract_version"] == SMOKE_CONTRACT_VERSION
    assert len(record["signature"]) == 64
    # A single failed check flips the record to FAILED_ADAPTER.
    failed = _passing_checks()
    failed[4]["passed"] = False
    assert _record(failed)["status"] == "FAILED_ADAPTER"


def test_partial_smoke_run_records_a_failure_instead_of_raising():
    """An aborted run must leave complete evidence, never only a traceback."""
    error = RuntimeError("selective_scan kernel is unavailable")
    checks = [
        *_passing_checks()[:1],
        smoke_check_result("pretrained_weights_complete", passed=False, error=error),
    ]
    record = _record(checks, failure={"check": "pretrained_weights_complete"})

    assert record["status"] == "FAILED_ADAPTER"
    assert record["checks"][1]["exception_type"] == "RuntimeError"
    assert "selective_scan" in record["checks"][1]["error"]
    assert any("stopped before completing" in reason for reason in record["reasons"])
    assert record["created_at"]
    assert record["fingerprint"]["gpu"] == "Test GPU"


def test_ready_requires_every_check_even_when_none_failed():
    record = _record(_passing_checks()[:3])
    assert record["status"] == "FAILED_ADAPTER"
    assert "checkpoint_roundtrip" in record["reasons"][0]


def test_smoke_check_names_are_restricted_to_the_declared_contract():
    with pytest.raises(AdapterGateError, match="unknown smoke check"):
        smoke_check_result("looks_fine_to_me", passed=True)


def test_stale_or_mismatched_ready_record_cannot_authorize_a_run(tmp_path: Path):
    current = _fingerprint()
    assert smoke_gate_blockers(
        _record(), current, dataset_track="2class", image_size=640
    ) == []
    assert smoke_gate_blockers(
        None, current, dataset_track="2class", image_size=640
    ) == ["no adapter smoke record exists"]
    # A different track, resolution, commit, or contract version all block.
    assert smoke_gate_blockers(
        _record(), current, dataset_track="10class", image_size=640
    )
    assert smoke_gate_blockers(
        _record(), current, dataset_track="2class", image_size=1024
    )
    moved = dict(current, git_commit="a-different-commit")
    assert smoke_gate_blockers(
        _record(), moved, dataset_track="2class", image_size=640
    )
    tampered = {**_record(), "status": "READY", "reasons": []}
    tampered["checks"][3]["passed"] = False
    assert any(
        "signature" in reason
        for reason in smoke_gate_blockers(
            tampered, current, dataset_track="2class", image_size=640
        )
    )


def test_require_ready_adapter_gate_blocks_and_names_the_command(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    with pytest.raises(AdapterGateError, match="scripts.gpu_adapter_smoke"):
        require_ready_adapter_gate(
            repo_root, tmp_path, MODEL_ID, dataset_track="2class", image_size=640
        )
    assert smoke_record_path(tmp_path, MODEL_ID, "2class").name == (
        f"{MODEL_ID}__2class__smoke.json"
    )
