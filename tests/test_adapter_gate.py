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
