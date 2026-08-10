"""One HPO objective policy for every model family, RT-DETRv2 included.

RT-DETRv2 previously read its objectives with ``manifest.get(key, 0.0)`` while
every other family raised. A missing metric therefore became a legitimate-looking
score of zero, and the failed evaluation entered Optuna as a COMPLETE trial.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.hpo.workflow import (
    InvalidObjectiveError,
    _failure_kind,
    validated_objective_pair,
)

MODELS = (
    "faster_rcnn_resnet50",
    "faster_rcnn_swin_t",
    "faster_rcnn_vmamba_t",
    "rtdetrv2_l",
)
INVALID_MANIFESTS = {
    "missing both keys": ({}, "missing"),
    "missing one key": ({"best_validation_map": 0.2}, "missing"),
    "null value": (
        {"best_validation_map": 0.2, "best_validation_aptiny": None},
        "null",
    ),
    "non-numeric value": (
        {"best_validation_map": 0.2, "best_validation_aptiny": "0.1"},
        "non-numeric",
    ),
    "boolean value": (
        {"best_validation_map": True, "best_validation_aptiny": 0.1},
        "non-numeric",
    ),
    "NaN": (
        {"best_validation_map": float("nan"), "best_validation_aptiny": 0.1},
        "NaN",
    ),
    "positive infinity": (
        {"best_validation_map": float("inf"), "best_validation_aptiny": 0.1},
        "infinite",
    ),
    "negative infinity": (
        {"best_validation_map": 0.2, "best_validation_aptiny": float("-inf")},
        "infinite",
    ),
}


@pytest.mark.parametrize("model_id", MODELS)
@pytest.mark.parametrize(
    ("manifest", "reason"),
    list(INVALID_MANIFESTS.values()),
    ids=list(INVALID_MANIFESTS),
)
def test_every_family_rejects_an_unusable_objective_pair(
    model_id: str, manifest: dict[str, object], reason: str
) -> None:
    with pytest.raises(InvalidObjectiveError, match=reason):
        validated_objective_pair(manifest, model_id)


@pytest.mark.parametrize("model_id", MODELS)
def test_every_family_rejects_the_impossible_all_zero_pair(model_id: str) -> None:
    with pytest.raises(InvalidObjectiveError, match="all-zero"):
        validated_objective_pair(
            {"best_validation_map": 0.0, "best_validation_aptiny": 0.0}, model_id
        )


@pytest.mark.parametrize("model_id", MODELS)
def test_every_family_accepts_a_valid_objective_pair(model_id: str) -> None:
    assert validated_objective_pair(
        {"best_validation_map": 0.21, "best_validation_aptiny": 0.07}, model_id
    ) == (0.21, 0.07)
    # A zero secondary objective is legitimate; only the all-zero pair is not.
    assert validated_objective_pair(
        {"best_validation_map": 0.21, "best_validation_aptiny": 0.0}, model_id
    ) == (0.21, 0.0)


def test_rtdetr_missing_objectives_no_longer_become_zero() -> None:
    """The exact defect: RT-DETRv2 must not default a missing metric to 0.0."""
    with pytest.raises(InvalidObjectiveError):
        validated_objective_pair({}, "rtdetrv2_l")
    with pytest.raises(InvalidObjectiveError):
        validated_objective_pair({"best_validation_map": 0.3}, "rtdetrv2_l")


def test_error_names_the_trial_model_manifest_and_invalid_objectives(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "final_metrics.json"

    with pytest.raises(InvalidObjectiveError) as raised:
        validated_objective_pair(
            {"best_validation_map": float("nan")},
            "rtdetrv2_l",
            trial_number=7,
            manifest_path=manifest_path,
        )

    message = str(raised.value)
    assert "trial=7" in message
    assert "model_id=rtdetrv2_l" in message
    assert str(manifest_path) in message
    assert "best_validation_map: NaN" in message
    assert "best_validation_aptiny: missing" in message


def test_an_unusable_objective_is_fatal_and_never_pruned_as_divergence() -> None:
    """A NaN *objective* is a defect; a NaN *loss* is a prunable candidate."""
    assert _failure_kind(InvalidObjectiveError("... best_validation_map: NaN")) is None
    assert _failure_kind(RuntimeError("loss became nan")) == "numerical_divergence"
    assert _failure_kind(RuntimeError("CUDA out of memory")) == "out_of_memory"


def test_rtdetr_backend_does_not_write_zero_for_a_non_finite_metric() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_rtdetr_training.py"
    ).read_text(encoding="utf-8")

    assert "np.isfinite(best.best_map) else 0.0" not in source
    assert "without a finite validation objective" in source
