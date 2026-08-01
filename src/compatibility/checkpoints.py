"""Checkpoint v2 loading-mode classification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class CheckpointLoadingMode(str, Enum):
    FULL_RESUME = "full_resume"
    WEIGHTS_ONLY = "weights_only"
    EVALUATION_ONLY = "evaluation_only"
    INCOMPATIBLE = "incompatible"


@dataclass(frozen=True)
class CheckpointCompatibility:
    mode: CheckpointLoadingMode
    reasons: tuple[str, ...]


MODEL_FIELDS = ("model_id", "model_signature")
RESUME_FIELDS = (
    "config_hash",
    "dataset_hash",
    "optimizer_signature",
    "scheduler_signature",
    "accumulation_steps",
    "seed",
)
FULL_STATE_KEYS = frozenset({"model", "optimizer", "scheduler", "training_state"})


def classify_checkpoint(
    metadata: Mapping[str, Any] | None,
    expected: Mapping[str, Any],
) -> CheckpointCompatibility:
    """Classify without loading or mutating any checkpoint tensors."""

    if metadata is None or metadata.get("schema_version") != 2:
        return CheckpointCompatibility(
            CheckpointLoadingMode.EVALUATION_ONLY,
            ("legacy checkpoint has no v2 resume contract",),
        )
    model_drift = tuple(
        field for field in MODEL_FIELDS if metadata.get(field) != expected.get(field)
    )
    if model_drift:
        return CheckpointCompatibility(
            CheckpointLoadingMode.INCOMPATIBLE,
            tuple(f"{field} mismatch" for field in model_drift),
        )
    state_keys = frozenset(str(value) for value in metadata.get("state_keys", []))
    if "model" not in state_keys:
        return CheckpointCompatibility(
            CheckpointLoadingMode.INCOMPATIBLE,
            ("model state is missing",),
        )
    resume_drift = tuple(
        field for field in RESUME_FIELDS if metadata.get(field) != expected.get(field)
    )
    missing_state = tuple(sorted(FULL_STATE_KEYS - state_keys))
    if resume_drift or missing_state:
        reasons = tuple(f"{field} mismatch" for field in resume_drift) + tuple(
            f"{field} state is missing" for field in missing_state
        )
        return CheckpointCompatibility(CheckpointLoadingMode.WEIGHTS_ONLY, reasons)
    return CheckpointCompatibility(CheckpointLoadingMode.FULL_RESUME, ())


def require_loading_mode(
    compatibility: CheckpointCompatibility,
    requested: CheckpointLoadingMode,
) -> None:
    if compatibility.mode != requested:
        raise ValueError(
            f"checkpoint classified as {compatibility.mode.value}, not {requested.value}: "
            + "; ".join(compatibility.reasons)
        )
