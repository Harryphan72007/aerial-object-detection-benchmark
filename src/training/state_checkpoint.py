"""Atomic JSON checkpoints for shared engine state and serializable components."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.training.state import TrainingState
from src.utils.serialization import read_json, write_json


def save_training_state(
    path: str | Path,
    state: TrainingState,
    *,
    components: Mapping[str, Any] | None = None,
) -> None:
    payload = {
        "schema_version": 1,
        "training_state": state.to_dict(),
        "components": {
            name: component.state_dict() for name, component in (components or {}).items()
        },
    }
    write_json(path, payload)


def load_training_state(
    path: str | Path,
    *,
    components: Mapping[str, Any] | None = None,
) -> TrainingState:
    payload = read_json(path)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported training-state checkpoint schema")
    saved_components = payload.get("components", {})
    requested = components or {}
    if set(saved_components) != set(requested):
        raise ValueError("checkpoint component set does not match runtime")
    for name, component in requested.items():
        component.load_state_dict(saved_components[name])
    return TrainingState.from_dict(payload["training_state"])
