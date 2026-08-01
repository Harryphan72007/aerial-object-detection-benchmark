"""Serializable shared training state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass
class TrainingState:
    epoch: int = 0
    global_micro_step: int = 0
    optimizer_step: int = 0
    best_metrics: dict[str, float] = field(default_factory=dict)
    stopped_early: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingState":
        allowed = {"epoch", "global_micro_step", "optimizer_step", "best_metrics", "stopped_early"}
        if set(value) != allowed:
            raise ValueError("training state fields do not match schema v1")
        state = cls(
            epoch=int(value["epoch"]),
            global_micro_step=int(value["global_micro_step"]),
            optimizer_step=int(value["optimizer_step"]),
            best_metrics={str(k): float(v) for k, v in dict(value["best_metrics"]).items()},
            stopped_early=bool(value["stopped_early"]),
        )
        if min(state.epoch, state.global_micro_step, state.optimizer_step) < 0:
            raise ValueError("training counters must be non-negative")
        return state
