"""Update-based warm-up plus cosine scheduling for RT-DETR recipe v2."""

from __future__ import annotations

import math
from typing import Any, Iterable


def learning_rate_factor(
    update: int,
    *,
    total_updates: int,
    warmup_updates: int,
    warmup_start_factor: float = 0.01,
    minimum_factor: float = 0.01,
) -> float:
    if total_updates < 1:
        raise ValueError("total_updates must be positive")
    if not 0 <= warmup_updates < total_updates:
        raise ValueError("warmup_updates must be in [0, total_updates)")
    if not 0 < warmup_start_factor <= 1 or not 0 <= minimum_factor <= 1:
        raise ValueError("scheduler factors must be in their unit intervals")
    bounded = min(max(int(update), 0), total_updates)
    if warmup_updates and bounded < warmup_updates:
        progress = bounded / warmup_updates
        return warmup_start_factor + (1.0 - warmup_start_factor) * progress
    decay_updates = total_updates - warmup_updates
    progress = (bounded - warmup_updates) / decay_updates
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_factor + (1.0 - minimum_factor) * cosine


class WarmupCosineScheduler:
    """Small serializable scheduler stepped once after each optimizer update."""

    def __init__(
        self,
        optimizer: Any,
        *,
        total_updates: int,
        warmup_updates: int,
        warmup_start_factor: float = 0.01,
        minimum_factor: float = 0.01,
    ) -> None:
        self.optimizer = optimizer
        self.total_updates = int(total_updates)
        self.warmup_updates = int(warmup_updates)
        self.warmup_start_factor = float(warmup_start_factor)
        self.minimum_factor = float(minimum_factor)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.update = 0
        self._apply()

    def _factor(self) -> float:
        return learning_rate_factor(
            self.update,
            total_updates=self.total_updates,
            warmup_updates=self.warmup_updates,
            warmup_start_factor=self.warmup_start_factor,
            minimum_factor=self.minimum_factor,
        )

    def _apply(self) -> None:
        factor = self._factor()
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * factor

    def step(self) -> None:
        self.update = min(self.update + 1, self.total_updates)
        self._apply()

    def state_dict(self) -> dict[str, Any]:
        return {
            "update": self.update,
            "total_updates": self.total_updates,
            "warmup_updates": self.warmup_updates,
            "warmup_start_factor": self.warmup_start_factor,
            "minimum_factor": self.minimum_factor,
            "base_lrs": list(self.base_lrs),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for key in ("total_updates", "warmup_updates"):
            if int(state[key]) != getattr(self, key):
                raise ValueError(f"scheduler {key} changed across resume")
        self.update = int(state["update"])
        self.base_lrs = [float(value) for value in state["base_lrs"]]
        self._apply()


def learning_rate_trace(
    base_lrs: Iterable[float],
    *,
    steps: int,
    total_updates: int,
    warmup_updates: int,
    warmup_start_factor: float = 0.01,
    minimum_factor: float = 0.01,
) -> list[tuple[float, ...]]:
    return [
        tuple(
            float(base_lr)
            * learning_rate_factor(
                update,
                total_updates=total_updates,
                warmup_updates=warmup_updates,
                warmup_start_factor=warmup_start_factor,
                minimum_factor=minimum_factor,
            )
            for base_lr in base_lrs
        )
        for update in range(steps)
    ]
