"""Framework-light exponential moving average with explicit checkpoint state."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any, Iterator, Mapping


def _clone(value: Any) -> Any:
    detach = getattr(value, "detach", None)
    candidate = detach() if callable(detach) else value
    clone = getattr(candidate, "clone", None)
    return clone() if callable(clone) else copy.deepcopy(candidate)


class ModelEMA:
    def __init__(self, model: Any, *, decay: float = 0.9998) -> None:
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = {
            str(name): _clone(value) for name, value in model.state_dict().items()
        }

    def update(self, model: Any) -> None:
        current = model.state_dict()
        if set(current) != set(self.shadow):
            raise ValueError("EMA model state keys changed")
        for name, value in current.items():
            target = self.shadow[name]
            floating = getattr(value, "is_floating_point", None)
            is_floating = bool(floating()) if callable(floating) else isinstance(value, float)
            if is_floating and hasattr(target, "mul_") and hasattr(target, "add_"):
                target.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            elif is_floating:
                self.shadow[name] = self.decay * target + (1.0 - self.decay) * value
            else:
                self.shadow[name] = _clone(value)
        self.num_updates += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": {name: _clone(value) for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        decay = float(state["decay"])
        if decay != self.decay:
            raise ValueError(
                f"EMA decay changed across resume: checkpoint={decay}, requested={self.decay}"
            )
        shadow = dict(state["shadow"])
        if set(shadow) != set(self.shadow):
            raise ValueError("EMA checkpoint state keys are incompatible")
        self.shadow = {name: _clone(value) for name, value in shadow.items()}
        self.num_updates = int(state["num_updates"])

    def copy_to(self, model: Any) -> None:
        model.load_state_dict(
            {name: _clone(value) for name, value in self.shadow.items()}, strict=True
        )

    @contextmanager
    def average_parameters(self, model: Any) -> Iterator[None]:
        raw = {name: _clone(value) for name, value in model.state_dict().items()}
        self.copy_to(model)
        try:
            yield
        finally:
            model.load_state_dict(raw, strict=True)


def ema_checkpoint_fields(ema: ModelEMA | None) -> dict[str, Any]:
    return {} if ema is None else {"ema_state_dict": ema.state_dict()}


def load_ema_checkpoint(ema: ModelEMA | None, checkpoint: Mapping[str, Any]) -> bool:
    state = checkpoint.get("ema_state_dict")
    if ema is None or state is None:
        return False
    ema.load_state_dict(state)
    return True
