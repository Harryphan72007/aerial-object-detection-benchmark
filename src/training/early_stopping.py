"""Persisted validation-metric early stopping."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0
    best_metric: float = float("-inf")
    best_epoch: int = 0
    bad_epochs: int = 0
    stopped: bool = False

    def __post_init__(self) -> None:
        if self.patience < 1 or self.min_delta < 0:
            raise ValueError("patience must be positive and min_delta non-negative")

    def update(self, epoch: int, metric: float) -> bool:
        if self.stopped:
            return True
        if float(metric) > self.best_metric + self.min_delta:
            self.best_metric = float(metric)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            self.stopped = self.bad_epochs >= self.patience
        return self.stopped

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        expected = {
            "patience",
            "min_delta",
            "best_metric",
            "best_epoch",
            "bad_epochs",
            "stopped",
        }
        if set(value) != expected:
            raise ValueError("early-stopping state fields do not match")
        if int(value["patience"]) != self.patience or float(
            value["min_delta"]
        ) != self.min_delta:
            raise ValueError("early-stopping policy changed across resume")
        self.best_metric = float(value["best_metric"])
        self.best_epoch = int(value["best_epoch"])
        self.bad_epochs = int(value["bad_epochs"])
        self.stopped = bool(value["stopped"])
