from __future__ import annotations

import pytest

from src.models.rtdetrv2.optimizer import build_optimizer
from src.models.rtdetrv2.scheduler import (
    WarmupCosineScheduler,
    learning_rate_trace,
)


class _Parameter:
    requires_grad = True

    def numel(self) -> int:
        return 5


class _Model:
    def named_parameters(self):
        yield "model.backbone.stem.weight", _Parameter()
        yield "model.decoder.head.weight", _Parameter()


class _Optimizer:
    def __init__(self, groups):
        self.param_groups = groups


RECIPE = {
    "recipe_version": "rtdetr_recipe_v2",
    "detector_learning_rate": 1e-4,
    "backbone_lr_multiplier": 0.1,
    "weight_decay": 0.05,
    "gradient_clip_norm": 0.1,
}


def test_optimizer_uses_reviewed_differential_learning_rates() -> None:
    result = build_optimizer(_Model(), RECIPE, optimizer_class=_Optimizer)
    groups = {group["name"]: group for group in result.optimizer.param_groups}
    assert groups["backbone"]["lr"] == 1e-5
    assert groups["detector"]["lr"] == 1e-4
    assert result.report["differential_lr_enabled"] is True
    assert result.report["trainable_parameter_tensors"] == 2


def test_search_and_full_run_share_exact_early_lr_trajectory() -> None:
    kwargs = {
        "total_updates": 1000,
        "warmup_updates": 100,
        "warmup_start_factor": 0.01,
        "minimum_factor": 0.01,
    }
    search = learning_rate_trace([1e-5, 1e-4], steps=50, **kwargs)
    full = learning_rate_trace([1e-5, 1e-4], steps=1000, **kwargs)
    assert search == full[:50]
    assert search[0] == pytest.approx((1e-7, 1e-6))
    assert all(search[index][1] < search[index + 1][1] for index in range(49))


def test_scheduler_steps_once_and_round_trips_state() -> None:
    optimizer = _Optimizer([{"lr": 1e-5}, {"lr": 1e-4}])
    scheduler = WarmupCosineScheduler(
        optimizer, total_updates=10, warmup_updates=2
    )
    assert optimizer.param_groups[1]["lr"] == pytest.approx(1e-6)
    scheduler.step()
    assert optimizer.param_groups[1]["lr"] > 1e-6
    state = scheduler.state_dict()
    restored_optimizer = _Optimizer([{"lr": 1e-5}, {"lr": 1e-4}])
    restored = WarmupCosineScheduler(
        restored_optimizer, total_updates=10, warmup_updates=2
    )
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    assert restored_optimizer.param_groups[1]["lr"] == optimizer.param_groups[1]["lr"]
