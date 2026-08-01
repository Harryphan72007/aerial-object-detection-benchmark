from __future__ import annotations

import pytest

from src.training.accumulation import effective_batch, optimizer_updates
from src.training.ema import ModelEMA, ema_checkpoint_fields, load_ema_checkpoint
from src.training.engine import TrainingEngine
from src.training.state import TrainingState


class _Model:
    def __init__(self, weight: float, counter: int = 0) -> None:
        self.values = {"weight": weight, "counter": counter}

    def state_dict(self):
        return dict(self.values)

    def load_state_dict(self, state, strict=True):
        assert strict is True
        self.values = dict(state)


class _Counter:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1

    def zero_grad(self) -> None:
        return None


def test_effective_batch_and_optimizer_update_count() -> None:
    report = effective_batch(2, 4, 2)
    assert report.effective_batch_size == 16
    assert report.as_dict()["gradient_accumulation_steps"] == 4
    assert optimizer_updates(9, 4) == 3
    with pytest.raises(ValueError):
        effective_batch(0, 4)


def test_engine_steps_optimizer_scheduler_and_ema_only_at_boundaries() -> None:
    optimizer = _Counter()
    scheduler = _Counter()
    ema_updates = _Counter()
    state = TrainingState()
    result = TrainingEngine(accumulation_steps=2).run_epoch(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        state,
        forward_loss=float,
        backward=lambda loss: None,
        optimizer=optimizer,
        scheduler=scheduler,
        after_optimizer_step=ema_updates.step,
    )
    assert result.optimizer_steps == 3
    assert optimizer.steps == scheduler.steps == ema_updates.steps == 3


def test_ema_round_trip_and_raw_ema_evaluation_are_separate() -> None:
    model = _Model(2.0, 1)
    ema = ModelEMA(model, decay=0.5)
    model.values = {"weight": 4.0, "counter": 2}
    ema.update(model)
    assert ema.shadow == {"weight": 3.0, "counter": 2}
    checkpoint = ema_checkpoint_fields(ema)
    restored = ModelEMA(model, decay=0.5)
    assert load_ema_checkpoint(restored, checkpoint) is True
    assert restored.state_dict() == ema.state_dict()
    raw = dict(model.values)
    with restored.average_parameters(model):
        assert model.values == {"weight": 3.0, "counter": 2}
    assert model.values == raw
