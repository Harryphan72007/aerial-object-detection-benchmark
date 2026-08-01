"""Framework-neutral accumulation and state-transition engine."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from src.training.state import TrainingState
from src.training.accumulation import is_optimizer_boundary


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    losses: tuple[float, ...]
    optimizer_steps: int


class TrainingEngine:
    def __init__(self, *, accumulation_steps: int = 1, use_amp: bool = False) -> None:
        if accumulation_steps <= 0:
            raise ValueError("accumulation_steps must be positive")
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp

    def run_epoch(
        self,
        batches: Iterable[Any],
        state: TrainingState,
        *,
        forward_loss: Callable[[Any], Any],
        backward: Callable[[Any], None],
        optimizer: Any,
        scheduler: Any | None = None,
        amp_context: Callable[[], Any] | None = None,
        after_optimizer_step: Callable[[], None] | None = None,
    ) -> EpochResult:
        materialized = tuple(batches)
        if not materialized:
            raise ValueError("cannot train an empty epoch")
        losses: list[float] = []
        steps_before = state.optimizer_step
        optimizer.zero_grad()
        for index, batch in enumerate(materialized, start=1):
            context = amp_context if self.use_amp and amp_context else nullcontext
            with context():
                loss = forward_loss(batch)
            backward(loss / self.accumulation_steps)
            losses.append(float(loss))
            state.global_micro_step += 1
            if is_optimizer_boundary(index, len(materialized), self.accumulation_steps):
                optimizer.step()
                optimizer.zero_grad()
                state.optimizer_step += 1
                if after_optimizer_step is not None:
                    after_optimizer_step()
                if scheduler is not None:
                    scheduler.step()
        state.epoch += 1
        return EpochResult(
            epoch=state.epoch,
            losses=tuple(losses),
            optimizer_steps=state.optimizer_step - steps_before,
        )
