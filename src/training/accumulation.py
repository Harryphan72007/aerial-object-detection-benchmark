"""Gradient-accumulation accounting shared by all training backends."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EffectiveBatch:
    per_device_batch_size: int
    gradient_accumulation_steps: int
    world_size: int
    effective_batch_size: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def effective_batch(
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int = 1,
) -> EffectiveBatch:
    values = (per_device_batch_size, gradient_accumulation_steps, world_size)
    if any(isinstance(value, bool) or int(value) <= 0 for value in values):
        raise ValueError("batch, accumulation, and world size must be positive integers")
    return EffectiveBatch(
        per_device_batch_size=int(per_device_batch_size),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
        world_size=int(world_size),
        effective_batch_size=int(per_device_batch_size)
        * int(gradient_accumulation_steps)
        * int(world_size),
    )


def optimizer_updates(microbatches: int, accumulation_steps: int) -> int:
    if microbatches < 0 or accumulation_steps <= 0:
        raise ValueError("microbatches must be non-negative and accumulation positive")
    return math.ceil(microbatches / accumulation_steps)


def is_optimizer_boundary(
    microbatch_index: int,
    total_microbatches: int,
    accumulation_steps: int,
) -> bool:
    if not 1 <= microbatch_index <= total_microbatches:
        raise ValueError("microbatch_index is outside this epoch")
    return (
        microbatch_index % accumulation_steps == 0
        or microbatch_index == total_microbatches
    )
