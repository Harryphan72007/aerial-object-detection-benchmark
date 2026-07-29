"""Learning-rate-only search primitives for the controlled VisDrone benchmark.

The previous multidimensional Optuna space was intentionally removed. Search is
now a deterministic logarithmic grid with explicit successive-halving rungs.
"""
from __future__ import annotations

from src.training.lr_search import (
    PROMOTION_RUNGS,
    boundary_extension_candidates,
    boundary_status,
    candidate_id,
    exponential_moving_average,
    generate_lr_candidates,
    rank_candidates,
)

__all__ = [
    "PROMOTION_RUNGS",
    "boundary_extension_candidates",
    "boundary_status",
    "candidate_id",
    "exponential_moving_average",
    "generate_lr_candidates",
    "rank_candidates",
]
