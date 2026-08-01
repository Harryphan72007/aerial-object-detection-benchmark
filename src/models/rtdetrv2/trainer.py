"""RT-DETRv2 entry point onto the shared training engine."""

from __future__ import annotations

from src.training.engine import TrainingEngine


class RTDetrSharedTrainer(TrainingEngine):
    """Typed RT-DETR boundary; tensor callbacks remain dependency-injected."""
