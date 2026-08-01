"""ResNet-specific entry point onto the shared training engine."""

from __future__ import annotations

from src.training.engine import TrainingEngine


class ResNetSharedTrainer(TrainingEngine):
    """Typed migration boundary; framework callbacks remain dependency-injected."""
