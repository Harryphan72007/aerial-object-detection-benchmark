"""VMamba Faster R-CNN entry point onto the shared training engine."""

from __future__ import annotations

from src.training.engine import TrainingEngine


class VMambaSharedTrainer(TrainingEngine):
    """Typed VMamba boundary; framework callbacks remain dependency-injected."""
