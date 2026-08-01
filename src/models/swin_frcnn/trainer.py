"""Swin Faster R-CNN entry point onto the shared training engine."""

from __future__ import annotations

from src.training.engine import TrainingEngine


class SwinSharedTrainer(TrainingEngine):
    """Typed Swin boundary; framework callbacks remain dependency-injected."""
