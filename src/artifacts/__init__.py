"""Versioned artifact readers and dual-write producers."""

from src.artifacts.checkpoints import (
    load_checkpoint_artifact,
    write_checkpoint_artifact,
)
from src.artifacts.identity import ArtifactIdentity
from src.artifacts.metrics import (
    load_metric_artifact,
    load_prediction_artifact,
    write_metric_artifact,
    write_prediction_artifact,
)

__all__ = [
    "ArtifactIdentity",
    "load_checkpoint_artifact",
    "load_metric_artifact",
    "load_prediction_artifact",
    "write_checkpoint_artifact",
    "write_metric_artifact",
    "write_prediction_artifact",
]
