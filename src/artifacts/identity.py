"""Shared identity carried by versioned and legacy artifact views."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ArtifactIdentity:
    run_id: str
    model_id: str
    architecture_family: str
    dataset_track: str
    evaluation_resolution: int
    seed: int

    def __post_init__(self) -> None:
        for name in ("run_id", "model_id", "architecture_family"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if self.dataset_track not in {"2class", "10class"}:
            raise ValueError("dataset_track must be 2class or 10class")
        if self.evaluation_resolution <= 0:
            raise ValueError("evaluation_resolution must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
