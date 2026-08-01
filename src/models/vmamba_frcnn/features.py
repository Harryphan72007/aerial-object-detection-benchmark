"""Validate VMamba stage outputs before their FPN consumer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class VMambaFeatureValidator:
    def __init__(self, channels: Sequence[int] = (96, 192, 384, 768)) -> None:
        self.channels = tuple(int(value) for value in channels)

    def validate(self, features: Sequence[Any] | Mapping[str, Any]) -> tuple[Any, ...]:
        values = tuple(features.values()) if isinstance(features, Mapping) else tuple(features)
        if len(values) != len(self.channels):
            raise ValueError(f"expected {len(self.channels)} VMamba stages")
        previous: tuple[int, int] | None = None
        batch: int | None = None
        for index, (feature, channels) in enumerate(zip(values, self.channels)):
            shape = tuple(int(value) for value in feature.shape)
            if len(shape) != 4 or shape[1] != channels:
                raise ValueError(
                    f"VMamba stage {index} must be NCHW with {channels} channels, got {shape}"
                )
            if min(shape[2:]) <= 0:
                raise ValueError(f"VMamba stage {index} has empty spatial dimensions")
            if batch is None:
                batch = shape[0]
            elif shape[0] != batch:
                raise ValueError("VMamba stages must share a batch size")
            spatial = (shape[2], shape[3])
            if previous is not None and (
                spatial[0] > previous[0] or spatial[1] > previous[1]
            ):
                raise ValueError("VMamba stage resolutions must be non-increasing")
            previous = spatial
        return values
