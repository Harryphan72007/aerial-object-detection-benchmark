"""Validate and normalize Swin backbone feature maps for an FPN."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class SwinFeatureAdapter:
    """Convert per-stage NHWC tensors to the NCHW contract expected by FPN."""

    def __init__(self, channels: Sequence[int] = (96, 192, 384, 768)) -> None:
        self.channels = tuple(int(value) for value in channels)
        if not self.channels or any(value <= 0 for value in self.channels):
            raise ValueError("feature channels must be positive")

    def adapt(self, features: Sequence[Any] | Mapping[str, Any]) -> tuple[Any, ...]:
        values = tuple(features.values()) if isinstance(features, Mapping) else tuple(features)
        if len(values) != len(self.channels):
            raise ValueError(
                f"expected {len(self.channels)} feature levels, got {len(values)}"
            )
        normalized: list[Any] = []
        batch_size: int | None = None
        previous_height: int | None = None
        for index, (feature, expected_channels) in enumerate(zip(values, self.channels)):
            shape = tuple(int(value) for value in feature.shape)
            if len(shape) != 4:
                raise ValueError(f"feature level {index} must be rank-4, got {shape}")
            if shape[1] == expected_channels:
                converted = feature
            elif shape[-1] == expected_channels:
                converted = feature.permute(0, 3, 1, 2)
                if hasattr(converted, "contiguous"):
                    converted = converted.contiguous()
            else:
                raise ValueError(
                    f"feature level {index} has no channel dimension {expected_channels}: {shape}"
                )
            converted_shape = tuple(int(value) for value in converted.shape)
            if converted_shape[2] <= 0 or converted_shape[3] <= 0:
                raise ValueError(f"feature level {index} has empty spatial dimensions")
            if batch_size is None:
                batch_size = converted_shape[0]
            elif converted_shape[0] != batch_size:
                raise ValueError("all FPN features must have the same batch size")
            if previous_height is not None and converted_shape[2] > previous_height:
                raise ValueError("FPN feature resolutions must be non-increasing")
            previous_height = converted_shape[2]
            normalized.append(converted)
        return tuple(normalized)
