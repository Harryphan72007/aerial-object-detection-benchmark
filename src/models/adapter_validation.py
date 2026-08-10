"""The executable GPU-validation contract every model adapter must satisfy.

``DetectionModelAdapter`` describes inference. It cannot answer the question the
release gate actually asks: *does this adapter build the configured model, load
its pretrained weights, and take one real training step on this GPU?* This module
adds that second, explicit contract.

It is an ABC rather than an optional mixin on purpose. A family adapter that
forgets a check cannot be instantiated, so a missing method can never be silently
recorded as a passing smoke check.

The primitives here run on CPU with stub tensors and are unit-tested; the
adapter implementations that use them require the real framework stack and are
exercised by ``scripts/gpu_adapter_smoke.py`` on hardware.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# A checkpoint is either required and correctly loaded, required and broken, or
# genuinely not part of the recipe. "There is no checkpoint result" is not a
# state: it is the bug this contract exists to prevent.
CHECKPOINT_LOADED = "loaded"
CHECKPOINT_INCOMPLETE = "incomplete"
CHECKPOINT_MISSING = "missing"
CHECKPOINT_NOT_APPLICABLE = "not_applicable"
CHECKPOINT_STATES = (
    CHECKPOINT_LOADED,
    CHECKPOINT_INCOMPLETE,
    CHECKPOINT_MISSING,
    CHECKPOINT_NOT_APPLICABLE,
)

# Minimum fraction of the pretrained checkpoint's tensors that must be present,
# by value, inside the constructed backbone. Matching on values rather than key
# names is deliberate: every upstream loader renames keys, so a name-based check
# cannot distinguish "remapped correctly" from "not loaded at all", while a
# value-based check can. The thresholds are below 1.0 only for the tensors that
# legitimately do not transfer:
#   * faster_rcnn_resnet50 - the torchvision classifier head (fc.weight/fc.bias).
#   * faster_rcnn_swin_t   - the classifier head plus the ~9 patch-merging
#     tensors MMDetection's swin_converter reorders (values change, so they
#     cannot match by content).
#   * faster_rcnn_vmamba_t - the four classifier.* tensors of the VMamba-T
#     classification checkpoint.
# The measured coverage is always written into the smoke record, so a real GPU
# run reports the true number rather than only pass/fail.
MINIMUM_CHECKPOINT_VALUE_COVERAGE = {
    "faster_rcnn_resnet50": 0.95,
    "faster_rcnn_swin_t": 0.85,
    "faster_rcnn_vmamba_t": 0.95,
}
# Smallest tensor included in the coverage measurement. One- and two-element
# tensors collide across unrelated parameters and would inflate the fraction.
COVERAGE_MINIMUM_ELEMENTS = 16


@dataclass(frozen=True)
class CheckpointLoadResult:
    """What actually happened to the pretrained weights of one built model."""

    state: str
    source: str | None = None
    sha256: str | None = None
    missing_keys: tuple[str, ...] = ()
    unexpected_keys: tuple[str, ...] = ()
    value_coverage: float | None = None
    minimum_value_coverage: float | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.state not in CHECKPOINT_STATES:
            raise ValueError(
                f"unknown checkpoint state {self.state!r}; choose {CHECKPOINT_STATES}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "source": self.source,
            "sha256": self.sha256,
            "missing_keys": list(self.missing_keys),
            "unexpected_keys": list(self.unexpected_keys),
            "value_coverage": self.value_coverage,
            "minimum_value_coverage": self.minimum_value_coverage,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class FeatureMapSpec:
    """One backbone/neck output level, in NCHW, with its input stride."""

    shape: tuple[int, ...]
    stride: int

    def as_dict(self) -> dict[str, Any]:
        return {"shape": list(self.shape), "stride": int(self.stride)}


@dataclass(frozen=True)
class ForwardBackwardResult:
    """Evidence that one real optimizer-free training step executed."""

    loss: float
    trainable_parameters: int
    parameters_with_gradient: int
    maximum_absolute_gradient: float
    batch_size: int
    image_size: int
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "trainable_parameters": self.trainable_parameters,
            "parameters_with_gradient": self.parameters_with_gradient,
            "maximum_absolute_gradient": self.maximum_absolute_gradient,
            "batch_size": self.batch_size,
            "image_size": self.image_size,
            **self.extra,
        }


class AdapterValidationContract(ABC):
    """Every supported adapter answers the same seven GPU questions."""

    model_id: str

    @abstractmethod
    def build_for_validation(
        self, *, num_classes: int, image_size: int, device: str
    ) -> Any:
        """Construct the configured model through its real runtime path."""

    @abstractmethod
    def checkpoint_load_result(self) -> CheckpointLoadResult:
        """Describe what happened to the pretrained weights of the built model."""

    @abstractmethod
    def feature_maps(self, image_size: int) -> list[FeatureMapSpec]:
        """Return the architecture's multi-scale feature levels at ``image_size``."""

    @abstractmethod
    def head_class_count(self) -> int:
        """Return the class count of the *constructed* detection head."""

    @abstractmethod
    def forward_backward(
        self, *, image_size: int, batch_size: int = 2
    ) -> ForwardBackwardResult:
        """Run one forward + backward on a tiny synthetic batch."""

    @abstractmethod
    def checkpoint_roundtrip(self) -> tuple[dict[str, str], dict[str, str]]:
        """Save, reload into a compatible model, and return parameter digests."""

    def release(self) -> None:
        """Drop the built model and free accelerator memory."""


def tensor_digest(tensor: Any) -> str:
    """Content digest of one tensor, independent of its parameter name."""
    detached = tensor.detach().to("cpu").contiguous()
    return hashlib.sha256(
        f"{tuple(detached.shape)}|{detached.dtype}|".encode("utf-8")
        + detached.numpy().tobytes()
    ).hexdigest()


def _digest_set(tensors: Iterable[Any]) -> set[str]:
    digests: set[str] = set()
    for tensor in tensors:
        if getattr(tensor, "numel", None) is None:
            continue
        if int(tensor.numel()) < COVERAGE_MINIMUM_ELEMENTS:
            continue
        if not tensor.dtype.is_floating_point:
            continue
        digests.add(tensor_digest(tensor))
    return digests


def checkpoint_value_coverage(
    checkpoint_state: Mapping[str, Any], module: Any
) -> float:
    """Fraction of checkpoint tensors present, by value, inside ``module``.

    This is the load check that survives upstream key renaming: a remapped
    tensor keeps its values, a tensor that was never loaded does not.
    """
    wanted = _digest_set(checkpoint_state.values())
    if not wanted:
        raise ValueError("checkpoint contains no comparable floating-point tensors")
    present = _digest_set(
        [
            *(value for _, value in module.named_parameters()),
            *(value for _, value in module.named_buffers()),
        ]
    )
    return len(wanted & present) / len(wanted)


def parameter_digests(module: Any) -> dict[str, str]:
    """Map every parameter and buffer name to a content digest."""
    return {
        name: tensor_digest(value)
        for name, value in [
            *module.named_parameters(),
            *module.named_buffers(),
        ]
    }


def unwrap_checkpoint_state(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the tensor mapping from the usual checkpoint container shapes."""
    for key in ("state_dict", "model", "module"):
        value = payload.get(key)
        if isinstance(value, Mapping) and value:
            return dict(value)
    return dict(payload)


def filtered_keys(keys: Sequence[str], allowed_prefixes: Sequence[str]) -> list[str]:
    """Drop keys whose absence/excess the family contract declares expected."""
    return [
        key
        for key in keys
        if not any(key.startswith(prefix) for prefix in allowed_prefixes)
    ]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
