"""The controlled protocol has one source of truth: configs/controlled/benchmark.yaml.

PR-02 introduces this contract. It asserts that every controlled-track model
resolves the *same* frozen protocol (resolution / batch / accumulation /
effective batch / seed / trial count / final seeds), and that a per-model config
cannot silently override a frozen field. The only permitted per-model epoch
deviation is the explicit, recorded ``model_epoch_overrides`` block (removed in
PR-03). Learning rate is the only tuned value and is not part of this protocol.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.benchmark_tracks import (
    PROTOCOL_FROZEN_FIELDS,
    load_protocol,
    reject_frozen_protocol_overrides,
    resolve_controlled_protocol,
)

ROOT = Path(__file__).resolve().parents[1]

CONTROLLED_MODELS = (
    "faster_rcnn_resnet50",
    "faster_rcnn_swin_t",
    "faster_rcnn_vmamba_t",
    "rtdetrv2_l",
)

# Fields that must be byte-identical across every controlled-track model.
SHARED_FIELDS = (
    "protocol_id",
    "image_size",
    "batch_size",
    "gradient_accumulation_steps",
    "effective_batch_size",
    "use_amp",
    "search_seed",
    "phase_trials",
    "phase_a_epochs",
    "final_seeds",
)


def test_frozen_protocol_is_identical_across_all_controlled_models() -> None:
    resolved = {
        model_id: resolve_controlled_protocol(ROOT, model_id)
        for model_id in CONTROLLED_MODELS
    }
    reference = {
        field: resolved[CONTROLLED_MODELS[0]][field] for field in SHARED_FIELDS
    }
    for model_id in CONTROLLED_MODELS:
        shared = {field: resolved[model_id][field] for field in SHARED_FIELDS}
        assert shared == reference, model_id


def test_effective_batch_is_consistent_with_batch_and_accumulation() -> None:
    protocol = load_protocol(ROOT, "controlled")
    assert (
        int(protocol["batch_size"]) * int(protocol["gradient_accumulation_steps"])
        == int(protocol["effective_batch_size"])
    )


@pytest.mark.parametrize("field", sorted(PROTOCOL_FROZEN_FIELDS))
def test_per_model_config_cannot_override_a_frozen_field(field: str) -> None:
    with pytest.raises(ValueError, match="frozen protocol"):
        reject_frozen_protocol_overrides({field: 123})


def test_only_learning_rate_may_vary_between_models() -> None:
    """No frozen protocol field is exposed as a per-model degree of freedom."""
    resolved = {
        model_id: resolve_controlled_protocol(ROOT, model_id)
        for model_id in CONTROLLED_MODELS
    }
    # Every non-epoch protocol field is shared; the only permitted deviation is
    # the recorded epoch override (asserted equal in PR-03's stronger contract).
    for field in SHARED_FIELDS:
        values = {resolved[m][field] for m in CONTROLLED_MODELS}
        assert len(values) == 1, (field, values)
