"""Single source of truth for the controlled benchmark contract."""
from __future__ import annotations

from typing import Any, Mapping

PRIMARY_MODELS = (
    "faster_rcnn_resnet50",
    "faster_rcnn_swin_t",
    "faster_rcnn_vmamba_t",
    "rtdetrv2_l",
)

BENCHMARK_CONTRACT: dict[str, Any] = {
    "protocol_id": "lr_controlled_v1",
    "dataset_track": "2class",
    "search_seed": 42,
    "image_size": 640,
    "effective_batch_size": 8,
    "amp": True,
    "learning_rate_candidates": 9,
    "search_rungs": (
        {"epoch": 2, "keep": 5},
        {"epoch": 5, "keep": 3},
        {"epoch": 10, "keep": 2},
        {"epoch": 15, "keep": 1},
    ),
    "final_epochs": 25,
    "primary_metric": "mAP_50_95",
    "secondary_metric": "APtiny",
}


def require_primary_model(model_id: str) -> None:
    if model_id not in PRIMARY_MODELS:
        raise ValueError(
            f"Unsupported MODEL_ID {model_id!r}. Choose one of: "
            + ", ".join(PRIMARY_MODELS)
        )


def validate_final_config(config: Mapping[str, Any], *, selected_lr: float | None = None) -> None:
    """Reject a run that is not the controlled final-training protocol."""
    expected = {
        "protocol_id": BENCHMARK_CONTRACT["protocol_id"],
        "dataset_track": BENCHMARK_CONTRACT["dataset_track"],
        "image_size": BENCHMARK_CONTRACT["image_size"],
        "seed": BENCHMARK_CONTRACT["search_seed"],
        "epochs": BENCHMARK_CONTRACT["final_epochs"],
        "scheduler_horizon": BENCHMARK_CONTRACT["final_epochs"],
        "effective_batch_size": BENCHMARK_CONTRACT["effective_batch_size"],
        "run_kind": "final_complete_official_train",
    }
    changed = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if (
            config.get(key, "lr_controlled_v1")
            if key == "protocol_id"
            else config.get(key)
        )
        != value
    }
    if changed:
        raise ValueError(f"Incompatible final run: {changed}")
    if not bool(config.get("use_amp")):
        raise ValueError("Incompatible final run: AMP must remain enabled")
    if selected_lr is not None:
        applied = config.get("overrides", {}).get("learning_rate")
        if applied is None or abs(float(applied) - selected_lr) > max(
            abs(selected_lr), 1e-12
        ) * 1e-12:
            raise ValueError("Incompatible final run: selected LR does not match")
