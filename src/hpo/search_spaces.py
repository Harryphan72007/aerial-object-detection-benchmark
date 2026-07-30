"""Backend-supported model-specific HPO search spaces."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

SEARCH_SPACES: dict[str, dict[str, dict[str, Any]]] = {
    "faster_rcnn_resnet50": {
        "learning_rate": {"kind": "float", "low": 1e-5, "high": 5e-4, "log": True},
        "weight_decay": {"kind": "float", "low": 1e-5, "high": 0.1, "log": True},
        "backbone_lr_multiplier": {
            "kind": "float",
            "low": 0.05,
            "high": 0.5,
        },
        "rpn_nms_threshold": {"kind": "float", "low": 0.5, "high": 0.9},
        "rpn_proposals": {"kind": "categorical", "choices": [1000, 2000, 3000]},
        "roi_nms_threshold": {"kind": "float", "low": 0.3, "high": 0.7},
        "p2_enabled": {"kind": "categorical", "choices": [True, False]},
    },
    "faster_rcnn_swin_t": {
        "learning_rate": {"kind": "float", "low": 5e-6, "high": 2e-4, "log": True},
        "weight_decay": {"kind": "float", "low": 1e-4, "high": 0.2, "log": True},
        "backbone_lr_multiplier": {
            "kind": "float",
            "low": 0.03,
            "high": 0.3,
        },
        "drop_path_rate": {"kind": "float", "low": 0.0, "high": 0.3},
        "roi_nms_threshold": {"kind": "float", "low": 0.3, "high": 0.7},
        "p2_enabled": {"kind": "categorical", "choices": [True, False]},
    },
    "faster_rcnn_vmamba_t": {
        "learning_rate": {"kind": "float", "low": 5e-6, "high": 2e-4, "log": True},
        "weight_decay": {"kind": "float", "low": 1e-4, "high": 0.2, "log": True},
        "backbone_lr_multiplier": {
            "kind": "float",
            "low": 0.03,
            "high": 0.3,
        },
        "drop_path_rate": {"kind": "float", "low": 0.0, "high": 0.3},
        "roi_nms_threshold": {"kind": "float", "low": 0.3, "high": 0.7},
        "p2_enabled": {"kind": "categorical", "choices": [True, False]},
    },
    "rtdetrv2_l": {
        "learning_rate": {"kind": "float", "low": 1e-6, "high": 2e-4, "log": True},
        "weight_decay": {"kind": "float", "low": 1e-5, "high": 0.1, "log": True},
        "gradient_clip": {"kind": "float", "low": 0.05, "high": 1.0},
        "num_queries": {"kind": "categorical", "choices": [200, 300, 500]},
        "decoder_layers": {"kind": "categorical", "choices": [4, 6]},
        "num_denoising": {"kind": "categorical", "choices": [50, 100, 200]},
        "dropout": {"kind": "float", "low": 0.0, "high": 0.2},
    },
}


def broad_search_space(model_id: str) -> dict[str, dict[str, Any]]:
    try:
        return deepcopy(SEARCH_SPACES[model_id])
    except KeyError as error:
        raise ValueError(f"unsupported HPO model: {model_id}") from error


def refined_search_space(
    broad: dict[str, dict[str, Any]],
    strongest_parameters: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not strongest_parameters:
        raise ValueError("Phase B requires at least one valid Phase A trial")
    refined: dict[str, dict[str, Any]] = {}
    for name, definition in broad.items():
        values = [row[name] for row in strongest_parameters if name in row]
        if not values:
            raise ValueError(f"valid Phase A trials are missing {name}")
        if definition["kind"] == "categorical":
            choices = [value for value in definition["choices"] if value in values]
            refined[name] = {"kind": "categorical", "choices": choices}
            continue
        low = float(definition["low"])
        high = float(definition["high"])
        observed_low = min(float(value) for value in values)
        observed_high = max(float(value) for value in values)
        if definition.get("log"):
            ratio = (high / low) ** 0.15
            narrowed_low = max(low, observed_low / ratio)
            narrowed_high = min(high, observed_high * ratio)
        else:
            padding = (high - low) * 0.1
            narrowed_low = max(low, observed_low - padding)
            narrowed_high = min(high, observed_high + padding)
        if narrowed_low >= narrowed_high:
            narrowed_low, narrowed_high = low, high
        refined[name] = {
            **definition,
            "low": narrowed_low,
            "high": narrowed_high,
        }
    return refined
