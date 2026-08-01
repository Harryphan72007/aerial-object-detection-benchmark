"""RT-DETRv2 recipe-v2 optimizer construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.models.rtdetrv2.parameter_groups import (
    ParameterGroupDiscovery,
    discover_parameter_groups,
)
from src.utils.serialization import read_yaml

RECIPE_VERSION = "rtdetr_recipe_v2"


@dataclass(frozen=True)
class OptimizerRecipe:
    detector_learning_rate: float
    backbone_lr_multiplier: float
    weight_decay: float
    gradient_clip_norm: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OptimizerRecipe":
        if value.get("recipe_version") != RECIPE_VERSION:
            raise ValueError(f"expected recipe_version={RECIPE_VERSION!r}")
        recipe = cls(
            detector_learning_rate=float(value["detector_learning_rate"]),
            backbone_lr_multiplier=float(value["backbone_lr_multiplier"]),
            weight_decay=float(value["weight_decay"]),
            gradient_clip_norm=float(value["gradient_clip_norm"]),
        )
        if recipe.detector_learning_rate <= 0:
            raise ValueError("detector_learning_rate must be positive")
        if not 0 < recipe.backbone_lr_multiplier <= 1:
            raise ValueError("backbone_lr_multiplier must be in (0, 1]")
        if recipe.weight_decay < 0 or recipe.gradient_clip_norm <= 0:
            raise ValueError("weight decay must be non-negative and clipping positive")
        return recipe


@dataclass(frozen=True)
class OptimizerBuild:
    optimizer: Any
    discovery: ParameterGroupDiscovery
    report: dict[str, Any]


def load_optimizer_recipe(path: str | Path) -> dict[str, Any]:
    value = read_yaml(path)
    OptimizerRecipe.from_mapping(value)
    return value


def checked_in_recipe(repo_root: str | Path, profile: str = "performance") -> dict[str, Any]:
    if profile == "performance":
        relative = Path("configs/rtdetrv2_l/performance_recipe_v2.yaml")
    elif profile == "smoke":
        relative = Path("configs/rtdetrv2_l/smoke_recipe_v2.yaml")
    else:
        raise ValueError(f"unknown RT-DETR recipe profile: {profile}")
    return load_optimizer_recipe(Path(repo_root) / relative)


def build_optimizer(
    model: Any,
    recipe_value: Mapping[str, Any],
    *,
    optimizer_class: Any | None = None,
) -> OptimizerBuild:
    recipe = OptimizerRecipe.from_mapping(recipe_value)
    discovery = discover_parameter_groups(model)
    if not discovery.backbone or not discovery.detector:
        raise ValueError("recipe-v2 requires non-empty backbone and detector groups")
    groups = [
        {
            "name": "backbone",
            "params": list(discovery.backbone),
            "lr": recipe.detector_learning_rate * recipe.backbone_lr_multiplier,
            "initial_lr": recipe.detector_learning_rate
            * recipe.backbone_lr_multiplier,
            "lr_scale": recipe.backbone_lr_multiplier,
            "weight_decay": recipe.weight_decay,
        },
        {
            "name": "detector",
            "params": list(discovery.detector),
            "lr": recipe.detector_learning_rate,
            "initial_lr": recipe.detector_learning_rate,
            "lr_scale": 1.0,
            "weight_decay": recipe.weight_decay,
        },
    ]
    if optimizer_class is None:
        try:
            from torch.optim import AdamW
        except ImportError as exc:
            raise RuntimeError("PyTorch is required to build the RT-DETR optimizer") from exc
        optimizer_class = AdamW
    optimizer = optimizer_class(groups)
    report = {
        **discovery.report,
        "policy": RECIPE_VERSION,
        "differential_lr_enabled": True,
        "detector_learning_rate": recipe.detector_learning_rate,
        "backbone_lr_multiplier": recipe.backbone_lr_multiplier,
        "backbone_learning_rate": (
            recipe.detector_learning_rate * recipe.backbone_lr_multiplier
        ),
        "weight_decay": recipe.weight_decay,
        "gradient_clip_norm": recipe.gradient_clip_norm,
    }
    return OptimizerBuild(optimizer=optimizer, discovery=discovery, report=report)
