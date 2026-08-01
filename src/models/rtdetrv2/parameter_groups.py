"""Diagnostic-only RT-DETRv2 trainable parameter-group discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.utils.serialization import write_json

BACKBONE_MARKERS = ("backbone", "encoder.backbone")


@dataclass(frozen=True)
class ParameterGroupDiscovery:
    backbone: tuple[Any, ...]
    detector: tuple[Any, ...]
    report: dict[str, Any]


def default_parameter_group(name: str) -> str | None:
    if not name:
        return None
    segments = name.lower().split(".")
    return "backbone" if any(marker in name.lower() for marker in BACKBONE_MARKERS) or "backbone" in segments else "detector"


def discover_parameter_groups(
    model: Any,
    *,
    classifier: Callable[[str], str | None] = default_parameter_group,
) -> ParameterGroupDiscovery:
    groups: dict[str, list[Any]] = {"backbone": [], "detector": []}
    names: dict[str, list[str]] = {"backbone": [], "detector": []}
    seen: dict[int, str] = {}
    unassigned: list[str] = []
    duplicate: list[tuple[str, str]] = []
    total_numel = 0
    for name, parameter in model.named_parameters():
        if not bool(getattr(parameter, "requires_grad", False)):
            continue
        identity = id(parameter)
        if identity in seen:
            duplicate.append((seen[identity], name))
            continue
        seen[identity] = name
        group = classifier(name)
        if group not in groups:
            unassigned.append(name)
            continue
        groups[group].append(parameter)
        names[group].append(name)
        total_numel += int(parameter.numel())
    if duplicate or unassigned:
        raise ValueError(
            f"invalid RT-DETR parameter grouping: duplicate={duplicate}, unassigned={unassigned}"
        )
    assigned_count = sum(len(values) for values in groups.values())
    if assigned_count != len(seen):
        raise ValueError("RT-DETR parameter counts do not sum exactly")
    report = {
        "schema_version": 1,
        "policy": "diagnostic_global_lr",
        "differential_lr_enabled": False,
        "trainable_parameter_tensors": len(seen),
        "trainable_parameter_values": total_numel,
        "groups": {
            group: {
                "names": sorted(names[group]),
                "tensor_count": len(groups[group]),
                "parameter_values": sum(int(value.numel()) for value in groups[group]),
            }
            for group in ("backbone", "detector")
        },
        "duplicate": [],
        "unassigned": [],
    }
    return ParameterGroupDiscovery(
        backbone=tuple(groups["backbone"]),
        detector=tuple(groups["detector"]),
        report=report,
    )


def write_parameter_group_report(path: str | Path, discovery: ParameterGroupDiscovery) -> None:
    write_json(path, discovery.report)


def attach_parameter_group_report(
    manifest: dict[str, Any], discovery: ParameterGroupDiscovery
) -> dict[str, Any]:
    return {**manifest, "parameter_group_report": discovery.report}
