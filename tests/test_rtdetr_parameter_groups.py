from __future__ import annotations

from pathlib import Path

import pytest

from src.models.rtdetrv2.parameter_groups import (
    attach_parameter_group_report,
    discover_parameter_groups,
    write_parameter_group_report,
)
from src.utils.serialization import read_json


class _Parameter:
    def __init__(self, count: int, requires_grad: bool = True) -> None:
        self.count = count
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self.count


class _Model:
    def __init__(self, rows: list[tuple[str, _Parameter]]) -> None:
        self.rows = rows

    def named_parameters(self):
        yield from self.rows


def test_every_trainable_parameter_is_assigned_once_and_counts_sum(tmp_path: Path) -> None:
    model = _Model(
        [
            ("model.backbone.conv.weight", _Parameter(10)),
            ("model.encoder.backbone.stage.0", _Parameter(20)),
            ("model.encoder.layers.0.weight", _Parameter(30)),
            ("model.decoder.class_embed.weight", _Parameter(40)),
            ("frozen.weight", _Parameter(100, False)),
        ]
    )
    discovery = discover_parameter_groups(model)
    report = discovery.report
    assert report["trainable_parameter_tensors"] == 4
    assert report["trainable_parameter_values"] == 100
    assert report["groups"]["backbone"]["parameter_values"] == 30
    assert report["groups"]["detector"]["parameter_values"] == 70
    assert sum(group["tensor_count"] for group in report["groups"].values()) == 4
    assert report["differential_lr_enabled"] is False
    path = tmp_path / "parameter_groups.json"
    write_parameter_group_report(path, discovery)
    assert read_json(path) == report
    assert attach_parameter_group_report({"status": "running"}, discovery)[
        "parameter_group_report"
    ] == report


def test_duplicate_trainable_parameter_is_rejected() -> None:
    shared = _Parameter(5)
    with pytest.raises(ValueError, match="duplicate"):
        discover_parameter_groups(_Model([("backbone.a", shared), ("decoder.a", shared)]))


def test_unassigned_trainable_parameter_is_rejected() -> None:
    with pytest.raises(ValueError, match="unassigned"):
        discover_parameter_groups(_Model([("", _Parameter(5))]))
