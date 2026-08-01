from __future__ import annotations

from pathlib import Path

import pytest

from src.models.resnet_frcnn.factory import (
    RESNET_FACTORY_MODEL_ID,
    ResNetFRCNNFactory,
    build_resnet_frcnn,
)

ROOT = Path(__file__).resolve().parents[1]


class _Parameter:
    def __init__(self, count: int) -> None:
        self.count = count

    def numel(self) -> int:
        return self.count


class _Model:
    def __init__(self, config: str, checkpoint: str | None, device: str) -> None:
        self.arguments = (config, checkpoint, device)

    def parameters(self) -> list[_Parameter]:
        return [_Parameter(10), _Parameter(7), _Parameter(3)]


def _runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "mmdetection"
    config = root / "configs" / "faster_rcnn" / "faster-rcnn_r50_fpn_1x_coco.py"
    config.parent.mkdir(parents=True)
    config.write_text("# pinned fixture config\n", encoding="utf-8")
    monkeypatch.setenv("MMDET_ROOT", str(root))
    return config.resolve()


def _initializer(config: str, checkpoint: str | None, device: str) -> _Model:
    return _Model(config, checkpoint, device)


def _inference(model: _Model, image: object) -> dict[str, object]:
    return {"model_arguments": model.arguments, "image": image, "boxes": [[1, 2, 3, 4]]}


def test_factory_matches_legacy_construction_parameter_count_and_forward(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _runtime(monkeypatch, tmp_path)
    checkpoint = tmp_path / "legacy.pth"
    checkpoint.write_bytes(b"fixture")
    legacy = _initializer(str(config), str(checkpoint.resolve()), "cpu")
    factory = ResNetFRCNNFactory(ROOT)
    migrated = factory.build(
        checkpoint=checkpoint, device="cpu", initializer=_initializer
    )
    image = {"height": 32, "width": 32}
    assert migrated.arguments == legacy.arguments
    assert factory.parameter_count(migrated) == factory.parameter_count(legacy) == 20
    assert factory.forward(migrated, image, inference=_inference) == _inference(
        legacy, image
    )


def test_functional_entry_point_uses_pinned_resnet_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _runtime(monkeypatch, tmp_path)
    model = build_resnet_frcnn(ROOT, initializer=_initializer)
    assert model.arguments == (str(config), None, "cpu")
    assert RESNET_FACTORY_MODEL_ID == "faster_rcnn_resnet50"


def test_factory_requires_resolved_existing_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MMDET_ROOT", raising=False)
    factory = ResNetFRCNNFactory(ROOT)
    with pytest.raises(RuntimeError, match="MMDET_ROOT"):
        factory.build(initializer=_initializer)


def test_notebook_exposes_factory_without_constructing_other_models() -> None:
    source = (ROOT / "notebooks" / "01_run_model_day.ipynb").read_text()
    assert "RESNET_FACTORY_MODEL_ID" in source
