from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.models.swin_frcnn import (
    SWIN_FACTORY_MODEL_ID,
    SwinFeatureAdapter,
    SwinFRCNNFactory,
)
from src.models.swin_frcnn.factory import configure_swin_frcnn

ROOT = Path(__file__).resolve().parents[1]


class _Tensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.permutation: tuple[int, ...] | None = None

    def permute(self, *order: int) -> "_Tensor":
        result = _Tensor(tuple(self.shape[index] for index in order))
        result.permutation = order
        return result

    def contiguous(self) -> "_Tensor":
        return self


def _config() -> dict:
    return {
        "model": {
            "type": "MaskRCNN",
            "backbone": {"type": "SwinTransformer", "img_size": 224},
            "neck": {"in_channels": [96, 192, 384, 768]},
            "roi_head": {
                "bbox_head": {"num_classes": 80},
                "mask_roi_extractor": {},
                "mask_head": {"num_classes": 80},
            },
        }
    }


@pytest.mark.parametrize("image_size", [128, 640])
def test_dynamic_size_and_every_fpn_input_contract(image_size: int) -> None:
    config = configure_swin_frcnn(copy.deepcopy(_config()), image_size=image_size, num_classes=2)
    assert config["model"]["backbone"]["img_size"] == image_size
    assert config["model"]["type"] == "FasterRCNN"
    assert config["model"]["roi_head"]["bbox_head"]["num_classes"] == 2
    assert "mask_head" not in config["model"]["roi_head"]
    features = [
        _Tensor((2, image_size // scale, image_size // scale, channels))
        for scale, channels in zip((4, 8, 16, 32), (96, 192, 384, 768))
    ]
    normalized = SwinFeatureAdapter().adapt(features)
    assert [item.shape for item in normalized] == [
        (2, channels, image_size // scale, image_size // scale)
        for scale, channels in zip((4, 8, 16, 32), (96, 192, 384, 768))
    ]
    assert all(item.permutation == (0, 3, 1, 2) for item in normalized)


def test_adapter_accepts_nchw_and_rejects_invalid_features() -> None:
    adapter = SwinFeatureAdapter((4, 8))
    values = adapter.adapt([_Tensor((1, 4, 8, 8)), _Tensor((1, 8, 4, 4))])
    assert [item.shape for item in values] == [(1, 4, 8, 8), (1, 8, 4, 4)]
    with pytest.raises(ValueError, match="channel dimension"):
        adapter.adapt([_Tensor((1, 3, 8, 8)), _Tensor((1, 8, 4, 4))])


def test_factory_passes_configured_object_to_public_builder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mmdet = tmp_path / "mmdet"
    path = mmdet / "configs" / "swin" / "mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py"
    path.parent.mkdir(parents=True)
    path.write_text("# fixture\n")
    monkeypatch.setenv("MMDET_ROOT", str(mmdet))
    captured: dict[str, object] = {}

    def builder(config: dict, checkpoint: str | None, device: str) -> dict:
        captured.update(config=config, checkpoint=checkpoint, device=device)
        return captured

    model = SwinFRCNNFactory(ROOT).build(
        image_size=640,
        num_classes=2,
        config_loader=lambda _: copy.deepcopy(_config()),
        model_builder=builder,
    )
    assert model["config"]["model"]["backbone"]["img_size"] == 640
    assert model["device"] == "cpu"
    assert SWIN_FACTORY_MODEL_ID == "faster_rcnn_swin_t"
