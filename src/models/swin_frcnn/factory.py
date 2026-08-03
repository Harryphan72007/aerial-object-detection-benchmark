"""Dynamic-resolution Swin-T Faster R-CNN construction boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from src.models.registry import load_model_config

SWIN_FACTORY_MODEL_ID = "faster_rcnn_swin_t"
ConfigLoader = Callable[[str], Any]
ModelBuilder = Callable[[Any, str | None, str], Any]


def _set_num_classes(node: Any, number_of_classes: int) -> None:
    if isinstance(node, dict):
        if "num_classes" in node:
            node["num_classes"] = number_of_classes
        for value in node.values():
            _set_num_classes(value, number_of_classes)
    elif isinstance(node, list):
        for value in node:
            _set_num_classes(value, number_of_classes)


def configure_swin_frcnn(config: Any, *, image_size: int, num_classes: int) -> Any:
    """Apply only construction-time Swin/Faster R-CNN compatibility changes."""

    if image_size <= 0 or num_classes <= 0:
        raise ValueError("image_size and num_classes must be positive")
    model = config["model"]
    model["type"] = "FasterRCNN"
    # Detection resolution is controlled by the MMDetection input pipeline.
    # The pinned SwinTransformer constructor accepts ``pretrain_img_size`` but
    # not ``img_size``; injecting the latter prevents real registry construction.
    model.get("backbone", {}).pop("img_size", None)
    roi_head = model.get("roi_head", {})
    roi_head.pop("mask_roi_extractor", None)
    roi_head.pop("mask_head", None)
    _set_num_classes(model, num_classes)
    return config


class SwinFRCNNFactory:
    def __init__(self, repo_root: str | Path = ".") -> None:
        self.repo_root = Path(repo_root).resolve()
        self.config = load_model_config(SWIN_FACTORY_MODEL_ID, self.repo_root)

    @property
    def framework_config(self) -> Path:
        value = self.config.get("resolved_framework_config")
        if not value:
            raise RuntimeError("MMDET_ROOT must point to the pinned MMDetection checkout")
        path = Path(str(value)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def build(
        self,
        *,
        image_size: int,
        num_classes: int,
        checkpoint: str | Path | None = None,
        device: str = "cpu",
        config_loader: ConfigLoader | None = None,
        model_builder: ModelBuilder | None = None,
    ) -> Any:
        if config_loader is None:
            try:
                from mmengine.config import Config
            except ImportError as exc:
                raise RuntimeError("MMEngine is required to load the Swin config") from exc
            config_loader = Config.fromfile
        if model_builder is None:
            try:
                from mmdet.apis import init_detector
            except ImportError as exc:
                raise RuntimeError("MMDetection is required to build Swin Faster R-CNN") from exc
            model_builder = init_detector
        config = config_loader(str(self.framework_config))
        configure_swin_frcnn(config, image_size=image_size, num_classes=num_classes)
        checkpoint_value = str(Path(checkpoint).resolve()) if checkpoint else None
        return model_builder(config, checkpoint_value, device)
