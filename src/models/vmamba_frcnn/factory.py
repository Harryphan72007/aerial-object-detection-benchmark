"""Pinned VMamba-T Faster R-CNN construction boundary."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.models.registry import load_model_config
from src.models.vmamba_frcnn.importer import (
    detect_selective_scan_backend,
    register_vmamba_detection,
    verify_vmamba_revision,
)

VMAMBA_FACTORY_MODEL_ID = "faster_rcnn_vmamba_t"


@dataclass(frozen=True)
class VMambaBuildResult:
    model: Any
    report: dict[str, Any]


def _set_num_classes(node: Any, count: int) -> None:
    if isinstance(node, dict):
        if "num_classes" in node:
            node["num_classes"] = count
        for value in node.values():
            _set_num_classes(value, count)
    elif isinstance(node, list):
        for value in node:
            _set_num_classes(value, count)


class VMambaFRCNNFactory:
    def __init__(self, repo_root: str | Path = ".") -> None:
        self.repo_root = Path(repo_root).resolve()
        self.config = load_model_config(VMAMBA_FACTORY_MODEL_ID, self.repo_root)

    @property
    def vmamba_root(self) -> Path:
        value = self.config.get("external_root")
        if not value:
            raise RuntimeError("VMAMBA_ROOT must point to the pinned VMamba checkout")
        return Path(str(value)).expanduser().resolve()

    @property
    def framework_config(self) -> Path:
        value = self.config.get("resolved_framework_config")
        if not value:
            raise RuntimeError("VMAMBA_ROOT must resolve the VMamba detector config")
        path = Path(str(value)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def build(
        self,
        *,
        num_classes: int,
        device: str = "cpu",
        config_loader: Callable[[str], Any] | None = None,
        model_builder: Callable[[Any, str | None, str], Any] | None = None,
        revision_reader: Callable[[Path], str] | None = None,
        backend_finder: Callable[[str], Any] | None = None,
    ) -> VMambaBuildResult:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        revision = verify_vmamba_revision(
            self.vmamba_root,
            str(self.config["source_revision"]),
            revision_reader=revision_reader,
        )
        registration = register_vmamba_detection(self.vmamba_root)
        backend = (
            detect_selective_scan_backend(backend_finder)
            if backend_finder is not None
            else detect_selective_scan_backend()
        )
        if not backend["approved"]:
            raise RuntimeError("approved optimized VMamba selective scan is unavailable")
        pretrained_env = str(self.config.get("pretrained_weight_env", ""))
        pretrained = Path(os.environ.get(pretrained_env, "")).expanduser()
        if not str(pretrained) or not pretrained.is_file():
            raise RuntimeError(f"{pretrained_env} must point to a verified checkpoint")
        if config_loader is None:
            try:
                from mmengine.config import Config
            except ImportError as exc:
                raise RuntimeError("MMEngine is required to load VMamba") from exc
            config_loader = Config.fromfile
        if model_builder is None:
            try:
                from mmdet.apis import init_detector
            except ImportError as exc:
                raise RuntimeError("MMDetection is required to build VMamba") from exc
            model_builder = init_detector
        config = config_loader(str(self.framework_config))
        model = config["model"]
        model["type"] = "FasterRCNN"
        model.get("roi_head", {}).pop("mask_roi_extractor", None)
        model.get("roi_head", {}).pop("mask_head", None)
        model["backbone"]["pretrained"] = str(pretrained.resolve())
        model["backbone"]["init_cfg"] = None
        _set_num_classes(model, num_classes)
        built = model_builder(config, None, device)
        return VMambaBuildResult(
            model=built,
            report={
                "model_id": VMAMBA_FACTORY_MODEL_ID,
                "revision": revision,
                "registration_module": str(registration.__file__),
                "selective_scan": backend,
                "pretrained": str(pretrained.resolve()),
            },
        )
