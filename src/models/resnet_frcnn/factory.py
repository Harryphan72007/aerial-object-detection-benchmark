"""Construction boundary for the MMDetection ResNet-50 Faster R-CNN."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

from src.models.registry import load_model_config

RESNET_FACTORY_MODEL_ID = "faster_rcnn_resnet50"
Initializer = Callable[[str, str | None, str], Any]
InferenceFunction = Callable[[Any, Any], Any]


class ResNetFRCNNFactory:
    """Resolve and construct the frozen legacy ResNet detector recipe."""

    def __init__(self, repo_root: str | Path = ".") -> None:
        self.repo_root = Path(repo_root).resolve()
        self.config = load_model_config(RESNET_FACTORY_MODEL_ID, self.repo_root)
        if self.config.get("framework") != "mmdetection":
            raise ValueError("ResNet factory requires the MMDetection recipe")

    @property
    def framework_config(self) -> Path:
        value = self.config.get("resolved_framework_config")
        if not value:
            environment = self.config.get("external_root_env", "MMDET_ROOT")
            raise RuntimeError(
                f"{environment} must point to the pinned MMDetection checkout"
            )
        path = Path(str(value)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def build(
        self,
        *,
        checkpoint: str | Path | None = None,
        device: str = "cpu",
        initializer: Initializer | None = None,
    ) -> Any:
        """Construct through the same public API used by the legacy adapter."""

        if initializer is None:
            try:
                from mmdet.apis import init_detector
            except ImportError as exc:
                raise RuntimeError(
                    "MMDetection 3.3.0 and a matching MMCV wheel are required"
                ) from exc
            initializer = init_detector
        checkpoint_value = (
            str(Path(checkpoint).expanduser().resolve()) if checkpoint else None
        )
        return initializer(str(self.framework_config), checkpoint_value, device)

    @staticmethod
    def parameter_count(model: Any) -> int:
        parameters: Iterable[Any] = model.parameters()
        return sum(int(parameter.numel()) for parameter in parameters)

    @staticmethod
    def forward(
        model: Any,
        image: Any,
        *,
        inference: InferenceFunction | None = None,
    ) -> Any:
        if inference is None:
            try:
                from mmdet.apis import inference_detector
            except ImportError as exc:
                raise RuntimeError("MMDetection inference API is unavailable") from exc
            inference = inference_detector
        return inference(model, image)


def build_resnet_frcnn(
    repo_root: str | Path = ".",
    *,
    checkpoint: str | Path | None = None,
    device: str = "cpu",
    initializer: Initializer | None = None,
) -> Any:
    """Functional entry point used by notebooks and later trainer migration."""

    return ResNetFRCNNFactory(repo_root).build(
        checkpoint=checkpoint,
        device=device,
        initializer=initializer,
    )
