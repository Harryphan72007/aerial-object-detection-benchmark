"""RT-DETRv2 construction and legacy evaluation-weight loading only."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from src.utils.serialization import read_yaml

RTDETR_FACTORY_MODEL_ID = "rtdetrv2_l"


@dataclass(frozen=True)
class RTDetrBuildResult:
    model: Any
    processor: Any
    device: str
    report: dict[str, Any]


class RTDetrV2Factory:
    """Build the frozen Transformers model without touching optimizer policy."""

    def __init__(
        self,
        repo_root: str | Path = ".",
        *,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.config = (
            dict(config)
            if config is not None
            else read_yaml(
                self.repo_root / "configs" / RTDETR_FACTORY_MODEL_ID / "model.yaml"
            )
        )
        if self.config.get("framework") not in {None, "transformers"}:
            raise ValueError("RT-DETRv2 factory requires the Transformers recipe")

    def build(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        device: str | None = None,
        torch_module: Any | None = None,
        processor_class: Any | None = None,
        model_class: Any | None = None,
        state_loader: Callable[[Path], Mapping[str, Any]] | None = None,
    ) -> RTDetrBuildResult:
        if torch_module is None or processor_class is None or model_class is None:
            try:
                import torch
                from transformers import (
                    RTDetrImageProcessor,
                    RTDetrV2ForObjectDetection,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "PyTorch and Transformers with RT-DETRv2 support are required"
                ) from exc
            torch_module = torch if torch_module is None else torch_module
            processor_class = (
                RTDetrImageProcessor if processor_class is None else processor_class
            )
            model_class = (
                RTDetrV2ForObjectDetection if model_class is None else model_class
            )
        base_source = str(
            self.config.get(
                "pretrained_model_name_or_path", "PekingU/rtdetr_v2_r50vd"
            )
        )
        revision = self.config.get("pretrained_revision")
        common_kwargs = {"revision": str(revision)} if revision else {}
        processor_kwargs = dict(common_kwargs)
        resolution = self.config.get("input_resolution")
        if resolution:
            processor_kwargs["size"] = {
                "height": int(resolution),
                "width": int(resolution),
            }
        checkpoint = Path(checkpoint_path) if checkpoint_path else None
        local_checkpoint = checkpoint is not None and checkpoint.is_file()
        if local_checkpoint:
            if state_loader is None:
                state_loader = lambda path: torch_module.load(
                    path, map_location="cpu", weights_only=False
                )
            state = dict(state_loader(checkpoint))
            raw_id2label = state.get("id2label") or self.config.get("id2label")
            if not raw_id2label:
                raise ValueError("RT-DETR checkpoint is missing id2label metadata")
            id2label = {int(key): str(value) for key, value in raw_id2label.items()}
            label2id = {value: key for key, value in id2label.items()}
            processor = processor_class.from_pretrained(
                base_source, **processor_kwargs
            )
            model = model_class.from_pretrained(
                base_source,
                **common_kwargs,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )
            incompatibility = model.load_state_dict(
                state.get("model", state), strict=True
            )
            missing = list(getattr(incompatibility, "missing_keys", []))
            unexpected = list(getattr(incompatibility, "unexpected_keys", []))
            if missing or unexpected:
                raise RuntimeError(
                    f"RT-DETR checkpoint state is incompatible: missing={missing}, "
                    f"unexpected={unexpected}"
                )
        else:
            source = str(checkpoint_path or base_source)
            processor = processor_class.from_pretrained(source, **processor_kwargs)
            model = model_class.from_pretrained(source, **common_kwargs)
            id2label = dict(getattr(model.config, "id2label", {}))
        resolved_device = device or (
            "cuda" if torch_module.cuda.is_available() else "cpu"
        )
        model.to(resolved_device).eval()
        return RTDetrBuildResult(
            model=model,
            processor=processor,
            device=resolved_device,
            report={
                "model_id": RTDETR_FACTORY_MODEL_ID,
                "source": base_source,
                "revision": revision,
                "legacy_checkpoint_loaded": local_checkpoint,
                "checkpoint": str(checkpoint.resolve()) if local_checkpoint else None,
                "id2label": id2label,
            },
        )

    @staticmethod
    def forward(
        model: Any,
        inputs: Mapping[str, Any],
        *,
        inference_context: Callable[[], Any] | None = None,
    ) -> Any:
        context = inference_context or nullcontext
        with context():
            return model(**dict(inputs))
