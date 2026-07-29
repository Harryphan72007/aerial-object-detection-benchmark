"""RT-DETRv2 adapter based on the documented Transformers integration."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from src.models.base_adapter import DetectionModelAdapter


class RTDetrV2Adapter(DetectionModelAdapter):
    """Load Hugging Face RT-DETRv2 models or this project's ``.pth`` runs."""

    def __init__(self, model_id: str = "rtdetrv2_l", device: str | None = None):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self.config: dict[str, Any] = {}

    def load_model(
        self, checkpoint_path: str | Path, config: dict[str, Any]
    ) -> Any:
        try:
            import torch
            from transformers import (
                RTDetrImageProcessor,
                RTDetrV2ForObjectDetection,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Install transformers with RT-DETRv2 support."
            ) from exc

        checkpoint = Path(checkpoint_path)
        base_source = str(
            config.get("pretrained_model_name_or_path", "PekingU/rtdetr_v2_r101vd")
        )
        processor_kwargs: dict[str, Any] = {}
        resolution = config.get("input_resolution")
        if resolution:
            processor_kwargs["size"] = {
                "height": int(resolution),
                "width": int(resolution),
            }

        if checkpoint.exists() and checkpoint.is_file():
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            raw_id2label = state.get("id2label") or config.get("id2label")
            if not raw_id2label:
                raise ValueError(
                    "RT-DETR checkpoint is missing id2label metadata; "
                    "load a standardized project checkpoint or provide id2label."
                )
            id2label = {int(key): str(value) for key, value in raw_id2label.items()}
            label2id = {value: key for key, value in id2label.items()}
            self.processor = RTDetrImageProcessor.from_pretrained(
                base_source, **processor_kwargs
            )
            self.model = RTDetrV2ForObjectDetection.from_pretrained(
                base_source,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )
            model_state = state.get("model", state)
            incompat = self.model.load_state_dict(model_state, strict=True)
            if incompat.missing_keys or incompat.unexpected_keys:
                raise RuntimeError(
                    "RT-DETR checkpoint state is incompatible: "
                    f"missing={incompat.missing_keys}, "
                    f"unexpected={incompat.unexpected_keys}"
                )
        else:
            source = str(checkpoint_path or base_source)
            self.processor = RTDetrImageProcessor.from_pretrained(
                source, **processor_kwargs
            )
            self.model = RTDetrV2ForObjectDetection.from_pretrained(source)

        self.device = self.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device).eval()
        self.config = dict(config)
        return self.model

    def preprocess(self, images: Sequence[Any]) -> Any:
        if self.processor is None:
            raise RuntimeError("load_model must be called first")
        pil_images: list[Image.Image] = []
        for image in images:
            if isinstance(image, Image.Image):
                pil_images.append(image.convert("RGB"))
            elif isinstance(image, (str, Path)):
                pil_images.append(Image.open(image).convert("RGB"))
            elif isinstance(image, np.ndarray):
                pil_images.append(Image.fromarray(image).convert("RGB"))
            else:
                raise TypeError(f"unsupported image type: {type(image)!r}")
        return self.processor(images=pil_images, return_tensors="pt"), pil_images

    def predict(self, images: Sequence[Any]) -> list[dict[str, Any]]:
        import torch

        if self.model is None or self.processor is None:
            raise RuntimeError("load_model must be called first")
        inputs, pil_images = self.preprocess(images)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        sizes = torch.tensor(
            [(image.height, image.width) for image in pil_images],
            device=self.device,
        )
        threshold = float(self.config.get("confidence_threshold", 0.001))
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=sizes, threshold=threshold
        )
        maximum = int(self.config.get("max_detections", 500))
        normalized = []
        for result in results:
            if len(result["scores"]) > maximum:
                keep = result["scores"].argsort(descending=True)[:maximum]
                result = {key: value[keep] for key, value in result.items()}
            normalized.append(
                {
                    "boxes": result["boxes"].detach().cpu().tolist(),
                    "scores": result["scores"].detach().cpu().tolist(),
                    # HF labels are zero-based; project COCO category IDs are one-based.
                    "labels": (result["labels"].detach().cpu() + 1).tolist(),
                }
            )
        return normalized

    def postprocess(
        self, outputs: Any, original_sizes: Sequence[tuple[int, int]]
    ) -> list[dict[str, Any]]:
        return outputs

    def profile(
        self,
        sample_batch: Any,
        warmup: int = 100,
        iterations: int = 500,
        **_: Any,
    ) -> dict[str, Any]:
        import torch

        images = sample_batch if isinstance(sample_batch, list) else [sample_batch]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for _ in range(warmup):
            self.predict(images)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings: list[float] = []
        for _ in range(iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            self.predict(images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
        values = np.asarray(timings)
        return {
            "mean_latency_ms": float(values.mean()),
            "median_latency_ms": float(np.median(values)),
            "p90_latency_ms": float(np.quantile(values, 0.90)),
            "p95_latency_ms": float(np.quantile(values, 0.95)),
            "p99_latency_ms": float(np.quantile(values, 0.99)),
            "fps": float(1000 / values.mean()),
            "throughput_images_per_second": float(
                len(images) * 1000 / values.mean()
            ),
            "batch_size": len(images),
            "peak_inference_vram_bytes": int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else 0,
        }
