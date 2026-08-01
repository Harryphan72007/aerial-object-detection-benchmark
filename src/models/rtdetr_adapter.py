"""RT-DETRv2 adapter based on the documented Transformers integration."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from src.models.base_adapter import DetectionModelAdapter
from src.models.rtdetrv2.factory import RTDetrV2Factory


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
        result = RTDetrV2Factory(config=config).build(
            checkpoint_path, device=self.device
        )
        self.model = result.model
        self.processor = result.processor
        self.device = result.device
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
