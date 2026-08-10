"""RT-DETRv2 adapter based on the documented Transformers integration."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from src.models.adapter_validation import (
    CHECKPOINT_INCOMPLETE,
    CHECKPOINT_LOADED,
    AdapterValidationContract,
    CheckpointLoadResult,
    FeatureMapSpec,
    ForwardBackwardResult,
    parameter_digests,
)
from src.models.base_adapter import DetectionModelAdapter
from src.models.rtdetrv2.factory import RTDetrV2Factory


class RTDetrV2Adapter(DetectionModelAdapter, AdapterValidationContract):
    """Load Hugging Face RT-DETRv2 models or this project's ``.pth`` runs."""

    def __init__(
        self,
        model_id: str = "rtdetrv2_l",
        device: str | None = None,
        repo_root: str | Path = ".",
    ):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self.config: dict[str, Any] = {}
        self.repo_root = Path(repo_root).resolve()
        self._built: dict[str, Any] = {}

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

    # --- GPU validation contract ---------------------------------------------

    def _model_configuration(self) -> dict[str, Any]:
        from src.models.registry import load_model_config

        if not self.config:
            self.config = load_model_config(self.model_id, self.repo_root)
        return self.config

    def _pretrained_identity(self) -> tuple[str, str]:
        config = self._model_configuration()
        return (
            str(config["pretrained_model_name_or_path"]),
            str(config["pretrained_revision"]),
        )

    def build_for_validation(
        self, *, num_classes: int, image_size: int, device: str
    ) -> Any:
        """Construct exactly what ``scripts/run_rtdetr_training`` constructs."""
        from transformers import (
            RTDetrImageProcessor,
            RTDetrV2Config,
            RTDetrV2ForObjectDetection,
        )

        name, revision = self._pretrained_identity()
        id2label = {index: f"class_{index}" for index in range(num_classes)}
        model_configuration = RTDetrV2Config.from_pretrained(name, revision=revision)
        model_configuration.id2label = id2label
        model_configuration.label2id = {
            value: key for key, value in id2label.items()
        }
        model, loading_info = RTDetrV2ForObjectDetection.from_pretrained(
            name,
            revision=revision,
            config=model_configuration,
            ignore_mismatched_sizes=True,
            output_loading_info=True,
        )
        model.to(device)
        self.model = model
        self.processor = RTDetrImageProcessor.from_pretrained(
            name,
            revision=revision,
            size={"height": image_size, "width": image_size},
        )
        self.device = device
        self._built = {
            "model": model,
            "configuration": model_configuration,
            "loading_info": dict(loading_info),
            "device": device,
            "num_classes": int(num_classes),
            "image_size": int(image_size),
        }
        return model

    def _require_built(self) -> dict[str, Any]:
        if not self._built:
            raise RuntimeError("build_for_validation must be called first")
        return self._built

    def checkpoint_load_result(self) -> CheckpointLoadResult:
        built = self._require_built()
        info = built["loading_info"]
        name, revision = self._pretrained_identity()
        missing = tuple(str(key) for key in info.get("missing_keys", []))
        unexpected = tuple(str(key) for key in info.get("unexpected_keys", []))
        # ``ignore_mismatched_sizes=True`` intentionally re-initialises the
        # classification head for the benchmark's class count; those keys are
        # reported separately by Transformers and are not a partial load.
        mismatched = [str(key) for key in info.get("mismatched_keys", [])]
        complete = not missing and not unexpected
        return CheckpointLoadResult(
            state=CHECKPOINT_LOADED if complete else CHECKPOINT_INCOMPLETE,
            source=f"{name}@{revision}",
            missing_keys=missing,
            unexpected_keys=unexpected,
            detail=(
                f"{len(mismatched)} class-head tensors resized for "
                f"{built['num_classes']} classes: {mismatched[:4]}"
            ),
        )

    def feature_maps(self, image_size: int) -> list[FeatureMapSpec]:
        """Capture the hybrid encoder's projected multi-scale inputs.

        RT-DETRv2 has no FPN; its architecture-equivalent feature levels are the
        ``encoder_input_proj`` outputs consumed by the hybrid encoder.
        """
        import torch

        built = self._require_built()
        model = built["model"]
        projections = getattr(model.model, "encoder_input_proj", None)
        if projections is None:
            raise RuntimeError(
                "RTDetrV2Model.encoder_input_proj is unavailable; the pinned "
                "Transformers revision no longer exposes the encoder feature levels"
            )
        captured: list[Any] = []
        handles = [
            projection.register_forward_hook(
                lambda _module, _inputs, output: captured.append(output.detach())
            )
            for projection in projections
        ]
        model.eval()
        try:
            with torch.no_grad():
                model(
                    pixel_values=torch.zeros(
                        (1, 3, image_size, image_size), device=built["device"]
                    )
                )
        finally:
            for handle in handles:
                handle.remove()
            model.eval()
        specs: list[FeatureMapSpec] = []
        for level, feature in enumerate(captured):
            shape = tuple(int(value) for value in feature.shape)
            if len(shape) != 4 or shape[2] == 0:
                raise RuntimeError(f"encoder level {level} is not NCHW: {shape}")
            specs.append(
                FeatureMapSpec(shape=shape, stride=int(image_size // shape[2]))
            )
        return specs

    def head_class_count(self) -> int:
        model = self._require_built()["model"]
        heads = getattr(model, "class_embed", None)
        if heads is None:
            raise RuntimeError(
                "RTDetrV2ForObjectDetection.class_embed is unavailable to inspect"
            )
        counts = {int(head.out_features) for head in heads}
        if len(counts) != 1:
            raise RuntimeError(f"decoder heads disagree on class count: {sorted(counts)}")
        return counts.pop()

    def forward_backward(
        self, *, image_size: int, batch_size: int = 2
    ) -> ForwardBackwardResult:
        import torch

        built = self._require_built()
        model = built["model"]
        num_classes = built["num_classes"]
        model.train()
        model.zero_grad(set_to_none=True)
        # Normalised cxcywh boxes, as the Transformers detection loss expects.
        labels = [
            {
                "class_labels": torch.tensor(
                    [0, min(1, num_classes - 1)], dtype=torch.long, device=built["device"]
                ),
                "boxes": torch.tensor(
                    [[0.25, 0.25, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2]],
                    dtype=torch.float32,
                    device=built["device"],
                ),
            }
            for _ in range(batch_size)
        ]
        outputs = model(
            pixel_values=torch.rand(
                (batch_size, 3, image_size, image_size), device=built["device"]
            ),
            labels=labels,
        )
        loss = outputs.loss
        if loss is None:
            raise RuntimeError("RT-DETRv2 returned no loss for a labelled batch")
        loss.backward()
        trainable = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        with_gradient = [
            parameter for parameter in trainable if parameter.grad is not None
        ]
        maximum = max(
            (float(parameter.grad.detach().abs().max()) for parameter in with_gradient),
            default=0.0,
        )
        result = ForwardBackwardResult(
            loss=float(loss.detach()),
            trainable_parameters=len(trainable),
            parameters_with_gradient=len(with_gradient),
            maximum_absolute_gradient=maximum,
            batch_size=batch_size,
            image_size=image_size,
            extra={"num_queries": int(model.config.num_queries)},
        )
        model.zero_grad(set_to_none=True)
        model.eval()
        return result

    def checkpoint_roundtrip(self) -> tuple[dict[str, str], dict[str, str]]:
        import tempfile

        import torch
        from transformers import RTDetrV2ForObjectDetection

        built = self._require_built()
        model = built["model"]
        before = parameter_digests(model)
        with tempfile.TemporaryDirectory(prefix="adapter_smoke_") as directory:
            path = Path(directory) / "roundtrip.pth"
            torch.save(model.state_dict(), path)
            reloaded = RTDetrV2ForObjectDetection(built["configuration"]).to(
                built["device"]
            )
            reloaded.load_state_dict(
                torch.load(path, map_location=built["device"], weights_only=True),
                strict=True,
            )
            after = parameter_digests(reloaded)
            del reloaded
        return before, after

    def release(self) -> None:
        import torch

        self._built.clear()
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
