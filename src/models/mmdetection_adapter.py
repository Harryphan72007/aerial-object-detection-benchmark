"""MMDetection 3.x inference adapter using public APIs."""
from __future__ import annotations
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from PIL import Image
from src.models.adapter_validation import (
    CHECKPOINT_INCOMPLETE,
    CHECKPOINT_LOADED,
    CHECKPOINT_MISSING,
    CHECKPOINT_NOT_APPLICABLE,
    MINIMUM_CHECKPOINT_VALUE_COVERAGE,
    AdapterValidationContract,
    CheckpointLoadResult,
    FeatureMapSpec,
    ForwardBackwardResult,
    checkpoint_value_coverage,
    file_sha256,
    filtered_keys,
    parameter_digests,
    unwrap_checkpoint_state,
)
from src.models.base_adapter import DetectionModelAdapter

# Keys whose absence from (or excess in) the pretrained checkpoint is a property
# of the architecture, not a broken load. VMamba's ``Backbone_VSSM`` deletes the
# classifier and adds per-stage ``outnorm`` layers, so an ImageNet classification
# checkpoint necessarily differs by exactly those names.
ALLOWED_MISSING_BACKBONE_KEYS = {"faster_rcnn_vmamba_t": ("outnorm",)}
ALLOWED_UNEXPECTED_BACKBONE_KEYS = {"faster_rcnn_vmamba_t": ("classifier.",)}
# Families whose upstream loader calls ``load_state_dict`` on the raw checkpoint
# without renaming, so replaying that call reproduces the loader exactly. The
# MMDetection backbones convert key names inside ``init_weights`` and are checked
# by tensor value instead.
REPLAYABLE_BACKBONE_LOAD = frozenset({"faster_rcnn_vmamba_t"})

class MMDetectionAdapter(DetectionModelAdapter, AdapterValidationContract):
    def __init__(self, model_id: str, device: str = "cuda:0", repo_root: str | Path = "."):
        self.model_id=model_id; self.device=device; self.model=None; self.config={}
        self.repo_root=Path(repo_root).resolve(); self._built: dict[str,Any]={}
    def load_model(self, checkpoint_path: str|Path, config: dict[str,Any])->Any:
        try:
            from mmdet.apis import init_detector
        except ImportError as exc:
            raise RuntimeError("Install MMDetection 3.3.0 and a matching MMCV wheel before loading this model.") from exc
        self._register_external_sources(config)
        cfg_path=config.get("resolved_framework_config") or config.get("framework_config")
        if not cfg_path: raise ValueError("resolved_framework_config is required")
        self.config=config; self.model=init_detector(str(cfg_path),str(checkpoint_path),device=self.device)
        return self.model
    def preprocess(self, images: Sequence[Any])->list[Any]:
        result=[]
        for image in images:
            if isinstance(image,(str,Path,np.ndarray)): result.append(image)
            elif isinstance(image,Image.Image): result.append(np.asarray(image.convert("RGB"))[:,:,::-1])
            else: raise TypeError(f"unsupported image type: {type(image)!r}")
        return result
    def predict(self, images: Sequence[Any])->list[dict[str,Any]]:
        if self.model is None: raise RuntimeError("load_model must be called first")
        from mmdet.apis import inference_detector
        prepared=self.preprocess(images); outputs=inference_detector(self.model,prepared)
        if not isinstance(outputs,list): outputs=[outputs]
        predictions=[]
        for output in outputs:
            instances=output.pred_instances.to("cpu")
            labels=instances.labels.numpy().astype(int)+1
            predictions.append({"boxes":instances.bboxes.numpy().tolist(),"scores":instances.scores.numpy().tolist(),"labels":labels.tolist()})
        return predictions
    def postprocess(self,outputs:Any,original_sizes:Sequence[tuple[int,int]])->list[dict[str,Any]]:
        return outputs
    def profile(self,sample_batch:Any,warmup:int=100,iterations:int=500,**_:Any)->dict[str,Any]:
        if self.model is None: raise RuntimeError("load_model must be called first")
        import torch
        images=sample_batch if isinstance(sample_batch,list) else [sample_batch]
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        for _ in range(warmup): self.predict(images)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        times=[]
        for _ in range(iterations):
            if torch.cuda.is_available(): torch.cuda.synchronize()
            start=time.perf_counter(); self.predict(images)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000)
        arr=np.asarray(times)
        return {"mean_latency_ms":float(arr.mean()),"median_latency_ms":float(np.median(arr)),"p90_latency_ms":float(np.quantile(arr,.90)),"p95_latency_ms":float(np.quantile(arr,.95)),"p99_latency_ms":float(np.quantile(arr,.99)),"fps":float(1000/arr.mean()),"peak_inference_vram_bytes":int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0}

    # --- GPU validation contract ---------------------------------------------

    def _register_external_sources(self, config: dict[str, Any]) -> None:
        """Import the pinned VMamba registration module before config parsing."""
        registration_import = config.get("registration_import")
        if self.model_id == "faster_rcnn_vmamba_t" or registration_import == "model":
            from src.models.vmamba_frcnn.importer import register_vmamba_detection

            vmamba_root = os.environ.get("VMAMBA_ROOT") or config.get("external_root")
            if not vmamba_root:
                raise RuntimeError(
                    "VMAMBA_ROOT must point to the pinned VMamba checkout before evaluation"
                )
            register_vmamba_detection(Path(str(vmamba_root)))
        elif registration_import:
            raise RuntimeError(
                f"unsupported MMDetection registration import: {registration_import!r}"
            )

    def _model_configuration(self) -> dict[str, Any]:
        from src.models.registry import load_model_config

        if not self.config:
            self.config = load_model_config(self.model_id, self.repo_root)
        return self.config

    def _runtime_config(self, *, num_classes: int, load_pretrained: bool) -> Any:
        """Build the same runtime config the training backend builds."""
        from mmengine.config import Config

        from scripts.run_mmdetection import (
            apply_detection_policy,
            configure_faster_rcnn_from_mask,
            set_num_classes,
        )

        config = self._model_configuration()
        self._register_external_sources(config)
        base = config.get("resolved_framework_config") or config.get("framework_config")
        if not base:
            raise RuntimeError(
                f"{config.get('external_root_env')} must resolve the pinned framework config"
            )
        base_path = Path(str(base)).expanduser().resolve()
        if not base_path.is_file():
            raise FileNotFoundError(base_path)
        cfg = Config.fromfile(str(base_path))
        weight_env = config.get("pretrained_weight_env")
        pretrained_weight = os.environ.get(str(weight_env)) if weight_env else None
        if weight_env and load_pretrained:
            if not pretrained_weight:
                raise RuntimeError(
                    f"{weight_env} must point to the verified pretrained checkpoint"
                )
            cfg.model.backbone.pretrained = str(
                Path(pretrained_weight).expanduser().resolve()
            )
            if "init_cfg" in cfg.model.backbone:
                cfg.model.backbone.init_cfg = None
        elif not load_pretrained:
            if "pretrained" in cfg.model.backbone:
                cfg.model.backbone.pretrained = None
            if "init_cfg" in cfg.model.backbone:
                cfg.model.backbone.init_cfg = None
        if config.get("strip_mask_branch"):
            configure_faster_rcnn_from_mask(cfg)
        apply_detection_policy(cfg.model)
        set_num_classes(cfg.model, num_classes)
        return cfg

    def _construct(self, *, num_classes: int, device: str, load_pretrained: bool) -> Any:
        from mmdet.registry import MODELS
        from mmdet.utils import register_all_modules

        register_all_modules(init_default_scope=True)
        cfg = self._runtime_config(num_classes=num_classes, load_pretrained=load_pretrained)
        model = MODELS.build(cfg.model)
        if load_pretrained:
            # The real runtime path: this is where MMEngine (or VMamba's own
            # loader) reads the pretrained backbone weights.
            model.init_weights()
        # ``inference_detector`` reads both attributes; ``init_detector`` sets
        # them and a registry build does not.
        model.cfg = cfg
        model.dataset_meta = {
            "classes": tuple(f"class_{index}" for index in range(num_classes))
        }
        return model.to(device), cfg

    def build_for_validation(
        self, *, num_classes: int, image_size: int, device: str
    ) -> Any:
        model, cfg = self._construct(
            num_classes=num_classes, device=device, load_pretrained=True
        )
        self.model = model
        self.device = device
        self._built = {
            "model": model,
            "config": cfg,
            "device": device,
            "num_classes": int(num_classes),
            "image_size": int(image_size),
        }
        return model

    def _require_built(self) -> dict[str, Any]:
        if not self._built:
            raise RuntimeError("build_for_validation must be called first")
        return self._built

    @staticmethod
    def _pretrained_source(cfg: Any) -> str | None:
        backbone = cfg.model.backbone
        value = backbone.get("pretrained")
        if value:
            return str(value)
        init_cfg = backbone.get("init_cfg")
        if isinstance(init_cfg, dict) and init_cfg.get("type") == "Pretrained":
            return str(init_cfg["checkpoint"])
        return None

    def checkpoint_load_result(self) -> CheckpointLoadResult:
        from mmengine.runner import CheckpointLoader

        built = self._require_built()
        backbone = built["model"].backbone
        source = self._pretrained_source(built["config"])
        if source is None:
            return CheckpointLoadResult(
                state=CHECKPOINT_NOT_APPLICABLE,
                detail=(
                    f"{self.model_id} declares no pretrained backbone checkpoint in "
                    "its pinned framework config"
                ),
            )
        local = None if "://" in source else Path(source).expanduser()
        if local is not None and not local.is_file():
            return CheckpointLoadResult(
                state=CHECKPOINT_MISSING,
                source=str(local),
                detail=f"required pretrained checkpoint does not exist: {local}",
            )
        payload = CheckpointLoader.load_checkpoint(source, map_location="cpu")
        state = unwrap_checkpoint_state(payload)
        coverage = checkpoint_value_coverage(state, backbone)
        if self.model_id not in MINIMUM_CHECKPOINT_VALUE_COVERAGE:
            raise RuntimeError(
                f"{self.model_id} declares no minimum checkpoint value coverage in "
                "src/models/adapter_validation.py; add one before gating it"
            )
        minimum = MINIMUM_CHECKPOINT_VALUE_COVERAGE[self.model_id]
        missing: tuple[str, ...] = ()
        unexpected: tuple[str, ...] = ()
        if self.model_id in REPLAYABLE_BACKBONE_LOAD:
            incompatible = backbone.load_state_dict(state, strict=False)
            missing = tuple(
                filtered_keys(
                    list(incompatible.missing_keys),
                    ALLOWED_MISSING_BACKBONE_KEYS.get(self.model_id, ()),
                )
            )
            unexpected = tuple(
                filtered_keys(
                    list(incompatible.unexpected_keys),
                    ALLOWED_UNEXPECTED_BACKBONE_KEYS.get(self.model_id, ()),
                )
            )
        complete = not missing and not unexpected and coverage >= minimum
        return CheckpointLoadResult(
            state=CHECKPOINT_LOADED if complete else CHECKPOINT_INCOMPLETE,
            source=str(local) if local is not None else source,
            sha256=file_sha256(local) if local is not None else None,
            missing_keys=missing,
            unexpected_keys=unexpected,
            value_coverage=round(coverage, 6),
            minimum_value_coverage=minimum,
            detail=(
                f"{len(state)} checkpoint tensors compared against the constructed "
                f"{type(backbone).__name__} backbone"
            ),
        )

    def feature_maps(self, image_size: int) -> list[FeatureMapSpec]:
        import torch

        built = self._require_built()
        model = built["model"]
        model.eval()
        try:
            with torch.no_grad():
                features = model.extract_feat(
                    torch.zeros((1, 3, image_size, image_size), device=built["device"])
                )
        finally:
            model.eval()
        specs: list[FeatureMapSpec] = []
        for level, feature in enumerate(features):
            shape = tuple(int(value) for value in feature.shape)
            if len(shape) != 4 or shape[2] == 0:
                raise RuntimeError(f"feature level {level} is not NCHW: {shape}")
            specs.append(
                FeatureMapSpec(shape=shape, stride=int(image_size // shape[2]))
            )
        return specs

    def head_class_count(self) -> int:
        model = self._require_built()["model"]
        roi_head = getattr(model, "roi_head", None)
        bbox_head = getattr(roi_head, "bbox_head", None)
        if bbox_head is None:
            raise RuntimeError(
                f"{type(model).__name__} exposes no roi_head.bbox_head to inspect"
            )
        heads = list(bbox_head) if hasattr(bbox_head, "__len__") else [bbox_head]
        counts = {int(head.num_classes) for head in heads}
        if len(counts) != 1:
            raise RuntimeError(f"detection heads disagree on num_classes: {sorted(counts)}")
        return counts.pop()

    def forward_backward(
        self, *, image_size: int, batch_size: int = 2
    ) -> ForwardBackwardResult:
        import torch
        from mmdet.structures import DetDataSample
        from mmengine.structures import InstanceData

        built = self._require_built()
        model = built["model"]
        num_classes = built["num_classes"]
        model.train()
        model.zero_grad(set_to_none=True)
        boxes = torch.tensor(
            [
                [0.1 * image_size, 0.1 * image_size, 0.4 * image_size, 0.4 * image_size],
                [0.5 * image_size, 0.5 * image_size, 0.9 * image_size, 0.9 * image_size],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, min(1, num_classes - 1)], dtype=torch.long)
        samples = []
        for index in range(batch_size):
            sample = DetDataSample()
            sample.set_metainfo(
                {
                    "img_id": index,
                    "img_shape": (image_size, image_size),
                    "ori_shape": (image_size, image_size),
                    "pad_shape": (image_size, image_size),
                    "batch_input_shape": (image_size, image_size),
                    "scale_factor": (1.0, 1.0),
                }
            )
            sample.gt_instances = InstanceData(bboxes=boxes.clone(), labels=labels.clone())
            samples.append(sample)
        batch = model.data_preprocessor(
            {
                "inputs": [
                    torch.randint(
                        0, 255, (3, image_size, image_size), dtype=torch.uint8
                    )
                    for _ in range(batch_size)
                ],
                "data_samples": samples,
            },
            True,
        )
        losses = model(**batch, mode="loss")
        loss, _ = model.parse_losses(losses)
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
            extra={"loss_terms": sorted(losses)},
        )
        model.zero_grad(set_to_none=True)
        model.eval()
        return result

    def checkpoint_roundtrip(self) -> tuple[dict[str, str], dict[str, str]]:
        import torch

        built = self._require_built()
        model = built["model"]
        before = parameter_digests(model)
        # A temporary directory, never an experiment result directory: a smoke
        # pass is not a benchmark artifact.
        with tempfile.TemporaryDirectory(prefix="adapter_smoke_") as directory:
            path = Path(directory) / "roundtrip.pth"
            torch.save(model.state_dict(), path)
            reloaded, _ = self._construct(
                num_classes=built["num_classes"],
                device=built["device"],
                load_pretrained=False,
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
