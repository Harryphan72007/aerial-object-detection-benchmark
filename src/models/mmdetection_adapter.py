"""MMDetection 3.x inference adapter using public APIs."""
from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from PIL import Image
from src.models.base_adapter import DetectionModelAdapter

class MMDetectionAdapter(DetectionModelAdapter):
    def __init__(self, model_id: str, device: str = "cuda:0"):
        self.model_id=model_id; self.device=device; self.model=None; self.config={}
    def load_model(self, checkpoint_path: str|Path, config: dict[str,Any])->Any:
        try:
            from mmdet.apis import init_detector
        except ImportError as exc:
            raise RuntimeError("Install MMDetection 3.3.0 and a matching MMCV wheel before loading this model.") from exc
        registration_import=config.get("registration_import")
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
