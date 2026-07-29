"""Optional official YOLOX adapter."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Sequence
from src.models.base_adapter import DetectionModelAdapter

class YOLOXAdapter(DetectionModelAdapter):
    def __init__(self,model_id:str="yolox_s",device:str="cuda"):
        self.model_id=model_id; self.device=device
    def load_model(self,checkpoint_path:str|Path,config:dict[str,Any])->Any:
        raise RuntimeError("YOLOX is an optional control. Clone the official YOLOX repository and implement the deployment-specific predictor behind this adapter; the benchmark never silently substitutes a different YOLO package.")
    def preprocess(self,images:Sequence[Any])->Any: return images
    def predict(self,images:Sequence[Any])->list[dict[str,Any]]: raise RuntimeError("YOLOX adapter not initialized")
    def postprocess(self,outputs:Any,original_sizes:Sequence[tuple[int,int]])->list[dict[str,Any]]: return outputs
    def profile(self,sample_batch:Any,**kwargs:Any)->dict[str,Any]: raise RuntimeError("YOLOX adapter not initialized")
