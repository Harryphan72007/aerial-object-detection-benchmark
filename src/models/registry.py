"""Model adapter registry and configuration resolution."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any
from src.models.base_adapter import DetectionModelAdapter
from src.models.mmdetection_adapter import MMDetectionAdapter
from src.models.rtdetr_adapter import RTDetrV2Adapter
from src.models.yolox_adapter import YOLOXAdapter
from src.utils.serialization import read_yaml

MODEL_CONFIGS={
 "faster_rcnn_resnet50":"configs/faster_rcnn_resnet50/model.yaml",
 "faster_rcnn_swin_t":"configs/faster_rcnn_swin_t/model.yaml",
 "faster_rcnn_vmamba_t":"configs/faster_rcnn_vmamba_t/model.yaml",
 "rtdetrv2_l":"configs/rtdetrv2_l/model.yaml",
 "yolox_s":"configs/yolox_s/model.yaml",
}

def load_model_config(model_id:str,repo_root:str|Path=".")->dict[str,Any]:
    if model_id not in MODEL_CONFIGS: raise KeyError(f"unknown model_id {model_id!r}; choose {sorted(MODEL_CONFIGS)}")
    cfg=read_yaml(Path(repo_root)/MODEL_CONFIGS[model_id])
    env_name=cfg.get("external_root_env")
    if env_name:
        external=os.environ.get(str(env_name))
        if external:
            cfg["external_root"]=external
            cfg["resolved_framework_config"]=str(Path(external)/str(cfg["framework_config"]))
    return cfg

def create_adapter(model_id:str,device:str|None=None)->DetectionModelAdapter:
    if model_id.startswith("faster_rcnn_"): return MMDetectionAdapter(model_id,device or "cuda:0")
    if model_id=="rtdetrv2_l": return RTDetrV2Adapter(model_id,device)
    if model_id=="yolox_s": return YOLOXAdapter(model_id,device or "cuda")
    raise KeyError(model_id)
