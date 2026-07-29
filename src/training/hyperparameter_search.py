"""Optuna search-space definitions and resumable study storage."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Callable

def suggest_common(trial:Any)->dict[str,Any]:
    return {"learning_rate":trial.suggest_float("learning_rate",1e-6,5e-4,log=True),"weight_decay":trial.suggest_float("weight_decay",1e-6,0.2,log=True),"warmup_epochs":trial.suggest_int("warmup_epochs",0,10),"backbone_lr_multiplier":trial.suggest_float("backbone_lr_multiplier",0.01,1.0,log=True),"gradient_accumulation_steps":trial.suggest_categorical("gradient_accumulation_steps",[1,2,4,8,16]),"input_resolution":trial.suggest_categorical("input_resolution",[640,1024,1280]),"augmentation_strength":trial.suggest_categorical("augmentation_strength",["standard","strong"]),"gradient_clip":trial.suggest_float("gradient_clip",0.1,10.0,log=True),"max_detections":trial.suggest_categorical("max_detections",[100,300,500])}
def suggest_faster_rcnn(trial:Any)->dict[str,Any]:
    return {"anchor_sizes":trial.suggest_categorical("anchor_sizes",[[4,8,16,32,64],[8,16,32,64,128]]),"anchor_ratios":trial.suggest_categorical("anchor_ratios",[[0.5,1,2],[0.33,0.5,1,2,3]]),"rpn_nms_threshold":trial.suggest_float("rpn_nms_threshold",0.5,0.9),"rpn_proposals":trial.suggest_categorical("rpn_proposals",[1000,2000,4000]),"roi_score_threshold":trial.suggest_float("roi_score_threshold",0.001,0.1,log=True),"roi_nms_threshold":trial.suggest_float("roi_nms_threshold",0.3,0.7)}
def suggest_hierarchical_backbone(trial:Any)->dict[str,Any]:
    return {"drop_path_rate":trial.suggest_float("drop_path_rate",0.0,0.4),"layerwise_lr_decay":trial.suggest_float("layerwise_lr_decay",0.65,1.0)}
def suggest_rtdetr(trial:Any)->dict[str,Any]:
    return {"num_queries":trial.suggest_categorical("num_queries",[300,500,700]),"decoder_layers":trial.suggest_int("decoder_layers",3,8),"num_denoising":trial.suggest_categorical("num_denoising",[50,100,200]),"matcher_class_cost":trial.suggest_float("matcher_class_cost",1,4),"matcher_bbox_cost":trial.suggest_float("matcher_bbox_cost",2,8),"matcher_giou_cost":trial.suggest_float("matcher_giou_cost",1,4),"weight_loss_vfl":trial.suggest_float("weight_loss_vfl",0.5,2),"weight_loss_bbox":trial.suggest_float("weight_loss_bbox",2,8),"weight_loss_giou":trial.suggest_float("weight_loss_giou",1,4)}
def create_study(storage_dir:str|Path,study_name:str,direction:str="maximize"):
    import optuna
    root=Path(storage_dir); root.mkdir(parents=True,exist_ok=True)
    return optuna.create_study(study_name=study_name,storage=f"sqlite:///{root/study_name}.db",load_if_exists=True,direction=direction,pruner=optuna.pruners.MedianPruner(n_startup_trials=5,n_warmup_steps=3))
