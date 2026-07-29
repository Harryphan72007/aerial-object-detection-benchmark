import pytest
from src.models.mmdetection_adapter import MMDetectionAdapter
from src.models.registry import create_adapter,load_model_config
from src.models.rtdetr_adapter import RTDetrV2Adapter

def test_registry_constructs_without_heavy_imports():
    assert isinstance(create_adapter("faster_rcnn_resnet50","cpu"),MMDetectionAdapter)
    assert isinstance(create_adapter("rtdetrv2_l","cpu"),RTDetrV2Adapter)

def test_model_config_resolution():
    cfg=load_model_config("rtdetrv2_l")
    assert cfg["framework"]=="transformers"

def test_unknown_model():
    with pytest.raises(KeyError):create_adapter("unknown")
