"""Faster R-CNN ResNet-50 model factory."""

from src.models.resnet_frcnn.factory import (
    RESNET_FACTORY_MODEL_ID,
    ResNetFRCNNFactory,
    build_resnet_frcnn,
)

__all__ = ["RESNET_FACTORY_MODEL_ID", "ResNetFRCNNFactory", "build_resnet_frcnn"]
