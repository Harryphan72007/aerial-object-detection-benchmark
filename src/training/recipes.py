"""Fixed optimizer recipe values shared by training and LR audit code."""
from __future__ import annotations

RTDETR_OPTIMIZER_TYPE = "AdamW"
RTDETR_BASELINE_LR = 1e-4
RTDETR_WEIGHT_DECAY = 0.05
RTDETR_SCHEDULER_TYPE = "CosineAnnealingLR"
RTDETR_WARMUP_EPOCHS = 0
RTDETR_GRADIENT_CLIP = 1.0
RTDETR_MAX_DETECTIONS = 500
