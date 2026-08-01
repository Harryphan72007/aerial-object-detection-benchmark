"""Backward-compatible imports for the canonical dataset module."""

from src.data.dataset import (
    CocoDetectionDataset,
    CocoDetectionRecords,
    detection_collate,
)

__all__ = ["CocoDetectionDataset", "CocoDetectionRecords", "detection_collate"]
