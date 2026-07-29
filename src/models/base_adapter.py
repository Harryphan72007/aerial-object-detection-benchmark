"""Common detector adapter interface."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Sequence

class DetectionModelAdapter(ABC):
    """Normalize model loading, inference, prediction export, and profiling."""
    model_id: str
    @abstractmethod
    def load_model(self, checkpoint_path: str | Path, config: dict[str, Any]) -> Any: ...
    @abstractmethod
    def preprocess(self, images: Sequence[Any]) -> Any: ...
    @abstractmethod
    def predict(self, images: Sequence[Any]) -> list[dict[str, Any]]: ...
    @abstractmethod
    def postprocess(self, outputs: Any, original_sizes: Sequence[tuple[int,int]]) -> list[dict[str, Any]]: ...
    def export_predictions_to_coco(self, predictions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        rows=[]
        for item in predictions:
            image_id=int(item["image_id"])
            for box,score,label in zip(item["boxes"],item["scores"],item["labels"]):
                x1,y1,x2,y2=map(float,box); rows.append({"image_id":image_id,"category_id":int(label),"bbox":[x1,y1,x2-x1,y2-y1],"score":float(score)})
        return rows
    @abstractmethod
    def profile(self, sample_batch: Any, **kwargs: Any) -> dict[str, Any]: ...
