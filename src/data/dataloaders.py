"""Lightweight COCO dataset used by RT-DETR and visualization code."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable
from PIL import Image

class CocoDetectionRecords:
    def __init__(self, image_dir: str | Path, annotation_file: str | Path, transform: Callable | None = None):
        self.image_dir=Path(image_dir); self.data=json.loads(Path(annotation_file).read_text(encoding="utf-8")); self.transform=transform
        self.images=sorted(self.data["images"], key=lambda x:int(x["id"])); self.anns=defaultdict(list)
        for ann in self.data["annotations"]: self.anns[int(ann["image_id"])].append(ann)
        self.categories={int(c["id"]):c["name"] for c in self.data["categories"]}
    def __len__(self)->int: return len(self.images)
    def __getitem__(self,index:int)->dict[str,Any]:
        meta=self.images[index]; image=Image.open(self.image_dir/str(meta["file_name"])).convert("RGB")
        record={"image":image,"image_id":int(meta["id"]),"file_name":meta["file_name"],"annotations":self.anns[int(meta["id"])],"original_size":(int(meta["height"]),int(meta["width"]))}
        return self.transform(record) if self.transform else record


def detection_collate(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep variable-size images and targets as records for a detection batch."""
    if not batch:
        raise ValueError("cannot collate an empty detection batch")
    return batch
