"""Canonical lightweight COCO detection dataset and collator."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from PIL import Image


class CocoDetectionDataset:
    """Load stable image IDs and variable-length COCO targets without PyTorch."""

    def __init__(
        self,
        image_dir: str | Path,
        annotation_file: str | Path,
        transform: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.annotation_file = Path(annotation_file)
        self.data = json.loads(self.annotation_file.read_text(encoding="utf-8"))
        self.transform = transform
        self.images = sorted(self.data["images"], key=lambda item: int(item["id"]))
        self.annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in self.data["annotations"]:
            self.annotations[int(annotation["image_id"])].append(annotation)
        self.anns = self.annotations  # Legacy public attribute.
        self.categories = {
            int(category["id"]): str(category["name"])
            for category in self.data["categories"]
        }

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Any:
        metadata = self.images[index]
        image_id = int(metadata["id"])
        image = Image.open(self.image_dir / str(metadata["file_name"])).convert("RGB")
        record = {
            "image": image,
            "image_id": image_id,
            "file_name": str(metadata["file_name"]),
            "annotations": [dict(item) for item in self.annotations[image_id]],
            "original_size": (int(metadata["height"]), int(metadata["width"])),
        }
        return self.transform(record) if self.transform else record


CocoDetectionRecords = CocoDetectionDataset


def detection_collate(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep variable-size images and targets as records for a detection batch."""

    if not batch:
        raise ValueError("cannot collate an empty detection batch")
    return batch
