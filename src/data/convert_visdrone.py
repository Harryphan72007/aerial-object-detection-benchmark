"""Convert VisDrone text annotations to COCO JSON without redistributing data."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from PIL import Image
from src.data.collapse_classes import ClassMapping

@dataclass
class ConversionSummary:
    images: int = 0
    annotations: int = 0
    ignored_regions: int = 0
    skipped_invalid: int = 0


def parse_visdrone_line(line: str) -> tuple[int, int, int, int, int, int, int, int]:
    fields = [part.strip() for part in line.strip().split(",")]
    if len(fields) < 8:
        raise ValueError(f"expected at least 8 comma-separated fields, got {len(fields)}: {line!r}")
    return tuple(int(float(value)) for value in fields[:8])  # type: ignore[return-value]


def convert_split(
    image_dir: str | Path,
    annotation_dir: str | Path,
    output_json: str | Path,
    mapping: ClassMapping,
    keep_attributes: bool = True,
) -> ConversionSummary:
    image_dir, annotation_dir, output_json = Path(image_dir), Path(annotation_dir), Path(output_json)
    if not image_dir.is_dir(): raise FileNotFoundError(f"image directory not found: {image_dir}")
    if not annotation_dir.is_dir(): raise FileNotFoundError(f"annotation directory not found: {annotation_dir}")
    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    summary = ConversionSummary(); ann_id = 1
    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for image_id, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as img: width, height = img.size
        images.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})
        summary.images += 1
        annotation_path = annotation_dir / f"{image_path.stem}.txt"
        if not annotation_path.exists(): raise FileNotFoundError(f"missing annotation for {image_path.name}: {annotation_path}")
        for row_number, line in enumerate(annotation_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip(): continue
            try: x, y, w, h, score, category_id, truncation, occlusion = parse_visdrone_line(line)
            except ValueError as exc: raise ValueError(f"{annotation_path}:{row_number}: {exc}") from exc
            if category_id in {0, 11}:
                summary.ignored_regions += 1; continue
            mapped = mapping.map_category(category_id)
            if mapped is None: continue
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x + w), min(height, y + h)
            clipped_w, clipped_h = x2 - x1, y2 - y1
            if clipped_w <= 0 or clipped_h <= 0:
                summary.skipped_invalid += 1; continue
            ann: dict[str, object] = {
                "id": ann_id, "image_id": image_id, "category_id": mapped,
                "bbox": [x1, y1, clipped_w, clipped_h], "area": clipped_w * clipped_h,
                "iscrowd": 0, "segmentation": [],
            }
            if keep_attributes:
                ann["attributes"] = {"visdrone_score": score, "truncation": truncation, "occlusion": occlusion, "original_category_id": category_id}
            annotations.append(ann); ann_id += 1; summary.annotations += 1
    payload = {
        "info": {"description": f"VisDrone2019-DET converted to COCO ({mapping.track})", "research_only_warning": True},
        "licenses": [], "images": images, "annotations": annotations, "categories": mapping.coco_categories(),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary
