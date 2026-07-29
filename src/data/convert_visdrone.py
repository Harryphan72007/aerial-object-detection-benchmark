"""Deterministically convert VisDrone DET annotations to auditable COCO JSON."""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from src.data.collapse_classes import ClassMapping


@dataclass
class ConversionSummary:
    images: int = 0
    annotations: int = 0
    ignored_regions: int = 0
    missing_images: int = 0
    missing_annotations: int = 0
    malformed_rows: int = 0
    zero_area_boxes: int = 0
    negative_coordinates: int = 0
    out_of_bounds_boxes: int = 0
    unknown_category_ids: int = 0
    images_without_valid_objects: int = 0
    skipped_invalid: int = 0  # Backward-compatible aggregate.
    issue_examples: list[str] | None = None
    objects_by_class: dict[str, int] | None = None
    objects_by_size: dict[str, int] | None = None
    objects_by_truncation: dict[str, int] | None = None
    objects_by_occlusion: dict[str, int] | None = None

    def __post_init__(self) -> None:
        self.issue_examples = [] if self.issue_examples is None else self.issue_examples
        self.objects_by_class = {} if self.objects_by_class is None else self.objects_by_class
        self.objects_by_size = {} if self.objects_by_size is None else self.objects_by_size
        self.objects_by_truncation = (
            {} if self.objects_by_truncation is None else self.objects_by_truncation
        )
        self.objects_by_occlusion = (
            {} if self.objects_by_occlusion is None else self.objects_by_occlusion
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__  # type: ignore[attr-defined]
        }


def parse_visdrone_line(line: str) -> tuple[int, int, int, int, int, int, int, int]:
    """Parse the eight official DET columns, tolerating a trailing comma only."""
    fields = [part.strip() for part in line.strip().split(",")]
    if len(fields) == 9 and fields[-1] == "":
        fields.pop()
    if len(fields) != 8:
        raise ValueError(
            f"expected exactly 8 comma-separated fields, got {len(fields)}: {line!r}"
        )
    values: list[int] = []
    for index, value in enumerate(fields):
        try:
            numeric = float(value)
        except ValueError as exc:
            raise ValueError(f"column {index + 1} is not numeric: {value!r}") from exc
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"column {index + 1} must be a finite integer: {value!r}")
        values.append(int(numeric))
    return tuple(values)  # type: ignore[return-value]


def validate_box(
    x: int, y: int, width: int, height: int, image_width: int, image_height: int
) -> list[str]:
    """Return every box defect; conversion never silently clips a raw box."""
    issues: list[str] = []
    if width <= 0 or height <= 0:
        issues.append("zero_area")
    if x < 0 or y < 0:
        issues.append("negative_coordinates")
    if x + width > image_width or y + height > image_height:
        issues.append("out_of_bounds")
    return issues


def _size_name(area: int) -> str:
    if area < 16**2:
        return "tiny"
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def convert_split(
    image_dir: str | Path,
    annotation_dir: str | Path,
    output_json: str | Path,
    mapping: ClassMapping,
    keep_attributes: bool = True,
    split: str | None = None,
    report_json: str | Path | None = None,
    max_images: int | None = None,
) -> ConversionSummary:
    image_dir = Path(image_dir)
    annotation_dir = Path(annotation_dir)
    output_json = Path(output_json)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image directory not found: {image_dir}")
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"annotation directory not found: {annotation_dir}")
    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    summary = ConversionSummary()
    class_counts: Counter[str] = Counter()
    size_counts: Counter[str] = Counter()
    truncation_counts: Counter[str] = Counter()
    occlusion_counts: Counter[str] = Counter()
    ann_id = 1
    image_paths = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    image_stems = {path.stem for path in image_paths}
    orphan_annotations = sorted(
        path for path in annotation_dir.glob("*.txt") if path.stem not in image_stems
    )
    summary.missing_images = len(orphan_annotations)
    for path in orphan_annotations[:50]:
        summary.issue_examples.append(f"missing image for annotation: {path}")
    if max_images is not None:
        if max_images <= 0:
            raise ValueError("max_images must be positive")
        image_paths = image_paths[:max_images]
    for image_id, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as img:
            width, height = img.size
            img.verify()
        images.append(
            {"id": image_id, "file_name": image_path.name, "width": width, "height": height}
        )
        summary.images += 1
        annotation_path = annotation_dir / f"{image_path.stem}.txt"
        before = summary.annotations
        if not annotation_path.exists():
            summary.missing_annotations += 1
            summary.issue_examples.append(f"missing annotation: {annotation_path}")
            summary.images_without_valid_objects += 1
            continue
        lines = annotation_path.read_text(encoding="utf-8").splitlines()
        for row_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            location = f"{annotation_path}:{row_number}"
            try:
                x, y, w, h, score, category_id, truncation, occlusion = (
                    parse_visdrone_line(line)
                )
            except ValueError as exc:
                summary.malformed_rows += 1
                summary.skipped_invalid += 1
                if len(summary.issue_examples) < 50:
                    summary.issue_examples.append(f"{location}: {exc}")
                continue
            if category_id in {0, 11}:
                summary.ignored_regions += 1
                continue
            mapped = mapping.map_category(category_id)
            if mapped is None:
                if category_id not in range(1, 11):
                    summary.unknown_category_ids += 1
                    summary.skipped_invalid += 1
                    if len(summary.issue_examples) < 50:
                        summary.issue_examples.append(
                            f"{location}: unknown category id {category_id}"
                        )
                continue
            box_issues = validate_box(x, y, w, h, width, height)
            if box_issues:
                summary.skipped_invalid += 1
                summary.zero_area_boxes += "zero_area" in box_issues
                summary.negative_coordinates += "negative_coordinates" in box_issues
                summary.out_of_bounds_boxes += "out_of_bounds" in box_issues
                if len(summary.issue_examples) < 50:
                    summary.issue_examples.append(
                        f"{location}: invalid bbox {[x, y, w, h]} ({', '.join(box_issues)})"
                    )
                continue
            ann: dict[str, object] = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": mapped,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": [],
            }
            if keep_attributes:
                ann["attributes"] = {
                    "visdrone_score": score,
                    "truncation": truncation,
                    "occlusion": occlusion,
                    "original_category_id": category_id,
                }
            annotations.append(ann)
            ann_id += 1
            summary.annotations += 1
            class_counts[mapping.class_names[mapped - 1]] += 1
            size_counts[_size_name(w * h)] += 1
            truncation_counts[str(truncation)] += 1
            occlusion_counts[str(occlusion)] += 1
        if summary.annotations == before:
            summary.images_without_valid_objects += 1
    payload = {
        "info": {
            "description": f"VisDrone2019-DET converted to COCO ({mapping.track})",
            "split": split,
            "research_only_warning": True,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": mapping.coco_categories(),
    }
    summary.objects_by_class = dict(sorted(class_counts.items()))
    summary.objects_by_size = dict(sorted(size_counts.items()))
    summary.objects_by_truncation = dict(sorted(truncation_counts.items()))
    summary.objects_by_occlusion = dict(sorted(occlusion_counts.items()))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if report_json:
        report_path = Path(report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary
