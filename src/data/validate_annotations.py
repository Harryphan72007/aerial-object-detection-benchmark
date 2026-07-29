"""Strict COCO annotation validation."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ValidationReport:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    class_counts: dict[int, int] = field(default_factory=dict)
    image_count: int = 0
    annotation_count: int = 0

    def raise_for_errors(self) -> None:
        if not self.valid:
            raise ValueError(
                "invalid COCO annotations:\n"
                + "\n".join(f"- {error}" for error in self.errors[:50])
            )


def validate_coco(
    annotation_file: str | Path,
    image_dir: str | Path,
    require_nonempty: bool = True,
) -> ValidationReport:
    data = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
    image_dir = Path(image_dir)
    errors: list[str] = []
    warnings: list[str] = []
    images = {int(image["id"]): image for image in data.get("images", [])}
    categories = {int(category["id"]) for category in data.get("categories", [])}
    counts: Counter[int] = Counter()
    if require_nonempty and not images:
        errors.append("no images")
    if require_nonempty and not categories:
        errors.append("no categories")
    for image_id, image in images.items():
        path = image_dir / str(image["file_name"])
        if not path.exists():
            errors.append(f"missing image: {path}")
        if int(image.get("width", 0)) <= 0 or int(image.get("height", 0)) <= 0:
            errors.append(f"invalid dimensions for image {image_id}")
    seen_annotations: set[int] = set()
    for annotation in data.get("annotations", []):
        annotation_id = int(annotation["id"])
        if annotation_id in seen_annotations:
            errors.append(f"duplicate annotation id {annotation_id}")
        seen_annotations.add(annotation_id)
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        if image_id not in images:
            errors.append(
                f"annotation {annotation_id} references missing image {image_id}"
            )
        if category_id not in categories:
            errors.append(f"annotation {annotation_id} has invalid category {category_id}")
        bbox = annotation.get("bbox", [])
        if len(bbox) != 4:
            errors.append(f"annotation {annotation_id} bbox must have 4 values")
            continue
        x, y, width, height = map(float, bbox)
        if width <= 0 or height <= 0:
            errors.append(f"annotation {annotation_id} has non-positive area")
        if x < 0 or y < 0:
            errors.append(f"annotation {annotation_id} starts outside image")
        if image_id in images:
            image_width = float(images[image_id]["width"])
            image_height = float(images[image_id]["height"])
            if x + width > image_width + 1e-6 or y + height > image_height + 1e-6:
                errors.append(f"annotation {annotation_id} exceeds image bounds")
        area = float(annotation.get("area", width * height))
        if abs(area - width * height) > max(1.0, 0.01 * width * height):
            warnings.append(
                f"annotation {annotation_id} area differs from bbox area"
            )
        counts[category_id] += 1
    missing_classes = categories - set(counts)
    if missing_classes:
        warnings.append(f"categories with zero annotations: {sorted(missing_classes)}")
    return ValidationReport(
        not errors,
        errors,
        warnings,
        dict(counts),
        len(images),
        len(data.get("annotations", [])),
    )
