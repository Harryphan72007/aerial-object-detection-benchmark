from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

VISDRONE_CLASSES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)


def parse_annotation_line(line: str) -> dict[str, int]:
    values = [part.strip() for part in line.split(",")]
    if len(values) < 8:
        raise ValueError(f"Expected at least 8 comma-separated fields, received {len(values)}")
    x, y, width, height, score, class_id, truncation, occlusion = map(int, values[:8])
    if width < 0 or height < 0:
        raise ValueError("Bounding-box width and height must be non-negative")
    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "score": score,
        "class_id": class_id,
        "truncation": truncation,
        "occlusion": occlusion,
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_split(split_dir: str | Path, output: str | Path) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install the 'eval' extra to convert VisDrone images") from exc

    split_path = Path(split_dir)
    image_dir = split_path / "images"
    annotation_dir = split_path / "annotations"
    if not image_dir.is_dir() or not annotation_dir.is_dir():
        raise FileNotFoundError("Expected sibling 'images' and 'annotations' directories")

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    image_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    for image_id, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        images.append(
            {"id": image_id, "file_name": image_path.name, "width": width, "height": height}
        )
        annotation_path = annotation_dir / f"{image_path.stem}.txt"
        if not annotation_path.exists():
            raise FileNotFoundError(f"Missing annotation for {image_path.name}")
        for raw_line in annotation_path.read_text(encoding="utf-8-sig").splitlines():
            if not raw_line.strip():
                continue
            row = parse_annotation_line(raw_line)
            if row["class_id"] == 0 or row["score"] == 0:
                continue
            if row["class_id"] > len(VISDRONE_CLASSES):
                raise ValueError(f"Unsupported class id {row['class_id']} in {annotation_path}")
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": row["class_id"],
                    "bbox": [row["x"], row["y"], row["width"], row["height"]],
                    "area": row["width"] * row["height"],
                    "iscrowd": 0,
                    "truncation": row["truncation"],
                    "occlusion": row["occlusion"],
                }
            )
            annotation_id += 1

    payload = {
        "info": {
            "description": "VisDrone conversion; source images are not redistributed",
            "source_path": str(split_path.resolve()),
        },
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": index, "name": name} for index, name in enumerate(VISDRONE_CLASSES, start=1)
        ],
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "images": len(images),
        "annotations": len(annotations),
        "output": str(output_path),
        "sha256": file_sha256(output_path),
    }
