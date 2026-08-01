"""Deterministic tiled COCO generation with auditable clipping decisions."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.utils.serialization import write_json


@dataclass(frozen=True)
class TileWindow:
    x: int
    y: int
    width: int
    height: int


def _origins(length: int, tile_size: int, overlap: int) -> tuple[int, ...]:
    if length <= tile_size:
        return (0,)
    stride = tile_size - overlap
    values = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if values[-1] != final:
        values.append(final)
    return tuple(values)


def tile_windows(
    width: int, height: int, *, tile_size: int, overlap: int
) -> tuple[TileWindow, ...]:
    if min(width, height, tile_size) <= 0 or not 0 <= overlap < tile_size:
        raise ValueError("invalid image, tile size, or overlap")
    return tuple(
        TileWindow(
            x=x,
            y=y,
            width=min(tile_size, width - x),
            height=min(tile_size, height - y),
        )
        for y in _origins(height, tile_size, overlap)
        for x in _origins(width, tile_size, overlap)
    )


def clip_annotation(
    annotation: Mapping[str, Any],
    tile: TileWindow,
    *,
    minimum_visible_fraction: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not 0 <= minimum_visible_fraction <= 1:
        raise ValueError("minimum_visible_fraction must be in [0, 1]")
    x, y, width, height = (float(value) for value in annotation["bbox"])
    area = max(0.0, width) * max(0.0, height)
    left = max(x, float(tile.x))
    top = max(y, float(tile.y))
    right = min(x + width, float(tile.x + tile.width))
    bottom = min(y + height, float(tile.y + tile.height))
    clipped_width = max(0.0, right - left)
    clipped_height = max(0.0, bottom - top)
    visible_area = clipped_width * clipped_height
    visible_fraction = visible_area / area if area else 0.0
    ignored = bool(annotation.get("ignore", 0) or annotation.get("iscrowd", 0))
    if visible_area == 0:
        reason = "outside_tile"
        keep = False
    elif ignored:
        reason = "ignored_region"
        keep = True
    elif visible_fraction < minimum_visible_fraction:
        reason = "below_visible_fraction"
        keep = False
    else:
        reason = "kept"
        keep = True
    decision = {
        "source_annotation_id": annotation.get("id"),
        "keep": keep,
        "reason": reason,
        "visible_fraction": visible_fraction,
        "source_bbox": [x, y, width, height],
        "clipped_global_bbox": [left, top, clipped_width, clipped_height],
    }
    if not keep:
        return None, decision
    clipped = copy.deepcopy(dict(annotation))
    clipped["bbox"] = [
        left - tile.x,
        top - tile.y,
        clipped_width,
        clipped_height,
    ]
    clipped["area"] = visible_area
    clipped["visible_fraction"] = visible_fraction
    clipped["source_annotation_id"] = annotation.get("id")
    if ignored:
        clipped["ignore"] = 1
    return clipped, decision


def canonical_dataset_hash(dataset: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dataset, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tile_coco_dataset(
    dataset: Mapping[str, Any],
    *,
    tile_size: int,
    overlap: int,
    minimum_visible_fraction: float,
    tile_version: str = "v1",
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_hash = canonical_dataset_hash(dataset)
    annotations_by_image: dict[int, list[Mapping[str, Any]]] = {}
    for annotation in dataset.get("annotations", []):
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    tiled_images: list[dict[str, Any]] = []
    tiled_annotations: list[dict[str, Any]] = []
    manifest_tiles: list[dict[str, Any]] = []
    next_image_id = 1
    next_annotation_id = 1
    for image in sorted(dataset.get("images", []), key=lambda value: int(value["id"])):
        source_id = int(image["id"])
        stem = Path(str(image["file_name"])).stem
        suffix = Path(str(image["file_name"])).suffix
        for tile in tile_windows(
            int(image["width"]), int(image["height"]), tile_size=tile_size, overlap=overlap
        ):
            file_name = (
                f"tiles/{stem}__x{tile.x:05d}_y{tile.y:05d}"
                f"_w{tile.width}_h{tile.height}{suffix}"
            )
            tiled_images.append(
                {
                    **copy.deepcopy(dict(image)),
                    "id": next_image_id,
                    "file_name": file_name,
                    "width": tile.width,
                    "height": tile.height,
                    "source_image_id": source_id,
                    "tile_offset": [tile.x, tile.y],
                }
            )
            decisions: list[dict[str, Any]] = []
            for annotation in sorted(
                annotations_by_image.get(source_id, []),
                key=lambda value: int(value.get("id", 0)),
            ):
                clipped, decision = clip_annotation(
                    annotation,
                    tile,
                    minimum_visible_fraction=minimum_visible_fraction,
                )
                decisions.append(decision)
                if clipped is not None:
                    clipped["id"] = next_annotation_id
                    clipped["image_id"] = next_image_id
                    tiled_annotations.append(clipped)
                    next_annotation_id += 1
            manifest_tiles.append(
                {
                    "tile_image_id": next_image_id,
                    "file_name": file_name,
                    "source_image_id": source_id,
                    "source_file_name": image["file_name"],
                    "offset": [tile.x, tile.y],
                    "size": [tile.width, tile.height],
                    "empty": not any(decision["keep"] for decision in decisions),
                    "annotation_decisions": decisions,
                }
            )
            next_image_id += 1
    tiled = {
        "images": tiled_images,
        "annotations": tiled_annotations,
        "categories": copy.deepcopy(dataset.get("categories", [])),
    }
    manifest = {
        "schema_version": 1,
        "tile_version": tile_version,
        "source_dataset_hash": source_hash,
        "tiled_dataset_hash": canonical_dataset_hash(tiled),
        "tile_size": tile_size,
        "overlap": overlap,
        "minimum_visible_fraction": minimum_visible_fraction,
        "tiles": manifest_tiles,
    }
    if canonical_dataset_hash(dataset) != source_hash:
        raise RuntimeError("source dataset was mutated during tiling")
    return tiled, manifest


def write_tiled_dataset(
    output_root: str | Path,
    tiled_dataset: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Path]:
    root = Path(output_root)
    dataset_path = root / "annotations" / "instances_tiled.json"
    manifest_path = root / "tile_manifest.json"
    write_json(dataset_path, dict(tiled_dataset), atomic=True)
    write_json(manifest_path, dict(manifest), atomic=True)
    return {"dataset": dataset_path, "manifest": manifest_path}
