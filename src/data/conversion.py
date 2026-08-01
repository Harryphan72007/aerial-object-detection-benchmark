"""Provenance-aware VisDrone-to-COCO conversion."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.class_mapping import ClassMapping, VISDRONE_CLASSES
from src.data.convert_visdrone import convert_split
from src.data.download import EXTRACTION_MANIFEST_SCHEMA_VERSION, sha256_file
from src.data.statistics import compute_statistics
from src.data.validate_annotations import validate_coco
from src.git_utils import git_commit
from src.paths import ProjectPaths
from src.utils.serialization import write_json


CONVERTER_SCHEMA_VERSION = 1


def _class_mapping_payload(mapping: ClassMapping) -> dict[str, Any]:
    return {
        "track": mapping.track,
        "class_names": mapping.class_names,
        "original_to_output": {
            str(category_id): mapping.map_category(category_id)
            for category_id in sorted(VISDRONE_CLASSES)
        },
    }


def _source_payload(
    paths: ProjectPaths,
    repo_root: str | Path,
    track: str,
    split: str,
    *,
    exclude_light_vehicles: bool,
    max_images: int | None,
) -> dict[str, Any]:
    extraction_path = paths.dataset_manifests / f"{split}_extraction.json"
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    if extraction.get("schema_version") != EXTRACTION_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"stale extraction manifest: {extraction_path}")
    mapping = ClassMapping(track, exclude_light_vehicles=exclude_light_vehicles)
    return {
        "track": track,
        "split": split,
        "source_archive_sha256": extraction["archive_sha256"],
        "source_extraction_inventory_sha256": extraction[
            "relative_filename_inventory_sha256"
        ],
        "source_image_count": int(extraction["image_count"]),
        "source_annotation_count": int(extraction["annotation_count"]),
        "class_mapping": _class_mapping_payload(mapping),
        "exclude_light_vehicles": exclude_light_vehicles,
        "converter_schema_version": CONVERTER_SCHEMA_VERSION,
        "repository_git_commit": git_commit(repo_root),
        "max_images": max_images,
    }


def conversion_manifest_status(
    paths: ProjectPaths,
    repo_root: str | Path,
    track: str,
    split: str,
    *,
    exclude_light_vehicles: bool = False,
    max_images: int | None = None,
) -> dict[str, Any]:
    annotation_root = paths.coco(track) / "annotations"
    output = annotation_root / f"instances_{split}.json"
    manifest_path = annotation_root / f"conversion_manifest_{split}.json"
    errors: list[str] = []
    try:
        expected = _source_payload(
            paths,
            repo_root,
            track,
            split,
            exclude_light_vehicles=exclude_light_vehicles,
            max_images=max_images,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return {
            "valid": False,
            "errors": [str(exc)],
            "manifest": str(manifest_path),
            "output": str(output),
        }
    try:
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        recorded = {}
        errors.append(f"conversion manifest is missing or invalid: {manifest_path}")
    for field, value in expected.items():
        if recorded.get(field) != value:
            errors.append(
                f"conversion provenance field {field!r} is "
                f"{recorded.get(field)!r}, expected {value!r}"
            )
    if not output.is_file():
        errors.append(f"COCO output is missing: {output}")
    else:
        try:
            payload = json.loads(output.read_text(encoding="utf-8"))
            current_hash = sha256_file(output)
            if recorded.get("output_coco_sha256") != current_hash:
                errors.append("output COCO SHA-256 differs from conversion manifest")
            if recorded.get("output_image_count") != len(payload.get("images", [])):
                errors.append("output COCO image count differs from conversion manifest")
            if recorded.get("output_annotation_count") != len(
                payload.get("annotations", [])
            ):
                errors.append("output COCO annotation count differs from conversion manifest")
            expected_ids = [1, 2] if track == "2class" else list(range(1, 11))
            actual_ids = [int(row["id"]) for row in payload.get("categories", [])]
            if actual_ids != expected_ids:
                errors.append(
                    f"output category IDs are {actual_ids}, expected {expected_ids}"
                )
            validation = validate_coco(output, paths.images(split))
            errors.extend(validation.errors)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid COCO output: {exc}")
    return {
        "valid": not errors,
        "errors": errors,
        "manifest": str(manifest_path),
        "output": str(output),
        "recorded": recorded,
        "expected": expected,
    }


def ensure_conversion(
    paths: ProjectPaths,
    repo_root: str | Path,
    track: str,
    split: str,
    *,
    exclude_light_vehicles: bool = False,
    max_images: int | None = None,
) -> tuple[dict[str, Any], str]:
    """Reuse only a fully current conversion, otherwise regenerate every artifact."""
    current = conversion_manifest_status(
        paths,
        repo_root,
        track,
        split,
        exclude_light_vehicles=exclude_light_vehicles,
        max_images=max_images,
    )
    if current["valid"]:
        return current["recorded"], "reused"

    mapping = ClassMapping(track, exclude_light_vehicles=exclude_light_vehicles)
    source = _source_payload(
        paths,
        repo_root,
        track,
        split,
        exclude_light_vehicles=exclude_light_vehicles,
        max_images=max_images,
    )
    source_root = paths.official_split(split)
    annotation_root = paths.coco(track) / "annotations"
    annotation_root.mkdir(parents=True, exist_ok=True)
    output = annotation_root / f"instances_{split}.json"
    audit = annotation_root / f"conversion_audit_{split}.json"
    summary = convert_split(
        source_root / "images",
        source_root / "annotations",
        output,
        mapping,
        split=split,
        report_json=audit,
        max_images=max_images,
        source_archive_sha256=str(source["source_archive_sha256"]),
    )
    report = validate_coco(output, source_root / "images")
    report.raise_for_errors()
    write_json(annotation_root / f"statistics_{split}.json", compute_statistics(output))
    provenance = {
        **source,
        "output_coco_sha256": sha256_file(output),
        "output_image_count": summary.images,
        "output_annotation_count": summary.annotations,
        "conversion_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "conversion_audit": asdict(summary),
    }
    manifest_path = annotation_root / f"conversion_manifest_{split}.json"
    write_json(manifest_path, provenance)
    verified = conversion_manifest_status(
        paths,
        repo_root,
        track,
        split,
        exclude_light_vehicles=exclude_light_vehicles,
        max_images=max_images,
    )
    if not verified["valid"]:
        raise RuntimeError(f"conversion verification failed: {verified['errors']}")
    return provenance, "converted"
