#!/usr/bin/env python
"""Convert VisDrone train/val splits to validated, deterministic COCO tracks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.data.collapse_classes import ClassMapping
from src.data.convert_visdrone import convert_split
from src.data.download import ensure_visdrone_layout
from src.data.statistics import compute_statistics
from src.data.validate_annotations import validate_coco
from src.paths import ProjectPaths


def _link_images(source: Path, destination: Path) -> None:
    """Create a persistent reference without replacing an existing directory."""
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"{destination} points to the wrong image directory")
        return
    if destination.exists():
        if not destination.is_dir():
            raise RuntimeError(f"image destination is not a directory: {destination}")
        return
    try:
        destination.symlink_to(source.resolve(), target_is_directory=True)
    except OSError:
        # Some Drive/FUSE and Windows configurations cannot create symlinks.
        destination.mkdir(parents=True)
        for image in sorted(source.iterdir()):
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            target = destination / image.name
            try:
                os.link(image, target)
            except OSError:
                target.symlink_to(image.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-root", required=True)
    parser.add_argument("--raw-root")
    parser.add_argument(
        "--tracks",
        nargs="+",
        choices=["2class", "10class"],
        default=["2class", "10class"],
    )
    parser.add_argument("--exclude-light-vehicles", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--max-images-per-split", type=int)
    args = parser.parse_args()
    paths = ProjectPaths.from_value(args.drive_root).create()
    raw = Path(args.raw_root) if args.raw_root else paths.raw
    ensure_visdrone_layout(raw)
    for track in args.tracks:
        mapping = ClassMapping(
            track, exclude_light_vehicles=args.exclude_light_vehicles
        )
        output_root = paths.coco(track)
        annotation_root = output_root / "annotations"
        annotation_root.mkdir(parents=True, exist_ok=True)
        for split, source in (
            ("train", "VisDrone2019-DET-train"),
            ("val", "VisDrone2019-DET-val"),
        ):
            source_root = raw / source
            destination_images = output_root / split
            _link_images(source_root / "images", destination_images)
            annotation_file = annotation_root / f"instances_{split}.json"
            audit_file = annotation_root / f"conversion_audit_{split}.json"
            summary = convert_split(
                source_root / "images",
                source_root / "annotations",
                annotation_file,
                mapping,
                split=split,
                report_json=audit_file,
                max_images=args.max_images_per_split,
            )
            print(json.dumps({"track": track, "split": split, **summary.to_dict()}, indent=2))
            if args.validate:
                report = validate_coco(annotation_file, destination_images)
                print(json.dumps(report.__dict__, indent=2))
                report.raise_for_errors()
            stats = compute_statistics(annotation_file)
            (annotation_root / f"statistics_{split}.json").write_text(
                json.dumps(stats, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
