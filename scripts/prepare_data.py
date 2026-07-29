#!/usr/bin/env python
"""Convert VisDrone train/val splits to validated, deterministic COCO tracks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.conversion import ensure_conversion
from src.data.download import ensure_visdrone_layout
from src.paths import ProjectPaths


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
    if raw.resolve() != paths.raw.resolve():
        raise ValueError(
            "--raw-root must resolve to the canonical persistent VisDrone raw directory"
        )
    ensure_visdrone_layout(raw)
    for track in args.tracks:
        for split in ("train", "val"):
            manifest, action = ensure_conversion(
                paths,
                Path(__file__).resolve().parents[1],
                track,
                split,
                exclude_light_vehicles=args.exclude_light_vehicles,
                max_images=args.max_images_per_split,
            )
            print(
                json.dumps(
                    {"track": track, "split": split, "action": action, **manifest},
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
