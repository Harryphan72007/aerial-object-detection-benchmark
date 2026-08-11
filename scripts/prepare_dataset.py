#!/usr/bin/env python
"""Prepare and verify the VisDrone data contract on its own.

The model notebooks prepare the dataset automatically, so this is the escape
hatch for the cases they deliberately do not cover: preparing the opt-in
10-class track, forcing a redownload over a corrupt archive, or building the
dataset in one session and training in another.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.paths import configured_drive_root
from src.workflows.dataset_setup import prepare_visdrone

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root")
    parser.add_argument(
        "--config", default=str(REPOSITORY_ROOT / "project_config.yaml")
    )
    parser.add_argument(
        "--dataset-source",
        choices=("auto", "download", "drive", "manual"),
        default="auto",
    )
    parser.add_argument("--prepare-10class-track", action="store_true")
    parser.add_argument(
        "--redownload",
        action="store_true",
        help="refetch archives even when a verified copy exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    drive_root = configured_drive_root(args.config, args.drive_root)
    summary = prepare_visdrone(
        REPOSITORY_ROOT,
        drive_root,
        dataset_source=args.dataset_source,
        prepare_10class_track=args.prepare_10class_track,
        redownload=args.redownload,
    )
    contract = summary["data_contract"]
    print(json.dumps({**summary, "archives": len(summary["archives"])}, indent=2))
    print(
        "DATA CONTRACT VERIFIED: "
        + ("YES" if contract["verified"] else "NO")
    )


if __name__ == "__main__":
    main()
