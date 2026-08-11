#!/usr/bin/env python
"""Preview or publish one model's validated lightweight result bundle.

Publishing pushes to a public repository, so it never happens implicitly:
``--dry-run`` is the default and the real push additionally requires
``--publish``. The report notebook drives the same workflow function.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.paths import configured_drive_root
from src.workflows.contract import PRIMARY_MODELS
from src.workflows.publishing import publish_results

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", choices=PRIMARY_MODELS, required=True)
    parser.add_argument("--drive-root")
    parser.add_argument(
        "--config", default=str(REPOSITORY_ROOT / "project_config.yaml")
    )
    parser.add_argument(
        "--dataset-track", choices=("2class", "10class"), default="2class"
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="perform the real push; without it this is a dry run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    drive_root = configured_drive_root(args.config, args.drive_root)
    result = publish_results(
        REPOSITORY_ROOT,
        drive_root,
        args.model_id,
        dataset_track=args.dataset_track,
        publish_results=args.publish,
        dry_run=not args.publish,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
