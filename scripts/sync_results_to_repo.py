#!/usr/bin/env python
"""Preview or export approved lightweight results without staging or committing."""
from __future__ import annotations

import argparse
import json

from src.result_export import export_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-root", required=True)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--clean-target", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-file-size-mb", type=float, default=20)
    args = parser.parse_args()
    result = export_bundle(
        args.drive_root, args.bundle_id, args.repo_root,
        max_file_size_mb=args.max_file_size_mb,
        clean_target=args.clean_target,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))
    print("No files were staged or committed automatically.")


if __name__ == "__main__":
    main()
