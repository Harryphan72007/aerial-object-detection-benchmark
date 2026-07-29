#!/usr/bin/env python
"""Create a versioned Drive result bundle from completed evaluation outputs."""
from __future__ import annotations

import argparse

from src.result_export import create_result_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-root", required=True)
    parser.add_argument("--dataset-track", choices=["2class", "10class"], required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--bundle-id")
    args = parser.parse_args()
    output = create_result_bundle(args.drive_root, args.dataset_track, args.repo_root, args.bundle_id)
    print(output)


if __name__ == "__main__":
    main()
