#!/usr/bin/env python
"""Acquire and verify the pretrained checkpoints a run requires.

Provisioning fetches these automatically. This entry point exists for the cases
where it cannot: a runtime with automatic downloads disabled
(``VISDRONE_ALLOW_CHECKPOINT_DOWNLOAD=0``), an offline GPU host that needs the
file staged in advance, or a manual re-verification of an existing artifact
root. It applies exactly the same SHA-256 and size contract as provisioning.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.utils.serialization import read_yaml
from src.workflows.pretrained_checkpoints import (
    RUNTIME_CONFIG,
    CheckpointVerificationError,
    ensure_pretrained_checkpoint,
    load_pretrained_spec,
    verify_checkpoint,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--drive-root", required=True)
    parser.add_argument("--family", default=None, help="limit to one runtime family")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="fail if the checkpoint is absent or invalid instead of downloading",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    drive_root = Path(args.drive_root).expanduser().resolve()
    config = read_yaml(repo_root / RUNTIME_CONFIG)
    families = [args.family] if args.family else sorted(config)
    records: list[dict[str, object]] = []
    failed = False
    for family in families:
        block = config.get(family)
        if not isinstance(block, dict) or "pretrained" not in block:
            continue
        spec = load_pretrained_spec(block["pretrained"])
        destination = drive_root / "pretrained" / spec.filename
        try:
            record = (
                verify_checkpoint(destination, spec)
                if args.verify_only
                else ensure_pretrained_checkpoint(spec, destination)
            )
        except CheckpointVerificationError as error:
            failed = True
            print(f"{family}: FAILED {error}", file=sys.stderr)
            continue
        records.append({"family": family, **record})
        print(f"{family}: OK {record['path']} ({record.get('action', 'verified')})")
    print(json.dumps(records, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
