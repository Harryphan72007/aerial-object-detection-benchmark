#!/usr/bin/env python
"""Dry-run-first migration of completed runs to the single-checkpoint policy."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from src.paths import ProjectPaths
from src.training.checkpointing import (
    enforce_completed_checkpoint_policy,
    materialize_checkpoint_alias,
    model_checkpoint_files,
    resolve_manifest_checkpoint,
    validate_checkpoint_identity,
    validate_manifest_dict,
)
from src.utils.serialization import read_json, sha256_file, write_json


def _inside(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    boundary = root.resolve()
    return resolved == boundary or boundary in resolved.parents


def _manifest_paths(paths: ProjectPaths) -> list[Path]:
    roots = (paths.checkpoints, paths.root / "hpo")
    manifests: list[Path] = []
    for root in roots:
        if root.is_dir():
            manifests.extend(root.rglob("run_manifest.json"))
    return sorted(set(manifests))


def cleanup_checkpoints(
    drive_root: str | Path,
    *,
    apply: bool = False,
    report_path: str | Path | None = None,
    loader: Callable[[Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Plan or apply bounded cleanup without touching incomplete runs."""
    paths = ProjectPaths.from_value(drive_root)
    allowed_roots = (paths.checkpoints.resolve(), (paths.root / "hpo").resolve())
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "drive_root": str(paths.root),
        "mode": "apply" if apply else "dry-run",
        "runs": [],
    }
    for manifest_path in _manifest_paths(paths):
        run_dir = manifest_path.parent.resolve()
        entry: dict[str, Any] = {
            "run_id": run_dir.name,
            "status": "unknown",
            "selected_source_checkpoint": None,
            "canonical_destination": str(run_dir / "best.pth"),
            "checksum": None,
            "planned_removals": [],
            "removed_files": [],
            "skipped_files": [],
            "warnings": [],
            "errors": [],
        }
        report["runs"].append(entry)
        if not any(_inside(run_dir, root) for root in allowed_roots):
            entry["errors"].append("run directory is outside configured boundaries")
            continue
        try:
            manifest = read_json(manifest_path)
        except Exception as error:
            entry["errors"].append(f"manifest unreadable: {error}")
            continue
        if not isinstance(manifest, dict):
            entry["errors"].append("manifest is not a JSON object")
            continue
        entry["run_id"] = str(manifest.get("run_id", run_dir.name))
        entry["status"] = str(manifest.get("status", "unknown"))
        files = model_checkpoint_files(run_dir)
        if entry["status"] != "completed":
            entry["skipped_files"] = [path.name for path in files]
            entry["warnings"].append("incomplete/failed/unknown run was not modified")
            continue
        errors = validate_manifest_dict(manifest, check_files=False)
        if errors:
            entry["errors"].extend(errors)
            entry["skipped_files"] = [path.name for path in files]
            continue
        try:
            source = resolve_manifest_checkpoint(
                manifest,
                run_dir,
                allow_legacy_aliases=True,
            )
            if source.resolve().parent != run_dir:
                raise ValueError("selected checkpoint is outside its run directory")
            expected_identity = (
                manifest.get("checkpoint_identity")
                if int(manifest.get("schema_version", 1)) >= 2
                else None
            )
            validate_checkpoint_identity(source, expected_identity, loader=loader)
        except Exception as error:
            entry["errors"].append(f"checkpoint validation failed: {error}")
            entry["skipped_files"] = [path.name for path in files]
            continue
        canonical = run_dir / "best.pth"
        entry["selected_source_checkpoint"] = str(source)
        entry["checksum"] = sha256_file(source)
        hpo_trial = str(manifest.get("run_kind", "")).startswith("hpo_")
        planned = [path.name for path in files if path != canonical]
        if source != canonical and "best.pth" not in planned:
            planned.append(source.name)
        if hpo_trial:
            planned.append("best.pth")
            entry["warnings"].append(
                "completed HPO trial weights are disposable; all model files are planned"
            )
        entry["planned_removals"] = sorted(set(planned))
        if not apply:
            continue
        if source != canonical:
            materialize_checkpoint_alias(source, canonical)
        validate_checkpoint_identity(canonical, expected_identity, loader=loader)
        if sha256_file(canonical) != entry["checksum"]:
            entry["errors"].append("canonical checkpoint checksum mismatch")
            continue
        if hpo_trial:
            for path in model_checkpoint_files(run_dir):
                if path.resolve().parent != run_dir:
                    raise ValueError("checkpoint deletion escaped run directory")
                path.unlink()
                entry["removed_files"].append(path.name)
        else:
            entry["removed_files"] = enforce_completed_checkpoint_policy(run_dir)
        entry["removed_files"] = sorted(entry["removed_files"])
    destination = Path(
        report_path or (paths.reports / "checkpoint_cleanup_latest.json")
    )
    write_json(destination, report)
    report["report_path"] = str(destination)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args()
    if args.dry_run and args.apply:
        parser.error("choose either --dry-run or --apply")
    return args


def main() -> None:
    args = parse_args()
    report = cleanup_checkpoints(
        args.drive_root,
        apply=bool(args.apply),
        report_path=args.report,
    )
    print(f"Checkpoint cleanup {report['mode']}: {report['report_path']}")


if __name__ == "__main__":
    main()
