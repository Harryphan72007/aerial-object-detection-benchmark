#!/usr/bin/env python
"""Validate the Git-side lightweight result publication."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from src.models.registry import MODEL_CONFIGS
from src.result_export import (
    EXCLUDED_EXTENSIONS,
    find_secret_like_content,
    validate_bundle,
)


def validate_repo_results(results_root: str | Path, max_file_size_mb: float = 20) -> list[str]:
    root = Path(results_root)
    errors: list[str] = []
    manifest_path = root / "manifests" / "latest_result_manifest.json"
    if not manifest_path.exists():
        return [f"missing {manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"invalid latest result manifest: {exc}"]
    bundle_path = root / str(
        manifest.get(
            "bundle_path",
            f"bundles/{manifest.get('result_bundle_id', '')}",
        )
    )
    errors.extend(validate_bundle(bundle_path, max_file_size_mb))
    track = manifest.get("dataset_track")
    if track not in {"2class", "10class"}:
        errors.append("latest manifest has invalid dataset_track")
    bundle_id = manifest.get("result_bundle_id", "")
    if track and track not in bundle_id:
        errors.append("result bundle ID does not include its dataset track")
    model_id = manifest.get("model_id")
    if model_id not in MODEL_CONFIGS:
        errors.append(f"unrecognized model ID in latest manifest: {model_id}")
    if not manifest.get("run_id"):
        errors.append("latest manifest has no run_id")
    for file in root.rglob("*"):
        if not file.is_file():
            continue
        if file.suffix.lower() in EXCLUDED_EXTENSIONS or file.suffix.lower() in {".zip", ".tar", ".gz", ".db"}:
            errors.append(f"excluded artifact present: {file.relative_to(root)}")
        if file.stat().st_size > max_file_size_mb * 1024 * 1024:
            errors.append(f"oversized result file: {file.relative_to(root)}")
        try:
            text = file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if "/content/drive/" in text or re.search(r"[A-Za-z]:[\\/]Users[\\/]", text):
            errors.append(f"private absolute path: {file.relative_to(root)}")
        for pattern in find_secret_like_content(text):
            errors.append(f"secret-like content in {file.relative_to(root)}: {pattern}")
    reports = list(bundle_path.glob("reports/*.md")) + list(bundle_path.glob("reports/*.html"))
    if reports and not any(re.search(r"single-seed|multi-seed", report.read_text(encoding="utf-8"), re.I) for report in reports):
        errors.append("reports must state single-seed or multi-seed status")
    if reports and bundle_id and not any(bundle_id in report.read_text(encoding="utf-8") for report in reports):
        errors.append("reports must include the result bundle ID")
    return sorted(set(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bundle-path")
    group.add_argument("--repo-results")
    parser.add_argument("--max-file-size-mb", type=float, default=20)
    args = parser.parse_args()
    target = args.bundle_path or args.repo_results
    errors = (
        validate_bundle(args.bundle_path, args.max_file_size_mb)
        if args.bundle_path
        else validate_repo_results(args.repo_results, args.max_file_size_mb)
    )
    if errors:
        print("Result validation failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        raise SystemExit(1)
    print(f"Result validation passed: {Path(target).resolve()}")


if __name__ == "__main__":
    main()
