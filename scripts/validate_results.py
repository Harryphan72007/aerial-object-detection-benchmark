#!/usr/bin/env python
"""Validate the Git-side lightweight result publication."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from src.models.registry import MODEL_CONFIGS
from src.result_export import (
    EXCLUDED_EXTENSIONS,
    find_secret_like_content,
    validate_metric_value,
)


def validate_repo_results(results_root: str | Path, max_file_size_mb: float = 20) -> list[str]:
    root = Path(results_root)
    errors: list[str] = []
    manifest_path = root / "manifests" / "latest_result_manifest.json"
    if not manifest_path.exists():
        return [f"missing {manifest_path}"]
    for required in (
        "manifests/latest_result_manifest.json",
        "tables/final_results.csv",
        "tables/final_results.json",
        "tables/training_efficiency.csv",
        "tables/inference_efficiency.csv",
        "tables/per_class_metrics.csv",
        "tables/per_size_metrics.csv",
        "tables/statistical_summary.csv",
        "tables/recommendation_matrix.csv",
        "reports/pull_request_summary.md",
    ):
        if not (root / required).exists():
            errors.append(f"missing required result file: {required}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"invalid latest result manifest: {exc}"]
    track = manifest.get("dataset_track")
    if track not in {"2class", "10class"}:
        errors.append("latest manifest has invalid dataset_track")
    bundle_id = manifest.get("result_bundle_id", "")
    if track and track not in bundle_id:
        errors.append("result bundle ID does not include its dataset track")
    for run_id in manifest.get("selected_run_ids", []):
        if not isinstance(run_id, str) or not run_id:
            errors.append("selected run ID is empty")
    class_names = tuple(manifest.get("class_names", []))
    if not class_names:
        errors.append("latest manifest has no class_names")
    manifest_models = manifest.get("model_ids", [])
    if not manifest_models:
        errors.append("latest manifest has no model_ids")
    for model_id in manifest_models:
        if model_id not in MODEL_CONFIGS:
            errors.append(f"unrecognized model ID in manifest: {model_id}")
    final_results = root / "tables" / "final_results.csv"
    if not final_results.exists():
        final_results = root / "final_results.csv"
    if final_results.exists():
        lines = final_results.read_text(encoding="utf-8").splitlines()
        if not lines or "model_id" not in lines[0]:
            errors.append("final_results.csv is missing model_id")
        else:
            columns = {column.strip() for column in lines[0].split(",")}
            if not columns.intersection({"AP", "ap", "mAP", "map", "AP50", "map50"}):
                errors.append("final_results.csv is missing AP/mAP metric columns")
            for row in csv.DictReader(lines):
                if row.get("model_id") not in MODEL_CONFIGS:
                    errors.append(f"unrecognized model ID: {row.get('model_id')}")
                if track and row.get("dataset_track") and row.get("dataset_track") != track:
                    errors.append("final results mix dataset tracks")
                row_classes = row.get("class_names")
                if row_classes and tuple(str(row_classes).strip("[]").replace("'", "").split(", ")) != class_names:
                    errors.append("final results have incompatible class names")
                for name, value in row.items():
                    errors.extend(validate_metric_value(name, value))
    final_json = root / "tables" / "final_results.json"
    if final_json.exists():
        try:
            rows = json.loads(final_json.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                errors.append("final_results.json must contain a list")
            for row in rows:
                if row.get("model_id") not in MODEL_CONFIGS:
                    errors.append(f"unrecognized model ID: {row.get('model_id')}")
                if track and row.get("dataset_track") and row.get("dataset_track") != track:
                    errors.append("final JSON results mix dataset tracks")
                if row.get("class_names") and tuple(row["class_names"]) != class_names:
                    errors.append("final JSON results have incompatible class names")
                for name, value in row.items():
                    errors.extend(validate_metric_value(name, value))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid final_results.json: {exc}")
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
    reports = list((root / "reports").glob("*.md")) + list((root / "reports").glob("*.html"))
    if reports and not any(re.search(r"single-seed|multi-seed", report.read_text(encoding="utf-8"), re.I) for report in reports):
        errors.append("reports must state single-seed or multi-seed status")
    if reports and bundle_id and not any(bundle_id in report.read_text(encoding="utf-8") for report in reports):
        errors.append("reports must include the result bundle ID")
    return sorted(set(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-results", default="results")
    parser.add_argument("--max-file-size-mb", type=float, default=20)
    args = parser.parse_args()
    errors = validate_repo_results(args.repo_results, args.max_file_size_mb)
    if errors:
        print("Result validation failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        raise SystemExit(1)
    print(f"Result validation passed: {Path(args.repo_results).resolve()}")


if __name__ == "__main__":
    main()
