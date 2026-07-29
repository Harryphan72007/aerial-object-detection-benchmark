"""Dry-run-first result discovery, validation, and publishing."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from src.benchmark_status import discover_model_status
from src.git_utils import git_status, run_git
from src.paths import ProjectPaths
from src.result_export import create_result_bundle, export_bundle, validate_bundle
from src.workflows.contract import require_primary_model


def _latest_valid_bundle(paths: ProjectPaths, model_id: str, run_id: str) -> Path | None:
    matches = []
    for manifest in paths.result_bundles.glob("*/bundle_manifest.json"):
        try:
            import json

            value = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("model_id") == model_id and value.get("run_id") == run_id:
            if not validate_bundle(manifest.parent):
                matches.append(manifest.parent)
    return max(matches, key=lambda item: item.stat().st_mtime) if matches else None


def publish_results(
    repo_root: str | Path,
    drive_root: str | Path,
    model_id: str,
    *,
    publish_results: bool = False,
    dry_run: bool = True,
) -> dict[str, Any]:
    require_primary_model(model_id)
    if publish_results == dry_run:
        if publish_results:
            raise ValueError("Publishing requires PUBLISH_RESULTS=True and DRY_RUN=False")
        if not dry_run:
            raise ValueError("DRY_RUN=False requires PUBLISH_RESULTS=True")
    repo = Path(repo_root).resolve()
    paths = ProjectPaths.from_value(drive_root)
    status = discover_model_status(paths.root, model_id, repo)
    if status["final_training_status"] != "COMPLETE":
        raise RuntimeError(f"No compatible completed final run for {model_id}")
    model_report = (
        paths.reports
        / "models"
        / model_id
        / str(status["final_run_id"])
        / "final_report.md"
    )
    if status["evaluation_status"] != "COMPLETE" or not model_report.is_file():
        raise RuntimeError("Evaluation and model report must be complete before publishing")
    run_id = str(status["final_run_id"])
    bundle = _latest_valid_bundle(paths, model_id, run_id)
    if bundle is None:
        bundle = create_result_bundle(
            paths.root, "2class", repo, model_id=model_id, run_id=run_id
        )
    errors = validate_bundle(bundle)
    if errors:
        raise RuntimeError("Bundle validation failed:\n" + "\n".join(errors))
    before = git_status(repo)
    preview = export_bundle(paths.root, bundle.name, repo, dry_run=True)
    after = git_status(repo)
    if before != after:
        raise RuntimeError("Dry-run modified the Git worktree")
    result: dict[str, Any] = {
        "model_id": model_id,
        "run_id": run_id,
        "bundle_id": bundle.name,
        "bundle_path": str(bundle),
        "bundle_size_bytes": sum(
            path.stat().st_size for path in bundle.rglob("*") if path.is_file()
        ),
        "validation_errors": errors,
        "preview": preview,
        "git_status_before": before,
        "git_status_after": after,
        "published": False,
    }
    if dry_run:
        return result
    if before:
        raise RuntimeError("Refusing to publish from a dirty repository")
    run_git(repo, "fetch", "origin", "experiment-results")
    branches = run_git(repo, "branch", "--list", "experiment-results")
    if branches:
        run_git(repo, "switch", "experiment-results")
        run_git(repo, "pull", "--ff-only", "origin", "experiment-results")
    else:
        run_git(repo, "switch", "-c", "experiment-results", "origin/experiment-results")
    exported = export_bundle(paths.root, bundle.name, repo, dry_run=False)
    target = repo / "results" / "bundles" / bundle.name
    run_git(repo, "add", "--", str(target.relative_to(repo)))
    staged = run_git(repo, "diff", "--cached", "--name-only").splitlines()
    if not staged or any(not path.startswith("results/bundles/") for path in staged):
        raise RuntimeError(f"Staging safety check failed: {staged}")
    statistics = run_git(repo, "diff", "--cached", "--stat")
    run_git(repo, "commit", "-m", f"results({model_id}): add 2-class benchmark result")
    run_git(repo, "push", "-u", "origin", "experiment-results")
    pr_url = None
    try:
        pr_url = subprocess.check_output(
            [
                "gh",
                "pr",
                "create",
                "--base",
                "main",
                "--head",
                "experiment-results",
                "--title",
                f"results({model_id}): publish controlled benchmark result",
                "--body",
                f"Validated lightweight result bundle `{bundle.name}`.",
            ],
            cwd=repo,
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    result.update(
        {
            "published": True,
            "export": exported,
            "staged_files": staged,
            "staged_statistics": statistics,
            "pull_request": pr_url,
        }
    )
    return result
