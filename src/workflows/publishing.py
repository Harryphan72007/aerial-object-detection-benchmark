"""Dry-run-first publication through a disposable Git clone."""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from scripts.validate_results import validate_repo_results
from src.benchmark_status import discover_model_status
from src.hpo.result_bundle import create_hpo_result_bundle
from src.git_utils import git_status, run_git
from src.paths import ProjectPaths
from src.result_export import create_result_bundle, export_bundle, validate_bundle
from src.workflows.contract import require_primary_model

RESULTS_BRANCH = "experiment-results"
LATEST_MANIFEST = "results/manifests/latest_result_manifest.json"


def _latest_valid_bundle(
    paths: ProjectPaths, model_id: str, run_id: str
) -> Path | None:
    matches = []
    for manifest in paths.result_bundles.glob("*/bundle_manifest.json"):
        try:
            value = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("model_id") == model_id and value.get("run_id") == run_id:
            if not validate_bundle(manifest.parent):
                matches.append(manifest.parent)
    return max(matches, key=lambda item: item.stat().st_mtime) if matches else None


def _latest_valid_hpo_bundle(
    paths: ProjectPaths, model_id: str, dataset_track: str
) -> Path | None:
    matches = []
    for manifest in paths.result_bundles.glob("*/bundle_manifest.json"):
        try:
            value = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            int(value.get("schema_version", 0)) >= 3
            and value.get("model_id") == model_id
            and value.get("dataset_track") == dataset_track
            and value.get("protocol_id") == "two_stage_random_hpo_v1"
            and not validate_bundle(manifest.parent)
        ):
            matches.append(manifest.parent)
    return max(matches, key=lambda item: item.stat().st_mtime) if matches else None


def _repository_slug(remote: str) -> str:
    match = re.search(
        r"(?:github\.com[/:])(?P<slug>[^/:\s]+/[^/\s]+?)(?:\.git)?$",
        remote,
    )
    if not match:
        raise RuntimeError(f"origin is not a GitHub repository: {remote}")
    return match.group("slug")


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _authentication_environment(temporary_root: Path) -> dict[str, str]:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "Publishing authentication is missing. Add GH_TOKEN as a Colab "
            "secret, rerun the dry-run, then set PUBLISH_RESULTS=True and "
            "DRY_RUN=False."
        )
    _run(["gh", "auth", "status"])
    askpass = temporary_root / "git-askpass.sh"
    askpass.write_text(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        "  *Username*) printf '%s\\n' 'x-access-token' ;;\n"
        "  *) printf '%s\\n' \"$GH_TOKEN\" ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    askpass.chmod(0o700)
    environment = dict(os.environ)
    environment.update(
        {
            "GIT_ASKPASS": str(askpass),
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _verify_push_permission(slug: str) -> None:
    permission = _run(
        ["gh", "api", f"repos/{slug}", "--jq", ".permissions.push"]
    ).lower()
    if permission != "true":
        raise RuntimeError(f"Authenticated account cannot push to {slug}")


def _open_or_update_pr(clone: Path, slug: str) -> str:
    existing = _run(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            slug,
            "--base",
            "main",
            "--head",
            RESULTS_BRANCH,
            "--state",
            "open",
            "--json",
            "url",
            "--jq",
            ".[0].url // \"\"",
        ],
        cwd=clone,
    )
    if existing:
        return existing
    return _run(
        [
            "gh",
            "pr",
            "create",
            "--repo",
            slug,
            "--base",
            "main",
            "--head",
            RESULTS_BRANCH,
            "--title",
            "results: publish validated lightweight benchmark bundle",
            "--body",
            (
                "Validated lightweight result bundle. Runtime datasets, "
                "predictions, checkpoints, logs, and credentials remain excluded."
            ),
        ],
        cwd=clone,
    )


def _safe_cleanup(temporary_root: Path) -> None:
    resolved = temporary_root.resolve()
    expected_parent = Path(tempfile.gettempdir()).resolve()
    if resolved.parent != expected_parent or not resolved.name.startswith(
        "visdrone-results-publish-"
    ):
        raise RuntimeError(
            f"Refusing to clean unexpected temporary path: {resolved}"
        )
    def make_writable_and_retry(function, path, _error):
        os.chmod(path, stat.S_IWRITE)
        function(path)

    shutil.rmtree(resolved, onerror=make_writable_and_retry)


def _validate_source_checkout(repo: Path) -> None:
    dirty = git_status(repo)
    if dirty:
        raise RuntimeError(
            "Refusing to publish from a dirty source checkout:\n" + dirty
        )
    branch = run_git(repo, "branch", "--show-current")
    if branch != "main":
        raise RuntimeError(
            f"Publishing source checkout must remain on clean main, got {branch}"
        )


def _validate_staged_paths(staged: list[str], bundle_id: str) -> None:
    bundle_prefix = f"results/bundles/{bundle_id}/"
    unexpected = [
        path
        for path in staged
        if path != LATEST_MANIFEST and not path.startswith(bundle_prefix)
    ]
    if not staged or LATEST_MANIFEST not in staged or unexpected:
        raise RuntimeError(
            "Staging safety check failed. Expected exactly the selected "
            f"bundle and latest manifest; staged={staged}"
        )


def _prepare_publication_clone(
    source_repo: Path,
    temporary_root: Path,
    environment: dict[str, str],
) -> tuple[Path, bool]:
    remote = run_git(source_repo, "remote", "get-url", "origin")
    clone = temporary_root / "repository"
    _run(
        ["git", "clone", "--no-checkout", remote, str(clone)],
        environment=environment,
    )
    branch_probe = subprocess.run(
        [
            "git",
            "ls-remote",
            "--exit-code",
            "--heads",
            "origin",
            RESULTS_BRANCH,
        ],
        cwd=clone,
        env=environment,
        text=True,
        capture_output=True,
    )
    exists = branch_probe.returncode == 0
    if branch_probe.returncode not in {0, 2}:
        raise RuntimeError(
            f"Could not inspect {RESULTS_BRANCH}: {branch_probe.stderr.strip()}"
        )
    if exists:
        _run(
            ["git", "fetch", "origin", RESULTS_BRANCH],
            cwd=clone,
            environment=environment,
        )
        _run(
            [
                "git",
                "switch",
                "-c",
                RESULTS_BRANCH,
                "--track",
                f"origin/{RESULTS_BRANCH}",
            ],
            cwd=clone,
            environment=environment,
        )
        _run(
            ["git", "merge", "--ff-only", f"origin/{RESULTS_BRANCH}"],
            cwd=clone,
            environment=environment,
        )
    else:
        _run(
            ["git", "fetch", "origin", "main"],
            cwd=clone,
            environment=environment,
        )
        _run(
            ["git", "switch", "-c", RESULTS_BRANCH, "origin/main"],
            cwd=clone,
            environment=environment,
        )
    _run(["git", "config", "user.name", "VisDrone Results Bot"], cwd=clone)
    _run(
        ["git", "config", "user.email", "visdrone-results@users.noreply.github.com"],
        cwd=clone,
    )
    return clone, exists


def publish_results(
    repo_root: str | Path,
    drive_root: str | Path,
    model_id: str,
    *,
    dataset_track: str = "2class",
    publish_results: bool = False,
    dry_run: bool = True,
) -> dict[str, Any]:
    require_primary_model(model_id)
    if dataset_track not in {"2class", "10class"}:
        raise ValueError(f"unsupported dataset track: {dataset_track}")
    if publish_results == dry_run:
        if publish_results:
            raise ValueError(
                "Publishing requires PUBLISH_RESULTS=True and DRY_RUN=False"
            )
        if not dry_run:
            raise ValueError("DRY_RUN=False requires PUBLISH_RESULTS=True")
    repo = Path(repo_root).resolve()
    paths = ProjectPaths.from_value(drive_root)
    try:
        bundle = _latest_valid_hpo_bundle(paths, model_id, dataset_track)
        if bundle is None:
            bundle = create_hpo_result_bundle(
                paths.root, repo, model_id, dataset_track
            )
        bundle_manifest = json.loads(
            (bundle / "bundle_manifest.json").read_text(encoding="utf-8")
        )
        run_id = ",".join(bundle_manifest["run_ids"])
    except (
        FileNotFoundError,
        KeyError,
        RuntimeError,
        ValueError,
    ) as hpo_error:
        if dataset_track != "2class":
            raise RuntimeError(
                f"No publishable HPO result for {model_id} {dataset_track}: "
                f"{hpo_error}"
            ) from hpo_error
        status = discover_model_status(paths.root, model_id, repo)
        if status["final_training_status"] != "COMPLETE":
            raise RuntimeError(
                f"No compatible completed result for {model_id}: {hpo_error}"
            ) from hpo_error
        model_report = (
            paths.reports
            / "models"
            / model_id
            / str(status["final_run_id"])
            / "final_report.md"
        )
        if (
            status["evaluation_status"] != "COMPLETE"
            or not model_report.is_file()
        ):
            raise RuntimeError(
                "Evaluation and model report must be complete before publishing"
            )
        run_id = str(status["final_run_id"])
        bundle = _latest_valid_bundle(paths, model_id, run_id)
        if bundle is None:
            bundle = create_result_bundle(
                paths.root,
                "2class",
                repo,
                model_id=model_id,
                run_id=run_id,
            )
    errors = validate_bundle(bundle)
    if errors:
        raise RuntimeError("Bundle validation failed:\n" + "\n".join(errors))
    before = git_status(repo)
    preview = export_bundle(paths.root, bundle.name, repo, dry_run=True)
    if git_status(repo) != before:
        raise RuntimeError("Dry-run modified the Git worktree")
    result: dict[str, Any] = {
        "model_id": model_id,
        "dataset_track": dataset_track,
        "run_id": run_id,
        "bundle_id": bundle.name,
        "bundle_path": str(bundle),
        "validation_errors": errors,
        "preview": preview,
        "git_status_before": before,
        "git_status_after": git_status(repo),
        "published": False,
    }
    if dry_run:
        return result
    _validate_source_checkout(repo)
    remote = run_git(repo, "remote", "get-url", "origin")
    slug = _repository_slug(remote)
    temporary_root = Path(
        tempfile.mkdtemp(prefix="visdrone-results-publish-")
    ).resolve()
    try:
        environment = _authentication_environment(temporary_root)
        _verify_push_permission(slug)
        clone, result_branch_existed = _prepare_publication_clone(
            repo, temporary_root, environment
        )
        exported = export_bundle(
            paths.root, bundle.name, clone, dry_run=False
        )
        _run(
            [
                "git",
                "add",
                "--",
                f"results/bundles/{bundle.name}",
                LATEST_MANIFEST,
            ],
            cwd=clone,
        )
        staged = _run(
            ["git", "diff", "--cached", "--name-only"], cwd=clone
        ).splitlines()
        _validate_staged_paths(staged, bundle.name)
        validation_errors = validate_repo_results(clone / "results")
        if validation_errors:
            raise RuntimeError(
                "Complete staged results tree is invalid:\n"
                + "\n".join(validation_errors)
            )
        statistics = _run(
            ["git", "diff", "--cached", "--stat"], cwd=clone
        )
        _run(
            [
                "git",
                "commit",
                "-m",
                f"results({model_id}): publish validated benchmark bundle",
            ],
            cwd=clone,
        )
        _run(
            ["git", "push", "-u", "origin", RESULTS_BRANCH],
            cwd=clone,
            environment=environment,
        )
        pr_url = _open_or_update_pr(clone, slug)
        result.update(
            {
                "published": True,
                "result_branch_existed": result_branch_existed,
                "export": exported,
                "staged_files": staged,
                "staged_statistics": statistics,
                "pull_request": pr_url,
            }
        )
        return result
    finally:
        primary_error_active = sys.exc_info()[0] is not None
        try:
            _safe_cleanup(temporary_root)
        except Exception as cleanup_error:
            if primary_error_active:
                print(
                    "WARNING: publication cleanup also failed while another error "
                    f"was active: {cleanup_error!r}",
                    file=sys.stderr,
                )
            elif result.get("published"):
                result.setdefault("warnings", []).append(
                    {
                        "operation": "cleanup_publication_temporary_directory",
                        "exception_type": type(cleanup_error).__name__,
                        "message": str(cleanup_error),
                        "scientific_artifacts_valid": True,
                    }
                )
            else:
                raise
