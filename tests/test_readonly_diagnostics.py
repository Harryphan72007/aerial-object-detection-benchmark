from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.diagnostics.run_diagnostics import build_report
from src.diagnostics.environment import inspect_environment
from src.diagnostics.repository import inspect_repository


ROOT = Path(__file__).resolve().parents[1]


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    (repository / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    _git(repository, "add", "tracked.txt")
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Diagnostics Test",
            "-c",
            "user.email=diagnostics@example.invalid",
            "commit",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return repository


def test_repository_diagnostic_handles_clean_dirty_and_detached(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    clean = inspect_repository(repository)
    assert clean.is_repository and not clean.dirty
    assert clean.branch == "main" and not clean.detached

    (repository / "tracked.txt").write_text("changed\n", encoding="utf-8")
    dirty = inspect_repository(repository)
    assert dirty.dirty
    assert "tracked.txt" in dirty.changed_paths

    _git(repository, "restore", "tracked.txt")
    _git(repository, "checkout", "--detach", "HEAD")
    detached = inspect_repository(repository)
    assert detached.detached and detached.branch is None
    assert not detached.dirty


def test_repository_diagnostic_handles_non_git_directory(tmp_path: Path) -> None:
    state = inspect_repository(tmp_path)
    assert not state.is_repository
    assert state.commit is None


def test_environment_diagnostic_handles_cpu_only_and_no_gpu(monkeypatch) -> None:
    monkeypatch.setattr("src.diagnostics.environment.shutil.which", lambda _: None)
    report = inspect_environment(packages=("definitely-not-installed-package",))
    assert report["gpu"] == {
        "available": False,
        "nvidia_smi": None,
        "devices": [],
    }
    assert report["packages"]["definitely-not-installed-package"] is None
    assert report["model_modules_imported"] is False


def test_canonical_diagnostic_cli_is_json_and_read_only() -> None:
    before = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.diagnostics.run_diagnostics",
            "--repo-root",
            str(ROOT),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    after = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    )
    report = json.loads(completed.stdout)
    assert before == after
    assert report["repository"]["is_repository"] is True
    assert report["environment"]["read_only"] is True
    assert build_report(ROOT)["environment"]["model_construction_performed"] is False
