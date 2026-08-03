from __future__ import annotations

import subprocess
from pathlib import Path

import nbformat
import pytest

from src.colab_setup import DirtyRepositoryError, clone_or_checkout_repository


ROOT = Path(__file__).resolve().parents[1]


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _remote(tmp_path: Path) -> tuple[Path, str]:
    source = tmp_path / "source"
    remote = tmp_path / "remote.git"
    source.mkdir()
    _git(source, "init", "-b", "main")
    (source / "README.md").write_text("first\n", encoding="utf-8")
    _git(source, "add", "README.md")
    subprocess.run(
        ["git", "-C", str(source), "-c", "user.name=Bootstrap Test", "-c", "user.email=bootstrap@example.invalid", "commit", "-m", "first"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = _git(source, "rev-parse", "HEAD")
    _git(source, "tag", "v1")
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    _git(source, "remote", "add", "origin", str(remote))
    _git(source, "push", "origin", "main", "--tags")
    subprocess.run(["git", "-C", str(remote), "symbolic-ref", "HEAD", "refs/heads/main"], check=True)
    return remote, commit


def test_clone_branch_tag_commit_and_refuse_dirty_update(tmp_path: Path) -> None:
    remote, commit = _remote(tmp_path)
    clone = tmp_path / "clone"
    branch = clone_or_checkout_repository(str(remote), clone, "main", "branch")
    assert branch["commit"] == commit and not branch["detached"]

    tag = clone_or_checkout_repository(str(remote), clone, "v1", "tag")
    assert tag["commit"] == commit and tag["detached"]
    exact = clone_or_checkout_repository(str(remote), clone, commit, "commit")
    assert exact["commit"] == commit and exact["detached"]

    (clone / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(DirtyRepositoryError):
        clone_or_checkout_repository(str(remote), clone, "main", "branch")


def test_bootstrap_notebook_is_cross_platform_clean_and_never_starts_training() -> None:
    notebook = nbformat.read(ROOT / "notebooks" / "00_bootstrap_colab.ipynb", as_version=4)
    nbformat.validate(notebook)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert cell.outputs == []
    assert "checkout_repository_ref" in code
    assert "setup_notebook_environment" in code
    assert "IN_KAGGLE" in code
    assert "run_diagnostics" in code
    assert "trainer" not in code.lower()
    assert "START_EXPENSIVE_STAGE" not in code
