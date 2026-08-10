"""One Git bootstrap implementation, shared by every active notebook."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from src.git_utils import is_git_checkout
from src.notebook_bootstrap import (
    RepositoryStateError,
    bootstrap_notebook,
    inspect_repository,
)

ROOT = Path(__file__).resolve().parents[1]


def _git(path: Path, *arguments: str) -> None:
    subprocess.run(["git", "-C", str(path), *arguments], check=True, capture_output=True)


def _repository(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True, capture_output=True)
    _git(path, "config", "user.email", "test@example.invalid")
    _git(path, "config", "user.name", "Test")
    (path / "src").mkdir(exist_ok=True)
    (path / "src" / "__init__.py").write_text("", encoding="utf-8")
    (path / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "commit", "-qm", "initial")
    return path


def test_normal_git_directory_is_recognized(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repo")

    state = inspect_repository(repository)

    assert is_git_checkout(repository) is True
    assert len(state.commit) == 40
    assert state.dirty is False
    assert state.linked_worktree is False
    assert state.branch is not None
    assert state.detached is False


def test_linked_worktree_with_a_git_file_is_recognized(tmp_path: Path) -> None:
    """`(path / ".git").is_dir()` is False here - the old notebook check broke."""
    repository = _repository(tmp_path / "repo")
    worktree = tmp_path / "linked"
    _git(repository, "worktree", "add", "-q", "--detach", str(worktree))

    assert (worktree / ".git").is_file()
    assert not (worktree / ".git").is_dir()
    assert is_git_checkout(worktree) is True

    state = inspect_repository(worktree)

    assert state.linked_worktree is True
    assert state.detached is True
    assert state.commit == inspect_repository(repository).commit


def test_missing_repository_fails_with_an_actionable_message(tmp_path: Path) -> None:
    plain = tmp_path / "not-a-repo"
    plain.mkdir()

    assert is_git_checkout(plain) is False
    with pytest.raises(RepositoryStateError, match="BENCHMARK_REPO_ROOT"):
        inspect_repository(plain)


def test_dirty_tree_is_reported_and_only_blocks_when_asked(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repo")
    (repository / "src" / "__init__.py").write_text("# edited\n", encoding="utf-8")

    state = inspect_repository(repository)
    assert state.dirty is True
    assert "src/__init__.py" in state.status
    assert "dirty" in bootstrap_notebook(
        repository, platform="local", environ={}, use_google_drive=False
    ).summary()

    with pytest.raises(RepositoryStateError, match="uncommitted changes"):
        bootstrap_notebook(
            repository,
            platform="local",
            require_clean=True,
            environ={},
            use_google_drive=False,
        )


def test_pinned_revision_verification(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repo")
    commit = inspect_repository(repository).commit

    bootstrap = bootstrap_notebook(
        repository,
        platform="local",
        expected_revision=commit,
        environ={},
        use_google_drive=False,
    )
    assert bootstrap.repository.commit == commit

    with pytest.raises(RepositoryStateError, match="not the required revision"):
        bootstrap_notebook(
            repository,
            platform="local",
            expected_revision="0" * 40,
            environ={},
            use_google_drive=False,
        )


def test_bootstrap_never_updates_the_selected_checkout(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repo")
    before = inspect_repository(repository).commit

    bootstrap_notebook(
        repository, platform="local", environ={}, use_google_drive=False
    )

    assert inspect_repository(repository).commit == before


def test_every_notebook_uses_the_shared_bootstrap_and_no_local_git_logic() -> None:
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        assert "bootstrap_notebook" in source, path.name
        assert "rev-parse" not in source, path.name
        assert "--is-inside-work-tree" not in source, path.name
        assert '(REPO_PATH / ".git")' not in source, path.name


def test_no_worktree_hostile_checkout_probe_remains() -> None:
    """`(path / ".git").is_dir()` must not come back as a *checkout* probe.

    The framework provisioner is exempt: it validates clones it created itself
    with ``git clone``, where a linked worktree is impossible by construction and
    a ``.git`` file would mean something unexpected happened.
    """
    exempt = {
        "src/workflows/isolated_environment.py",
        "scripts/verify_model_environments.py",
    }
    offenders = [
        path.relative_to(ROOT).as_posix()
        for directory in ("src", "scripts")
        for path in (ROOT / directory).rglob("*.py")
        if path.relative_to(ROOT).as_posix() not in exempt
        # The trailing colon matches a branch on the probe, not a docstring
        # that explains why the probe is wrong.
        and '.git").is_dir():' in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
