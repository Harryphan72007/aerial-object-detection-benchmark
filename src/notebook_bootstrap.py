"""One repository-state and environment bootstrap for every active notebook.

Every notebook needs the same four things before it can do anything: a checkout
it can trust, the repository importable, the platform detected, and the artifact
paths resolved. Duplicating that across sixteen notebooks produced drifting
variants - some probed ``(REPO_PATH / ".git").is_dir()``, which is false in a
linked Git worktree where ``.git`` is a file, and some fast-forwarded the
checkout while others deliberately preserved the selected commit.

This module is the single implementation. Notebooks keep only the few lines that
cannot be shared: locating (and if necessary cloning) the repository so that
``src`` becomes importable at all.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, MutableMapping

from src.git_utils import is_git_checkout
from src.notebook_environment import (
    NotebookEnvironment,
    detect_notebook_platform,
    setup_notebook_environment,
)


class RepositoryStateError(RuntimeError):
    """The checkout a notebook was pointed at cannot be used."""


@dataclass(frozen=True)
class RepositoryState:
    root: Path
    git_dir: Path
    commit: str
    branch: str | None
    dirty: bool
    status: str
    linked_worktree: bool

    @property
    def detached(self) -> bool:
        return self.branch is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "git_dir": str(self.git_dir),
            "commit": self.commit,
            "branch": self.branch,
            "detached": self.detached,
            "dirty": self.dirty,
            "linked_worktree": self.linked_worktree,
        }


def _git(path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(path), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def inspect_repository(path: str | Path) -> RepositoryState:
    """Describe the checkout without modifying it in any way."""
    root = Path(path).expanduser().resolve()
    if not is_git_checkout(root):
        raise RepositoryStateError(
            f"{root} is not a Git checkout. Clone the benchmark repository, or set "
            "BENCHMARK_REPO_ROOT to an existing clone before running this notebook."
        )
    git_dir = _git(root, "rev-parse", "--absolute-git-dir")
    commit = _git(root, "rev-parse", "HEAD")
    if commit.returncode != 0:
        raise RepositoryStateError(
            f"{root} has no commits to run from: {commit.stderr.strip()}"
        )
    branch = _git(root, "branch", "--show-current").stdout.strip()
    status = _git(root, "status", "--porcelain")
    top_level = _git(root, "rev-parse", "--show-toplevel").stdout.strip()
    common_dir = _git(root, "rev-parse", "--path-format=absolute", "--git-common-dir")
    resolved_git_dir = Path(git_dir.stdout.strip() or (root / ".git"))
    return RepositoryState(
        root=Path(top_level or root),
        git_dir=resolved_git_dir,
        commit=commit.stdout.strip(),
        branch=branch or None,
        dirty=bool(status.stdout.strip()),
        status=status.stdout.strip(),
        # A linked worktree has a private git dir under the shared common dir.
        linked_worktree=bool(
            common_dir.returncode == 0
            and common_dir.stdout.strip()
            and Path(common_dir.stdout.strip()) != resolved_git_dir
        ),
    )


@dataclass(frozen=True)
class NotebookBootstrap:
    repository: RepositoryState
    environment: NotebookEnvironment

    def as_dict(self) -> dict[str, Any]:
        return {
            "repository": self.repository.as_dict(),
            "environment": self.environment.as_dict(),
        }

    def summary(self) -> str:
        state = self.repository
        lines = [
            f"Repository: {state.root}",
            f"Commit: {state.commit}"
            + (f" (branch {state.branch})" if state.branch else " (detached HEAD)"),
            f"Working tree: {'dirty' if state.dirty else 'clean'}"
            + (" [linked worktree]" if state.linked_worktree else ""),
            f"Platform: {self.environment.platform}",
            f"Artifact root: {self.environment.artifact_root}",
            f"Local cache root: {self.environment.local_cache_root}",
            f"Model runtime root: {self.environment.model_runtime_root}",
        ]
        if state.dirty:
            lines.append(
                "Uncommitted changes are recorded in every run's provenance; "
                "commit them before publishing results."
            )
        return "\n".join(lines)


def bootstrap_notebook(
    repository_path: str | Path,
    *,
    requirements_file: str | Path | None = None,
    use_google_drive: bool = True,
    smoke_test: bool = False,
    artifact_root: str | Path | None = None,
    platform: str | None = None,
    expected_revision: str | None = None,
    require_clean: bool = False,
    install_dependencies: bool | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> NotebookBootstrap:
    """Validate the checkout, then resolve one platform/path contract.

    The checkout is never updated: notebooks run the commit the operator
    selected, so a fast-forward here would silently change the code under a
    partially completed experiment.
    """
    state = inspect_repository(repository_path)
    if expected_revision and state.commit != expected_revision:
        raise RepositoryStateError(
            f"{state.root} is at {state.commit}, not the required revision "
            f"{expected_revision}. Check out the pinned revision and rerun."
        )
    if require_clean and state.dirty:
        raise RepositoryStateError(
            f"{state.root} has uncommitted changes:\n{state.status}\n"
            "Commit or stash them before running this notebook."
        )
    environment = setup_notebook_environment(
        state.root,
        platform=platform or detect_notebook_platform(),
        artifact_root=artifact_root,
        use_google_drive=use_google_drive,
        requirements_file=requirements_file,
        install_dependencies=install_dependencies,
        smoke_test=smoke_test,
        environ=environ,
    )
    return NotebookBootstrap(repository=state, environment=environment)
