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


BOOTSTRAP_CELL_TEMPLATE = '''\
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY_URL = "https://github.com/Harryphan72007/aerial-object-detection-benchmark.git"
REPOSITORY_BRANCH = "main"
SMOKE_TEST = os.environ.get("SMOKE_TEST", "").lower() in {{"1", "true", "yes"}}

# The only logic a notebook still owns: make `src` importable. Everything after
# this line - Git state, platform detection, paths, dependency policy - lives in
# src/notebook_bootstrap.py so all notebooks behave identically.
_override = os.environ.get("BENCHMARK_REPO_ROOT")
_candidates = (
    [Path(_override).expanduser()]
    if _override
    else [
        Path.cwd(),
        *Path.cwd().parents,
        Path("/content/aerial-object-detection-benchmark"),
        Path("/kaggle/working/aerial-object-detection-benchmark"),
    ]
)
REPO_PATH = next(
    (
        candidate.resolve()
        for candidate in _candidates
        if (candidate / "src" / "notebook_bootstrap.py").is_file()
    ),
    None,
)
if REPO_PATH is None:
    _host = (
        Path("/content")
        if Path("/content").is_dir()
        else Path("/kaggle/working")
        if Path("/kaggle/working").is_dir()
        else None
    )
    if _host is None:
        raise RuntimeError(
            "Run this notebook from the repository, or set BENCHMARK_REPO_ROOT "
            "to an existing clone."
        )
    REPO_PATH = (_host / "aerial-object-detection-benchmark").resolve()
    REPO_PATH.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--branch", REPOSITORY_BRANCH, REPOSITORY_URL, str(REPO_PATH)],
        check=True,
    )
sys.path.insert(0, str(REPO_PATH))

from src.notebook_bootstrap import bootstrap_notebook

bootstrap = bootstrap_notebook(
    REPO_PATH,
    requirements_file={requirements_file},
    use_google_drive=USE_GOOGLE_DRIVE,
    smoke_test=SMOKE_TEST,
)
notebook_environment = bootstrap.environment
REPO_PATH = notebook_environment.repository_root
DRIVE_ROOT = notebook_environment.artifact_root
NOTEBOOK_PLATFORM = notebook_environment.platform
print(bootstrap.summary())
'''


def render_bootstrap_cell(requirements_file: str | None) -> str:
    """Render the one bootstrap cell every canonical notebook must contain.

    The cell cannot become an import: its whole job is to make ``src``
    importable, by locating an existing checkout or cloning one. Copying it into
    each notebook is therefore unavoidable — copying it by hand is not, and hand
    copying is what let one notebook drift to a different dependency policy than
    its three siblings. ``scripts/validate_notebooks.py`` asserts every notebook
    matches this rendering byte for byte.
    """
    return BOOTSTRAP_CELL_TEMPLATE.format(
        requirements_file=repr(requirements_file) if requirements_file else "None"
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
