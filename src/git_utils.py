"""Safe Git inspection and provenance helpers used by Colab and training."""
from __future__ import annotations

import subprocess
from pathlib import Path


class DirtyRepositoryError(RuntimeError):
    """Raised when an update would risk overwriting local changes."""


def is_git_checkout(path: str | Path) -> bool:
    """Recognize a normal clone *and* a linked worktree.

    ``(path / ".git").is_dir()`` is false inside a linked worktree, where
    ``.git`` is a file pointing at the shared repository, so every caller must
    ask Git itself.
    """
    probe = subprocess.run(
        ["git", "-C", str(Path(path)), "rev-parse", "--is-inside-work-tree"],
        check=False,
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0 and probe.stdout.strip().lower() == "true"


def run_git(repo_root: str | Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def git_status(repo_root: str | Path) -> str:
    """Return exactly the short porcelain status used for run provenance."""
    return run_git(repo_root, "status", "--porcelain")


def git_commit(repo_root: str | Path) -> str:
    return run_git(repo_root, "rev-parse", "HEAD")


def git_provenance(repo_root: str | Path) -> dict[str, str | bool | None]:
    """Capture commit, status, and a patch when the source tree is dirty."""
    status = git_status(repo_root)
    return {
        "git_commit": git_commit(repo_root),
        "git_status": status,
        "git_dirty": bool(status),
        "source_patch": None,
    }


def write_git_provenance(repo_root: str | Path, run_dir: str | Path) -> dict[str, str | bool | None]:
    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    provenance = git_provenance(repo_root)
    (destination / "git_commit.txt").write_text(
        f"{provenance['git_commit']}\n", encoding="utf-8"
    )
    (destination / "git_status.txt").write_text(
        f"{provenance['git_status']}\n", encoding="utf-8"
    )
    if provenance["git_dirty"]:
        patch = subprocess.run(
            ["git", "-C", str(repo_root), "diff"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        patch_path = destination / "source_changes.patch"
        patch_path.write_text(patch, encoding="utf-8")
        provenance["source_patch"] = str(patch_path)
    return provenance


def ensure_clean_for_update(repo_root: str | Path) -> None:
    status = git_status(repo_root)
    if status:
        raise DirtyRepositoryError(
            "Repository has uncommitted changes; refusing to pull:\n" + status
        )


def configure_identity(repo_root: str | Path, name: str, email: str) -> None:
    """Set identity only in the current repository."""
    run_git(repo_root, "config", "user.name", name)
    run_git(repo_root, "config", "user.email", email)
