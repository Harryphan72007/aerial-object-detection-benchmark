"""Small, reusable setup helpers for Google Colab notebooks."""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.git_utils import (
    DirtyRepositoryError,
    ensure_clean_for_update,
    is_git_checkout,
)


def mount_drive() -> None:
    """Mount Google Drive; raise a clear error outside Colab."""
    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError("mount_drive() must be run inside Google Colab") from exc
    drive.mount("/content/drive")


def load_project_config(config_path: str | Path) -> dict[str, Any]:
    """Load the YAML configuration used by notebooks and scripts."""
    from src.utils.serialization import read_yaml

    return read_yaml(config_path)


def clone_or_update_repository(
    repository_url: str, local_path: str | Path, branch: str = "main"
) -> None:
    """Clone or fast-forward a clean checkout without deleting local changes."""
    path = Path(local_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--branch", branch, repository_url, str(path)], check=True
        )
        return
    if not is_git_checkout(path):
        raise RuntimeError(
            f"Refusing to clone into an existing non-Git directory: {path}. "
            "Choose an empty path or move the local files first."
        )
    checkout_repository_ref(path, branch, "branch")


def checkout_repository_ref(
    repository_path: str | Path,
    reference: str,
    reference_type: str = "branch",
) -> dict[str, str | bool]:
    """Fetch and select a branch, tag, or commit without resetting a clone."""
    path = Path(repository_path)
    if not is_git_checkout(path):
        raise RuntimeError(f"Not a Git checkout: {path}")
    if reference_type not in {"branch", "tag", "commit"}:
        raise ValueError("reference_type must be branch, tag, or commit")
    ensure_clean_for_update(path)
    subprocess.run(
        ["git", "-C", str(path), "fetch", "--tags", "--prune", "origin"],
        check=True,
    )
    if reference_type == "branch":
        remote = f"refs/remotes/origin/{reference}"
        subprocess.run(
            ["git", "-C", str(path), "show-ref", "--verify", remote], check=True
        )
        local = subprocess.run(
            ["git", "-C", str(path), "show-ref", "--verify", f"refs/heads/{reference}"],
            check=False,
            capture_output=True,
            text=True,
        )
        if local.returncode == 0:
            subprocess.run(["git", "-C", str(path), "checkout", reference], check=True)
        else:
            subprocess.run(
                ["git", "-C", str(path), "checkout", "--track", "-b", reference, f"origin/{reference}"],
                check=True,
            )
        subprocess.run(
            ["git", "-C", str(path), "merge", "--ff-only", f"origin/{reference}"],
            check=True,
        )
    elif reference_type == "tag":
        tag = f"refs/tags/{reference}"
        subprocess.run(
            ["git", "-C", str(path), "show-ref", "--verify", tag], check=True
        )
        subprocess.run(
            ["git", "-C", str(path), "checkout", "--detach", tag], check=True
        )
    else:
        subprocess.run(
            ["git", "-C", str(path), "cat-file", "-e", f"{reference}^{{commit}}"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(path), "checkout", "--detach", reference], check=True
        )
    commit = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    branch = subprocess.run(
        ["git", "-C", str(path), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "reference": reference,
        "reference_type": reference_type,
        "commit": commit,
        "branch": branch,
        "detached": not bool(branch),
    }


def clone_or_checkout_repository(
    repository_url: str,
    local_path: str | Path,
    reference: str = "main",
    reference_type: str = "branch",
) -> dict[str, str | bool]:
    """Clone when absent, then safely select an exact repository reference."""
    path = Path(local_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", repository_url, str(path)], check=True)
    elif not is_git_checkout(path):
        raise RuntimeError(
            f"Refusing to use an existing non-Git directory: {path}"
        )
    return checkout_repository_ref(path, reference, reference_type)


def install_project(repository_path: str | Path) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(repository_path)], check=True)


def initialize_drive_directories(drive_root: str | Path) -> Any:
    """Create and return the canonical persistent Google Drive layout."""
    from src.drive_sync import initialize_drive_directories as _initialize

    return _initialize(drive_root)


def validate_drive_writable(drive_root: str | Path) -> None:
    """Raise before training when the configured Drive root cannot be written."""
    from src.drive_sync import validate_drive_writable as _validate

    _validate(drive_root)


def print_environment_summary() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.version.cuda}; available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            print(f"GPU: {properties.name}; memory_gb={properties.total_memory / 2**30:.2f}")
    except ImportError:
        print("PyTorch: not installed")


__all__ = [
    "DirtyRepositoryError",
    "clone_or_update_repository",
    "clone_or_checkout_repository",
    "checkout_repository_ref",
    "initialize_drive_directories",
    "install_project",
    "load_project_config",
    "mount_drive",
    "print_environment_summary",
    "validate_drive_writable",
]
