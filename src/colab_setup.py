"""Small, reusable setup helpers for Google Colab notebooks."""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.drive_sync import (
    initialize_drive_directories as _initialize_drive_directories,
    validate_drive_writable as _validate_drive_writable,
)
from src.git_utils import DirtyRepositoryError, ensure_clean_for_update
from src.paths import ProjectPaths
from src.utils.serialization import read_yaml


def mount_drive() -> None:
    """Mount Google Drive; raise a clear error outside Colab."""
    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError("mount_drive() must be run inside Google Colab") from exc
    drive.mount("/content/drive")


def load_project_config(config_path: str | Path) -> dict[str, Any]:
    """Load the YAML configuration used by notebooks and scripts."""
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
    if not (path / ".git").exists():
        raise RuntimeError(
            f"Refusing to clone into an existing non-Git directory: {path}. "
            "Choose an empty path or move the local files first."
        )
    ensure_clean_for_update(path)
    subprocess.run(["git", "-C", str(path), "fetch", "origin"], check=True)
    subprocess.run(["git", "-C", str(path), "checkout", branch], check=True)
    subprocess.run(
        ["git", "-C", str(path), "pull", "--ff-only", "origin", branch], check=True
    )


def install_project(repository_path: str | Path) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(repository_path)], check=True)


def initialize_drive_directories(drive_root: str | Path) -> ProjectPaths:
    """Create and return the canonical persistent Google Drive layout."""
    return _initialize_drive_directories(drive_root)


def validate_drive_writable(drive_root: str | Path) -> None:
    """Raise before training when the configured Drive root cannot be written."""
    _validate_drive_writable(drive_root)


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
    "initialize_drive_directories",
    "install_project",
    "load_project_config",
    "mount_drive",
    "print_environment_summary",
    "validate_drive_writable",
]
