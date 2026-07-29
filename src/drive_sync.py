"""Google Drive storage layout and write-safety checks."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from src.paths import ProjectPaths


def initialize_drive_directories(drive_root: str | Path) -> ProjectPaths:
    """Create the complete persistent artifact layout."""
    return ProjectPaths.from_value(drive_root).create()


def validate_drive_writable(drive_root: str | Path) -> None:
    """Fail early if Drive is absent, not a directory, or not writable."""
    root = Path(drive_root).expanduser()
    if not root.exists() or not root.is_dir():
        raise RuntimeError(f"Drive root is unavailable: {root}")
    try:
        fd, name = tempfile.mkstemp(prefix=".write-test-", dir=root)
        os.close(fd)
        Path(name).unlink()
    except OSError as exc:
        raise RuntimeError(f"Drive root is not writable: {root}") from exc


def validate_dataset(paths: ProjectPaths, track: str) -> None:
    paths.validate_track(track)
    dataset = paths.coco(track)
    required = (
        dataset / "annotations" / "instances_train.json",
        dataset / "annotations" / "instances_val.json",
        paths.images("train"),
        paths.images("val"),
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Converted dataset is incomplete:\n" + "\n".join(missing))
