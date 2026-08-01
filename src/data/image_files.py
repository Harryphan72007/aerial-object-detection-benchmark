"""Deterministic discovery of supported raster image inputs."""
from __future__ import annotations

from pathlib import Path

SUPPORTED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def supported_image_files(root: str | Path) -> list[Path]:
    directory = Path(root)
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def first_supported_image(root: str | Path) -> Path:
    images = supported_image_files(root)
    if not images:
        raise FileNotFoundError(f"No supported image files found in {Path(root)}")
    return images[0]
