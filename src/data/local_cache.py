"""Ephemeral local image cache for Colab throughput."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.paths import ProjectPaths


@dataclass(frozen=True)
class DataAccessPaths:
    mode: str
    train_images: Path
    validation_images: Path
    coco_annotation_dir: Path
    lr_manifest_dir: Path
    cache_root: Path | None
    status: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {
            **payload,
            "train_images": str(self.train_images),
            "validation_images": str(self.validation_images),
            "coco_annotation_dir": str(self.coco_annotation_dir),
            "lr_manifest_dir": str(self.lr_manifest_dir),
            "cache_root": str(self.cache_root) if self.cache_root else None,
        }


class InsufficientLocalCacheSpace(RuntimeError):
    """Raised instead of silently switching away from the requested cache mode."""


def _filename_inventory(directory: Path) -> tuple[int, str, int]:
    files = sorted(
        (path for path in directory.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(directory).as_posix(),
    )
    names = [path.relative_to(directory).as_posix() for path in files]
    checksum = hashlib.sha256(
        ("\n".join(names) + ("\n" if names else "")).encode("utf-8")
    ).hexdigest()
    return len(files), checksum, sum(path.stat().st_size for path in files)


def _synchronize_directory(source: Path, destination: Path) -> tuple[int, int]:
    destination.mkdir(parents=True, exist_ok=True)
    source_files = {
        path.relative_to(source): path
        for path in source.rglob("*")
        if path.is_file()
    }
    copied = 0
    copied_bytes = 0
    for relative, source_path in sorted(
        source_files.items(), key=lambda item: item[0].as_posix()
    ):
        target = destination / relative
        if not target.is_file() or target.stat().st_size != source_path.stat().st_size:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target)
            copied += 1
            copied_bytes += source_path.stat().st_size
    for target in sorted(
        (path for path in destination.rglob("*") if path.is_file()), reverse=True
    ):
        if target.relative_to(destination) not in source_files:
            target.unlink()
    for directory in sorted(
        (path for path in destination.rglob("*") if path.is_dir()), reverse=True
    ):
        if not any(directory.iterdir()):
            directory.rmdir()
    return copied, copied_bytes


def _copy_json_files(source: Path, destination: Path) -> tuple[int, int]:
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    copied_bytes = 0
    expected = {path.name for path in source.glob("*.json")}
    for source_path in sorted(source.glob("*.json")):
        target = destination / source_path.name
        if (
            not target.is_file()
            or target.stat().st_size != source_path.stat().st_size
            or target.read_bytes() != source_path.read_bytes()
        ):
            shutil.copy2(source_path, target)
            copied += 1
            copied_bytes += source_path.stat().st_size
    for target in destination.glob("*.json"):
        if target.name not in expected:
            target.unlink()
    return copied, copied_bytes


def verify_local_cache(
    paths: ProjectPaths,
    cache_root: str | Path,
) -> dict[str, Any]:
    root = Path(cache_root)
    errors: list[str] = []
    splits: dict[str, Any] = {}
    for split, local_name in (("train", "train"), ("val", "val")):
        drive_images = paths.images(split)
        local_images = root / local_name / "images"
        try:
            extraction_manifest = json.loads(
                (
                    paths.dataset_manifests / f"{split}_extraction.json"
                ).read_text(encoding="utf-8")
            )
            manifest_image_count = int(extraction_manifest["image_count"])
            manifest_image_hash = str(
                extraction_manifest["image_filename_inventory_sha256"]
            )
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"{split} extraction manifest is unavailable: {exc}")
            continue
        if not local_images.is_dir():
            errors.append(f"local cache image directory is missing: {local_images}")
            continue
        drive_count, drive_hash, drive_bytes = _filename_inventory(drive_images)
        local_count, local_hash, local_bytes = _filename_inventory(local_images)
        split_valid = (
            drive_count == local_count
            and drive_hash == local_hash
            and drive_bytes == local_bytes
            and drive_count == manifest_image_count
            and drive_hash == manifest_image_hash
        )
        if not split_valid:
            errors.append(
                f"{split} local image inventory differs from verified Drive data"
            )
        splits[split] = {
            "valid": split_valid,
            "drive_image_count": drive_count,
            "local_image_count": local_count,
            "drive_filename_inventory_sha256": drive_hash,
            "local_filename_inventory_sha256": local_hash,
            "manifest_filename_inventory_sha256": manifest_image_hash,
            "drive_bytes": drive_bytes,
            "local_bytes": local_bytes,
        }
    required_json = [
        (
            paths.coco("2class") / "annotations" / f"instances_{split}.json",
            root / "annotations" / "coco_2class" / f"instances_{split}.json",
        )
        for split in ("train", "val")
    ] + [
        (
            paths.lr_search_manifests / name,
            root / "annotations" / "lr_search" / name,
        )
        for name in (
            "search_train_seed42.json",
            "search_validation_seed42.json",
            "official_full_train.json",
            "official_validation.json",
            "split_summary.json",
        )
    ]
    for source, local in required_json:
        if not local.is_file():
            errors.append(f"local cache JSON is missing: {local}")
        elif not source.is_file():
            errors.append(f"persistent source JSON is missing: {source}")
        elif source.read_bytes() != local.read_bytes():
            errors.append(f"local cache JSON differs from Drive: {local}")
    return {
        "enabled": True,
        "valid": not errors,
        "errors": errors,
        "root": str(root),
        "splits": splits,
        "size_bytes": sum(
            path.stat().st_size for path in root.rglob("*") if path.is_file()
        )
        if root.exists()
        else 0,
    }


def prepare_local_cache(
    paths: ProjectPaths,
    cache_root: str | Path = "/content/visdrone_cache",
) -> DataAccessPaths:
    root = Path(cache_root)
    required_bytes = sum(
        path.stat().st_size
        for split in ("train", "val")
        for path in paths.images(split).rglob("*")
        if path.is_file()
    )
    existing_bytes = (
        sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        if root.exists()
        else 0
    )
    root.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(root.parent).free
    additional_bytes = max(0, required_bytes - existing_bytes)
    reserve_bytes = 512 * 1024 * 1024
    if free_bytes < additional_bytes + reserve_bytes:
        raise InsufficientLocalCacheSpace(
            "Insufficient local disk for DATA_ACCESS_MODE='local_cache': "
            f"need about {(additional_bytes + reserve_bytes) / 2**30:.2f} GiB, "
            f"have {free_bytes / 2**30:.2f} GiB free. "
            "Set DATA_ACCESS_MODE='drive_direct' explicitly and rerun; no automatic "
            "fallback was performed."
        )

    started = time.monotonic()
    copied_files = 0
    copied_bytes = 0
    for split, local_name in (("train", "train"), ("val", "val")):
        count, size = _synchronize_directory(
            paths.images(split), root / local_name / "images"
        )
        copied_files += count
        copied_bytes += size
    count, size = _copy_json_files(
        paths.coco("2class") / "annotations",
        root / "annotations" / "coco_2class",
    )
    copied_files += count
    copied_bytes += size
    count, size = _copy_json_files(
        paths.lr_search_manifests,
        root / "annotations" / "lr_search",
    )
    copied_files += count
    copied_bytes += size
    status = verify_local_cache(paths, root)
    status.update(
        {
            "copy_seconds": round(time.monotonic() - started, 3),
            "copied_files": copied_files,
            "copied_bytes": copied_bytes,
        }
    )
    if not status["valid"]:
        raise RuntimeError(f"local cache verification failed: {status['errors']}")
    print(
        json.dumps(
            {
                "local_cache": "VERIFIED",
                "root": str(root),
                "size_gib": round(status["size_bytes"] / 2**30, 3),
                "copy_seconds": status["copy_seconds"],
                "copied_files": copied_files,
            },
            indent=2,
        )
    )
    return DataAccessPaths(
        mode="local_cache",
        train_images=root / "train" / "images",
        validation_images=root / "val" / "images",
        coco_annotation_dir=root / "annotations" / "coco_2class",
        lr_manifest_dir=root / "annotations" / "lr_search",
        cache_root=root,
        status=status,
    )


def resolve_data_access(
    paths: ProjectPaths,
    mode: str,
    *,
    cache_root: str | Path = "/content/visdrone_cache",
) -> DataAccessPaths:
    if mode == "local_cache":
        return prepare_local_cache(paths, cache_root)
    if mode != "drive_direct":
        raise ValueError("DATA_ACCESS_MODE must be 'local_cache' or 'drive_direct'")
    annotation_dir = paths.coco("2class") / "annotations"
    return DataAccessPaths(
        mode="drive_direct",
        train_images=paths.images("train"),
        validation_images=paths.images("val"),
        coco_annotation_dir=annotation_dir,
        lr_manifest_dir=paths.lr_search_manifests,
        cache_root=None,
        status={"enabled": False, "valid": True, "root": None},
    )
