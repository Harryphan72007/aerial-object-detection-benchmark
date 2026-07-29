"""Download, verify, inventory, and atomically extract official VisDrone archives."""
from __future__ import annotations

import hashlib
import json
import shutil
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.utils.serialization import write_json


ARCHIVE_MANIFEST_SCHEMA_VERSION = 1
EXTRACTION_MANIFEST_SCHEMA_VERSION = 1

# Token-free mirrors used by Ultralytics' public VisDrone recipe. Checksums are
# recorded from the downloaded bytes; no unofficial checksum is invented here.
VISDRONE_ARCHIVES = {
    "train": {
        "folder": "VisDrone2019-DET-train",
        "filename": "VisDrone2019-DET-train.zip",
        "url": (
            "https://github.com/ultralytics/yolov5/releases/download/v1.0/"
            "VisDrone2019-DET-train.zip"
        ),
        "minimum_bytes": 1_000_000_000,
    },
    "val": {
        "folder": "VisDrone2019-DET-val",
        "filename": "VisDrone2019-DET-val.zip",
        "url": (
            "https://github.com/ultralytics/yolov5/releases/download/v1.0/"
            "VisDrone2019-DET-val.zip"
        ),
        "minimum_bytes": 50_000_000,
    },
}


@dataclass(frozen=True)
class ArchiveManifest:
    schema_version: int
    split: str
    source_url: str
    archive_path: str
    size_bytes: int
    sha256: str
    downloaded_at_utc: str


@dataclass(frozen=True)
class ArchiveEnsureResult:
    manifest: ArchiveManifest
    action: str


@dataclass(frozen=True)
class ExtractionManifest:
    schema_version: int
    split: str
    archive_sha256: str
    archive_size_bytes: int
    top_level_folder: str
    image_count: int
    annotation_count: int
    relative_filename_inventory_sha256: str
    image_filename_inventory_sha256: str
    annotation_filename_inventory_sha256: str
    total_extracted_bytes: int
    completed_at_utc: str


class ManualArchiveRequired(RuntimeError):
    """Raised when valid archives must be placed on Drive by the user."""


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_zip(
    archive_path: str | Path,
    expected_folder: str,
    minimum_bytes: int = 1,
) -> None:
    path = Path(archive_path)
    if not path.is_file():
        raise FileNotFoundError(f"archive not found: {path}")
    if path.stat().st_size < minimum_bytes:
        raise ValueError(
            f"archive is unexpectedly small: {path} "
            f"({path.stat().st_size:,} < {minimum_bytes:,} bytes)"
        )
    if not zipfile.is_zipfile(path):
        raise ValueError(f"not a valid ZIP archive: {path}")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if not any(name.startswith(f"{expected_folder}/") for name in names):
            raise ValueError(
                f"{path} does not contain expected top-level folder {expected_folder}/"
            )
        corrupt_member = archive.testzip()
        if corrupt_member:
            raise ValueError(f"CRC failure in {path}: {corrupt_member}")


def download_resumable(
    url: str,
    destination: str | Path,
    *,
    redownload: bool = False,
    progress: Callable[[int], None] | None = None,
) -> Path:
    """Download with a Range request when a `.part` file already exists."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not redownload:
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    if redownload:
        partial.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "visdrone-benchmark/0.1"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except (OSError, urllib.error.HTTPError) as exc:
        status = f"HTTP {exc.code}" if isinstance(exc, urllib.error.HTTPError) else repr(exc)
        raise RuntimeError(f"download failed ({status}): {url}") from exc
    status = getattr(response, "status", 200)
    if offset and status != 206:
        offset = 0
        partial.unlink(missing_ok=True)
    mode = "ab" if offset else "wb"
    with response, partial.open(mode) as output:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            if progress:
                progress(len(chunk))
    partial.replace(destination)
    return destination


def manual_archive_instructions(archive_dir: str | Path) -> str:
    destination = Path(archive_dir)
    filenames = [str(spec["filename"]) for spec in VISDRONE_ARCHIVES.values()]
    return "\n".join(
        [
            "Manual VisDrone archive placement is required.",
            "Place or upload these exact files:",
            *(f"- {filename}" for filename in filenames),
            f"Destination: {destination}",
            "In Colab, upload the ZIPs or copy them from another Drive folder, then rerun.",
            "The same size, ZIP structure, CRC, SHA-256, and manifest checks will run.",
        ]
    )


def _read_archive_manifest(path: Path) -> ArchiveManifest | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.setdefault("schema_version", ARCHIVE_MANIFEST_SCHEMA_VERSION)
        return ArchiveManifest(**payload)
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        return None


def verify_archive_manifest(
    split: str,
    archive_dir: str | Path,
    manifest_dir: str | Path,
) -> dict[str, Any]:
    spec = VISDRONE_ARCHIVES[split]
    archive = Path(archive_dir) / str(spec["filename"])
    manifest_path = Path(manifest_dir) / f"{split}_archive.json"
    errors: list[str] = []
    manifest = _read_archive_manifest(manifest_path)
    try:
        validate_zip(archive, str(spec["folder"]), int(spec["minimum_bytes"]))
    except (FileNotFoundError, OSError, ValueError) as exc:
        errors.append(str(exc))
    if manifest is None:
        errors.append(f"archive manifest is missing or invalid: {manifest_path}")
    elif archive.is_file():
        if manifest.schema_version != ARCHIVE_MANIFEST_SCHEMA_VERSION:
            errors.append("archive manifest schema version is stale")
        if manifest.split != split:
            errors.append(f"archive manifest split is {manifest.split!r}, expected {split!r}")
        size = archive.stat().st_size
        if manifest.size_bytes != size:
            errors.append(f"archive size is {size}, manifest records {manifest.size_bytes}")
        checksum = sha256_file(archive)
        if manifest.sha256 != checksum:
            errors.append("archive SHA-256 differs from its manifest")
    return {
        "valid": not errors,
        "errors": errors,
        "archive": str(archive),
        "manifest": str(manifest_path),
        "sha256": manifest.sha256 if manifest else None,
        "size_bytes": manifest.size_bytes if manifest else None,
    }


def ensure_archive(
    split: str,
    archive_dir: str | Path,
    manifest_dir: str | Path,
    *,
    source_mode: str = "auto",
    redownload: bool = False,
) -> ArchiveEnsureResult:
    if split not in VISDRONE_ARCHIVES:
        raise ValueError(f"unsupported VisDrone split: {split!r}")
    if source_mode not in {"auto", "download", "drive", "manual"}:
        raise ValueError(f"unsupported archive source mode: {source_mode!r}")
    spec = VISDRONE_ARCHIVES[split]
    archive_dir = Path(archive_dir)
    manifest_dir = Path(manifest_dir)
    archive = archive_dir / str(spec["filename"])
    manifest_path = manifest_dir / f"{split}_archive.json"

    if archive.is_file() and (
        not redownload or source_mode in {"drive", "manual"}
    ):
        try:
            validate_zip(archive, str(spec["folder"]), int(spec["minimum_bytes"]))
            checksum = sha256_file(archive)
            cached = _read_archive_manifest(manifest_path)
            if (
                cached
                and cached.schema_version == ARCHIVE_MANIFEST_SCHEMA_VERSION
                and cached.split == split
                and Path(cached.archive_path) == archive
                and cached.size_bytes == archive.stat().st_size
                and cached.sha256 == checksum
            ):
                return ArchiveEnsureResult(cached, "reused")
            manifest = ArchiveManifest(
                schema_version=ARCHIVE_MANIFEST_SCHEMA_VERSION,
                split=split,
                source_url=str(spec["url"]),
                archive_path=str(archive),
                size_bytes=archive.stat().st_size,
                sha256=checksum,
                downloaded_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            write_json(manifest_path, asdict(manifest))
            return ArchiveEnsureResult(manifest, "manifest_refreshed")
        except (OSError, ValueError) as exc:
            if source_mode == "manual":
                raise ManualArchiveRequired(
                    f"{exc}\n\n{manual_archive_instructions(archive_dir)}"
                ) from exc
            if source_mode == "drive":
                raise

    if source_mode in {"drive", "manual"}:
        raise ManualArchiveRequired(manual_archive_instructions(archive_dir))

    archive_dir.mkdir(parents=True, exist_ok=True)
    try:
        if archive.exists():
            archive.unlink()
        download_resumable(str(spec["url"]), archive, redownload=redownload)
        validate_zip(archive, str(spec["folder"]), int(spec["minimum_bytes"]))
    except (OSError, RuntimeError, ValueError) as exc:
        if source_mode == "auto":
            raise ManualArchiveRequired(
                f"{exc}\n\n{manual_archive_instructions(archive_dir)}"
            ) from exc
        raise
    manifest = ArchiveManifest(
        schema_version=ARCHIVE_MANIFEST_SCHEMA_VERSION,
        split=split,
        source_url=str(spec["url"]),
        archive_path=str(archive),
        size_bytes=archive.stat().st_size,
        sha256=sha256_file(archive),
        downloaded_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_json(manifest_path, asdict(manifest))
    return ArchiveEnsureResult(manifest, "downloaded")


def _inventory(split_root: Path) -> dict[str, int | str]:
    files = sorted(
        (path for path in split_root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(split_root).as_posix(),
    )
    names = [path.relative_to(split_root).as_posix() for path in files]
    inventory_hash = hashlib.sha256(
        ("\n".join(names) + ("\n" if names else "")).encode("utf-8")
    ).hexdigest()
    images = [
        path
        for path in files
        if path.parent == split_root / "images"
        and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    annotations = [
        path
        for path in files
        if path.parent == split_root / "annotations" and path.suffix.lower() == ".txt"
    ]
    image_names = [path.relative_to(split_root / "images").as_posix() for path in images]
    annotation_names = [
        path.relative_to(split_root / "annotations").as_posix()
        for path in annotations
    ]
    names_hash = lambda values: hashlib.sha256(
        ("\n".join(values) + ("\n" if values else "")).encode("utf-8")
    ).hexdigest()
    return {
        "image_count": len(images),
        "annotation_count": len(annotations),
        "relative_filename_inventory_sha256": inventory_hash,
        "image_filename_inventory_sha256": names_hash(image_names),
        "annotation_filename_inventory_sha256": names_hash(annotation_names),
        "total_extracted_bytes": sum(path.stat().st_size for path in files),
    }


def write_extraction_manifest(
    *,
    split: str,
    archive_path: str | Path,
    split_root: str | Path,
    expected_folder: str,
    manifest_path: str | Path,
    verified_archive_sha256: str | None = None,
    verified_archive_size_bytes: int | None = None,
) -> ExtractionManifest:
    archive = Path(archive_path)
    root = Path(split_root)
    stats = _inventory(root)
    if int(stats["image_count"]) == 0 or int(stats["annotation_count"]) == 0:
        raise ValueError(f"extracted split has no images or annotations: {root}")
    manifest = ExtractionManifest(
        schema_version=EXTRACTION_MANIFEST_SCHEMA_VERSION,
        split=split,
        archive_sha256=verified_archive_sha256 or sha256_file(archive),
        archive_size_bytes=(
            verified_archive_size_bytes
            if verified_archive_size_bytes is not None
            else archive.stat().st_size
        ),
        top_level_folder=expected_folder,
        image_count=int(stats["image_count"]),
        annotation_count=int(stats["annotation_count"]),
        relative_filename_inventory_sha256=str(
            stats["relative_filename_inventory_sha256"]
        ),
        image_filename_inventory_sha256=str(
            stats["image_filename_inventory_sha256"]
        ),
        annotation_filename_inventory_sha256=str(
            stats["annotation_filename_inventory_sha256"]
        ),
        total_extracted_bytes=int(stats["total_extracted_bytes"]),
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_json(manifest_path, asdict(manifest))
    return manifest


def verify_extraction_manifest(
    *,
    split: str,
    archive_path: str | Path,
    split_root: str | Path,
    expected_folder: str,
    manifest_path: str | Path,
    verified_archive_sha256: str | None = None,
    verified_archive_size_bytes: int | None = None,
) -> dict[str, Any]:
    archive = Path(archive_path)
    root = Path(split_root)
    manifest_path = Path(manifest_path)
    errors: list[str] = []
    manifest: ExtractionManifest | None = None
    try:
        manifest = ExtractionManifest(
            **json.loads(manifest_path.read_text(encoding="utf-8"))
        )
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        errors.append(f"extraction manifest is missing or invalid: {manifest_path}")
    if not (root / "images").is_dir():
        errors.append(f"expected image directory is missing: {root / 'images'}")
    if not (root / "annotations").is_dir():
        errors.append(f"expected annotation directory is missing: {root / 'annotations'}")
    staging = root.parent / f".{expected_folder}.extracting"
    if staging.exists():
        errors.append(f"stale extraction staging directory remains: {staging}")
    observed = _inventory(root) if root.is_dir() else {}
    if manifest:
        if manifest.schema_version != EXTRACTION_MANIFEST_SCHEMA_VERSION:
            errors.append("extraction manifest schema version is stale")
        if manifest.split != split:
            errors.append(f"extraction manifest split is {manifest.split!r}")
        if manifest.top_level_folder != expected_folder:
            errors.append("extraction top-level folder differs from the manifest")
        if not archive.is_file():
            errors.append(f"source archive is missing: {archive}")
        else:
            archive_size = (
                verified_archive_size_bytes
                if verified_archive_size_bytes is not None
                else archive.stat().st_size
            )
            archive_hash = verified_archive_sha256 or sha256_file(archive)
            if archive_size != manifest.archive_size_bytes:
                errors.append("source archive size differs from the extraction manifest")
            if archive_hash != manifest.archive_sha256:
                errors.append("source archive SHA-256 differs from the extraction manifest")
        for field in (
            "image_count",
            "annotation_count",
            "relative_filename_inventory_sha256",
            "image_filename_inventory_sha256",
            "annotation_filename_inventory_sha256",
            "total_extracted_bytes",
        ):
            if observed.get(field) != getattr(manifest, field):
                errors.append(
                    f"extracted {field} is {observed.get(field)!r}, "
                    f"manifest records {getattr(manifest, field)!r}"
                )
    return {
        "valid": not errors,
        "errors": errors,
        "split": split,
        "archive": str(archive),
        "root": str(root),
        "manifest": str(manifest_path),
        "observed": observed,
        "recorded": asdict(manifest) if manifest else None,
    }


def extract_idempotent(
    archive_path: str | Path,
    raw_root: str | Path,
    expected_folder: str,
    *,
    split: str | None = None,
    manifest_path: str | Path | None = None,
    verified_archive_sha256: str | None = None,
    verified_archive_size_bytes: int | None = None,
) -> tuple[Path, str]:
    """Validate or atomically re-extract one split while retaining the ZIP."""
    archive_path = Path(archive_path)
    raw_root = Path(raw_root)
    split = split or expected_folder.rsplit("-", 1)[-1]
    manifest_path = Path(
        manifest_path or raw_root.parent / "manifests" / f"{split}_extraction.json"
    )
    destination = raw_root / expected_folder
    report = verify_extraction_manifest(
        split=split,
        archive_path=archive_path,
        split_root=destination,
        expected_folder=expected_folder,
        manifest_path=manifest_path,
        verified_archive_sha256=verified_archive_sha256,
        verified_archive_size_bytes=verified_archive_size_bytes,
    )
    if report["valid"]:
        return destination, "reused"

    raw_root.mkdir(parents=True, exist_ok=True)
    staging = raw_root / f".{expected_folder}.extracting"
    if staging.exists():
        shutil.rmtree(staging)
    if destination.exists():
        shutil.rmtree(destination)
    staging.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        staging_resolved = staging.resolve()
        for member in archive.infolist():
            destination_member = (staging / member.filename).resolve()
            if staging_resolved not in destination_member.parents and (
                destination_member != staging_resolved
            ):
                raise ValueError(f"unsafe path in ZIP archive: {member.filename}")
        archive.extractall(staging)
    extracted = staging / expected_folder
    if not (extracted / "images").is_dir() or not (extracted / "annotations").is_dir():
        raise ValueError(
            f"archive extracted without the expected images/annotations layout: {archive_path}"
        )
    extracted.replace(destination)
    shutil.rmtree(staging)
    write_extraction_manifest(
        split=split,
        archive_path=archive_path,
        split_root=destination,
        expected_folder=expected_folder,
        manifest_path=manifest_path,
        verified_archive_sha256=verified_archive_sha256,
        verified_archive_size_bytes=verified_archive_size_bytes,
    )
    verified = verify_extraction_manifest(
        split=split,
        archive_path=archive_path,
        split_root=destination,
        expected_folder=expected_folder,
        manifest_path=manifest_path,
        verified_archive_sha256=verified_archive_sha256,
        verified_archive_size_bytes=verified_archive_size_bytes,
    )
    if not verified["valid"]:
        raise RuntimeError(f"atomic extraction verification failed: {verified['errors']}")
    return destination, "extracted"


def dataset_split_ready(
    split_root: str | Path,
    *,
    archive_path: str | Path,
    manifest_path: str | Path,
    split: str,
) -> bool:
    expected_folder = Path(split_root).name
    return bool(
        verify_extraction_manifest(
            split=split,
            archive_path=archive_path,
            split_root=split_root,
            expected_folder=expected_folder,
            manifest_path=manifest_path,
        )["valid"]
    )


def ensure_visdrone_layout(raw_root: str | Path) -> None:
    root = Path(raw_root)
    dataset_root = root.parent
    missing: list[str] = []
    for split, spec in VISDRONE_ARCHIVES.items():
        split_root = root / str(spec["folder"])
        if not dataset_split_ready(
            split_root,
            archive_path=dataset_root / "archives" / str(spec["filename"]),
            manifest_path=dataset_root / "manifests" / f"{split}_extraction.json",
            split=split,
        ):
            missing.append(str(split_root))
    if missing:
        raise FileNotFoundError(
            "VisDrone2019-DET is incomplete or unverified. Invalid split folders:\n"
            + "\n".join(missing)
        )
