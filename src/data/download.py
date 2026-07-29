"""Download, cache, verify, and extract the official-format VisDrone DET archives."""
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
from typing import Callable


# Token-free mirrors used by Ultralytics' public VisDrone recipe. The notebook
# records the resolved URL and content hash; no checksum is invented here.
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
    split: str
    source_url: str
    archive_path: str
    size_bytes: int
    sha256: str
    downloaded_at_utc: str


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
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"download failed with HTTP {exc.code}: {url}") from exc
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


def ensure_archive(
    split: str,
    archive_dir: str | Path,
    manifest_dir: str | Path,
    *,
    redownload: bool = False,
) -> ArchiveManifest:
    if split not in VISDRONE_ARCHIVES:
        raise ValueError(f"unsupported VisDrone split: {split!r}")
    spec = VISDRONE_ARCHIVES[split]
    archive = Path(archive_dir) / str(spec["filename"])
    manifest_path = Path(manifest_dir) / f"{split}_archive.json"
    if archive.exists() and not redownload:
        try:
            validate_zip(archive, str(spec["folder"]), int(spec["minimum_bytes"]))
            checksum = sha256_file(archive)
            if manifest_path.exists():
                cached = json.loads(manifest_path.read_text(encoding="utf-8"))
                if cached.get("sha256") != checksum:
                    raise ValueError(
                        f"cached archive checksum differs from manifest: {archive}"
                    )
                return ArchiveManifest(**cached)
        except (OSError, ValueError, json.JSONDecodeError):
            archive.unlink(missing_ok=True)
    download_resumable(str(spec["url"]), archive, redownload=redownload)
    validate_zip(archive, str(spec["folder"]), int(spec["minimum_bytes"]))
    manifest = ArchiveManifest(
        split=split,
        source_url=str(spec["url"]),
        archive_path=str(archive),
        size_bytes=archive.stat().st_size,
        sha256=sha256_file(archive),
        downloaded_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def extract_idempotent(
    archive_path: str | Path,
    raw_root: str | Path,
    expected_folder: str,
) -> Path:
    """Extract into a temporary folder and atomically publish the split."""
    archive_path = Path(archive_path)
    raw_root = Path(raw_root)
    destination = raw_root / expected_folder
    if dataset_split_ready(destination):
        return destination
    raw_root.mkdir(parents=True, exist_ok=True)
    staging = raw_root / f".{expected_folder}.extracting"
    if staging.exists():
        shutil.rmtree(staging)
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
    if not dataset_split_ready(extracted):
        raise ValueError(
            f"archive extracted without the expected images/annotations layout: {archive_path}"
        )
    if destination.exists():
        shutil.rmtree(destination)
    extracted.replace(destination)
    shutil.rmtree(staging)
    return destination


def dataset_split_ready(split_root: str | Path) -> bool:
    split_root = Path(split_root)
    return (
        (split_root / "images").is_dir()
        and (split_root / "annotations").is_dir()
        and any((split_root / "images").glob("*.jpg"))
        and any((split_root / "annotations").glob("*.txt"))
    )


def ensure_visdrone_layout(raw_root: str | Path) -> None:
    root = Path(raw_root)
    missing = [
        str(root / folder)
        for folder in ("VisDrone2019-DET-train", "VisDrone2019-DET-val")
        if not dataset_split_ready(root / folder)
    ]
    if missing:
        raise FileNotFoundError(
            "VisDrone2019-DET is incomplete. Missing valid split folders:\n"
            + "\n".join(missing)
        )
