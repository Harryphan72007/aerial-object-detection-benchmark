"""Reproducible acquisition and verification of required pretrained weights.

A checkpoint that is only checked for "exists and is non-empty" is not
reproducible: a truncated download, a browser error page saved under the right
name, or a different architecture's weights all pass that test and then fail
much later as a silent partial load. Every required checkpoint therefore
declares its source URL, SHA-256, and size in ``configs/runtime_environments.yaml``
and is verified against them before any run starts.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from src.utils.serialization import read_yaml

RUNTIME_CONFIG = "configs/runtime_environments.yaml"
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
# Set to 0 to require a manually placed file instead of a network download.
ALLOW_DOWNLOAD_ENV = "VISDRONE_ALLOW_CHECKPOINT_DOWNLOAD"

Downloader = Callable[[str, Path], None]


class CheckpointVerificationError(RuntimeError):
    """A required checkpoint is missing, truncated, or not the pinned file."""


@dataclass(frozen=True)
class PretrainedCheckpointSpec:
    model: str
    filename: str
    source_url: str
    sha256: str
    size_bytes: int
    required: bool = True
    upstream_filename: str | None = None
    license: str | None = None
    rationale: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "filename": self.filename,
            "source_url": self.source_url,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "required": self.required,
            "upstream_filename": self.upstream_filename,
            "license": self.license,
            "rationale": self.rationale,
        }


def load_pretrained_spec(value: Mapping[str, Any]) -> PretrainedCheckpointSpec:
    """Build a spec from a validated ``pretrained:`` configuration block."""
    required_fields = ("model", "filename", "source_url", "sha256", "size_bytes")
    missing = [field for field in required_fields if not value.get(field)]
    if missing:
        raise CheckpointVerificationError(
            f"pretrained checkpoint specification is incomplete: {missing}"
        )
    digest = str(value["sha256"]).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise CheckpointVerificationError(
            f"pretrained checkpoint sha256 is not a hex digest: {value['sha256']!r}"
        )
    return PretrainedCheckpointSpec(
        model=str(value["model"]),
        filename=str(value["filename"]),
        source_url=str(value["source_url"]),
        sha256=digest,
        size_bytes=int(value["size_bytes"]),
        required=bool(value.get("required", True)),
        upstream_filename=(
            str(value["upstream_filename"]) if value.get("upstream_filename") else None
        ),
        license=str(value["license"]) if value.get("license") else None,
        rationale=str(value["rationale"]) if value.get("rationale") else None,
    )


def family_pretrained_spec(
    repo_root: str | Path, family: str
) -> PretrainedCheckpointSpec | None:
    """Return the declared checkpoint for a runtime family, if it has one."""
    config = read_yaml(Path(repo_root) / RUNTIME_CONFIG)
    block = config.get(family, {}).get("pretrained")
    return load_pretrained_spec(block) if block else None


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(path: str | Path, spec: PretrainedCheckpointSpec) -> dict[str, Any]:
    """Raise unless ``path`` is exactly the pinned checkpoint."""
    target = Path(path)
    if not target.is_file():
        raise CheckpointVerificationError(
            f"required {spec.model} checkpoint is missing: {target}"
        )
    size = target.stat().st_size
    if size != spec.size_bytes:
        raise CheckpointVerificationError(
            f"{target} is {size} bytes, expected {spec.size_bytes}; the file is "
            "truncated or is not the pinned checkpoint"
        )
    digest = file_sha256(target)
    if digest != spec.sha256:
        raise CheckpointVerificationError(
            f"{target} has SHA-256 {digest}, expected {spec.sha256}; refusing to "
            "train from an unverified checkpoint"
        )
    return {
        "path": str(target.resolve()),
        "sha256": digest,
        "size_bytes": size,
        "source_url": spec.source_url,
        "verified": True,
    }


def download_allowed(environ: Mapping[str, str] | None = None) -> bool:
    values = os.environ if environ is None else environ
    return values.get(ALLOW_DOWNLOAD_ENV, "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def _urlretrieve(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle, DOWNLOAD_CHUNK_BYTES)


def ensure_pretrained_checkpoint(
    spec: PretrainedCheckpointSpec,
    destination: str | Path,
    *,
    downloader: Downloader | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Guarantee a verified checkpoint at ``destination``.

    A valid file is reused untouched. Anything else is downloaded to a temporary
    name, verified, and only then moved into place, so an interrupted or corrupt
    download can never be mistaken for the real checkpoint.
    """
    target = Path(destination)
    if target.is_file():
        try:
            return {**verify_checkpoint(target, spec), "action": "reused"}
        except CheckpointVerificationError as error:
            if not download_allowed(environ):
                raise
            print(f"Replacing invalid {spec.model} checkpoint: {error}")
    if not download_allowed(environ):
        raise CheckpointVerificationError(
            f"required {spec.model} checkpoint is missing at {target} and "
            f"automatic download is disabled ({ALLOW_DOWNLOAD_ENV}=0). Fetch it "
            "with: python -m scripts.fetch_pretrained_checkpoints --drive-root "
            "<artifact_root>"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f"{target.name}.download-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    fetch = downloader or _urlretrieve
    print(f"Downloading {spec.model} checkpoint from {spec.source_url}")
    try:
        fetch(spec.source_url, staging)
        record = verify_checkpoint(staging, spec)
        os.replace(staging, target)
    finally:
        if staging.exists():
            staging.unlink()
    return {**record, "path": str(target.resolve()), "action": "downloaded"}
