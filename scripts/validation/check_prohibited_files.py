#!/usr/bin/env python
"""Reject datasets, runtime artifacts, secrets, and oversized Git files."""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Iterable


DEFAULT_MAX_BYTES = 20 * 1024 * 1024
FIXTURE_MAX_BYTES = 1024 * 1024
PROHIBITED_SUFFIXES = {
    ".7z",
    ".ckpt",
    ".db",
    ".engine",
    ".key",
    ".npy",
    ".npz",
    ".onnx",
    ".pem",
    ".pt",
    ".pth",
    ".safetensors",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".tgz",
    ".trt",
    ".weights",
    ".zip",
}
PROHIBITED_ROOTS = {
    "cache",
    "checkpoints",
    "data",
    "datasets",
    "evaluation",
    "frameworks",
    "lightning_logs",
    "local_artifacts",
    "mlruns",
    "optuna",
    "predictions",
    "reports",
    "runs",
    "tensorboard",
    "wandb",
    "weights",
}
PROHIBITED_NAMES = {
    ".env",
    "credentials.json",
    "github_token.txt",
    "service-account.json",
    "token.json",
}
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
)
ALLOWED_RESULT_SUFFIXES = {
    ".csv",
    ".html",
    ".jpeg",
    ".jpg",
    ".json",
    ".md",
    ".png",
    ".txt",
    ".yaml",
    ".yml",
}


def _is_fixture(relative: PurePosixPath) -> bool:
    return relative.parts[:2] == ("tests", "fixtures")


def _is_published_result(relative: PurePosixPath) -> bool:
    return relative.parts[:2] in {
        ("results", "bundles"),
        ("results", "manifests"),
    }


def validate_file(
    path: Path,
    relative_path: str | PurePosixPath,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> list[str]:
    relative = PurePosixPath(str(relative_path).replace("\\", "/"))
    lowered_parts = tuple(part.lower() for part in relative.parts)
    name = relative.name.lower()
    suffix = relative.suffix.lower()
    errors: list[str] = []
    fixture = _is_fixture(relative)
    published_result = _is_published_result(relative)

    if lowered_parts and lowered_parts[0] in PROHIBITED_ROOTS and not (
        lowered_parts == ("local_artifacts", ".gitkeep")
    ):
        errors.append("runtime artifact directory is prohibited")
    if name in PROHIBITED_NAMES or name.startswith("credentials"):
        errors.append("credential filename is prohibited")
    if "token" in name and suffix == ".json":
        errors.append("token filename is prohibited")
    if suffix in PROHIBITED_SUFFIXES and not fixture:
        errors.append(f"prohibited artifact type: {suffix}")
    if published_result and suffix not in ALLOWED_RESULT_SUFFIXES:
        errors.append("published result has an unapproved file type")

    size_limit = min(max_bytes, FIXTURE_MAX_BYTES) if fixture else max_bytes
    try:
        size = path.stat().st_size
    except OSError as exc:
        return [f"cannot inspect file: {exc}"]
    if size > size_limit:
        errors.append(f"file exceeds {size_limit} bytes")

    if size <= 2 * 1024 * 1024:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            text = ""
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(f"secret-like content: {pattern.pattern}")
    return errors


def _git_paths(repository_root: Path, staged: bool) -> list[str]:
    command = (
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"]
        if staged
        else ["git", "ls-files", "-z"]
    )
    output = subprocess.check_output(command, cwd=repository_root)
    return [value.decode("utf-8") for value in output.split(b"\0") if value]


def validate_repository(
    repository_root: Path,
    *,
    staged: bool = False,
    explicit_paths: Iterable[str] | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> list[str]:
    root = repository_root.resolve()
    relatives = list(explicit_paths) if explicit_paths is not None else _git_paths(root, staged)
    errors: list[str] = []
    for relative in relatives:
        path = root / relative
        if not path.is_file():
            continue
        for error in validate_file(path, relative, max_bytes=max_bytes):
            errors.append(f"{PurePosixPath(relative)}: {error}")
    return sorted(set(errors))


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--path", action="append", dest="paths")
    parser.add_argument("--max-file-size-mb", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors = validate_repository(
        args.repo_root,
        staged=args.staged,
        explicit_paths=args.paths,
        max_bytes=int(args.max_file_size_mb * 1024 * 1024),
    )
    if errors:
        raise SystemExit(
            "Prohibited repository files detected:\n"
            + "\n".join(f"- {error}" for error in errors)
        )
    scope = "staged files" if args.staged else "tracked files"
    print(f"Prohibited-file validation passed for {scope}.")


if __name__ == "__main__":
    main()
