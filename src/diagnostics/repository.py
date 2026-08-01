"""Read-only Git repository inspection with explicit non-Git states."""
from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


@dataclass(frozen=True)
class RepositoryState:
    root: str
    is_repository: bool
    commit: str | None = None
    branch: str | None = None
    detached: bool = False
    dirty: bool = False
    changed_paths: tuple[str, ...] = ()
    remote_origin: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _redact_remote(value: str | None) -> str | None:
    if not value or "://" not in value:
        return value
    parsed = urlsplit(value)
    if parsed.username is None and parsed.password is None:
        return value
    hostname = parsed.hostname or ""
    if parsed.port:
        hostname = f"{hostname}:{parsed.port}"
    return urlunsplit((parsed.scheme, hostname, parsed.path, parsed.query, parsed.fragment))


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )


def inspect_repository(root: str | Path = ".") -> RepositoryState:
    path = Path(root).expanduser().resolve()
    try:
        probe = _git(path, "rev-parse", "--is-inside-work-tree")
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        return RepositoryState(
            root=str(path), is_repository=False, error=type(exc).__name__
        )
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        return RepositoryState(root=str(path), is_repository=False)

    commit_result = _git(path, "rev-parse", "HEAD")
    branch_result = _git(path, "symbolic-ref", "--quiet", "--short", "HEAD")
    status_result = _git(
        path, "status", "--porcelain=v1", "--untracked-files=all"
    )
    remote_result = _git(path, "remote", "get-url", "origin")
    errors = [
        result.stderr.strip()
        for result in (commit_result, status_result)
        if result.returncode != 0 and result.stderr.strip()
    ]
    status_lines = tuple(
        line for line in status_result.stdout.splitlines() if line.strip()
    )
    changed_paths = tuple(
        line[3:].strip() if len(line) >= 4 else line.strip()
        for line in status_lines
    )
    branch = (
        branch_result.stdout.strip() if branch_result.returncode == 0 else None
    )
    return RepositoryState(
        root=str(path),
        is_repository=True,
        commit=commit_result.stdout.strip() if commit_result.returncode == 0 else None,
        branch=branch,
        detached=branch is None,
        dirty=bool(status_lines),
        changed_paths=changed_paths,
        remote_origin=_redact_remote(
            remote_result.stdout.strip() if remote_result.returncode == 0 else None
        ),
        error="; ".join(errors) or None,
    )
