"""Pre-write guards for isolated artifact namespaces."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.pathing.layout import RunPathIdentity
from src.pathing.resolver import ArtifactPathResolver


class NamespaceCollisionError(ValueError):
    """Raised before a write could cross an experiment namespace boundary."""


def guard_write_target(
    resolver: ArtifactPathResolver,
    artifact: str,
    identity: RunPathIdentity,
    destination: str | Path,
) -> Path:
    """Return a safe destination or reject it before filesystem mutation."""

    expected = resolver.run_path(artifact, identity).resolve(strict=False)
    candidate = Path(destination).expanduser().resolve(strict=False)
    if candidate != expected and expected not in candidate.parents:
        raise NamespaceCollisionError(
            f"write target {candidate} is outside expected namespace {expected}"
        )
    return candidate


def assert_namespace_set_isolated(
    resolver: ArtifactPathResolver,
    requests: Iterable[tuple[str, RunPathIdentity]],
) -> tuple[Path, ...]:
    """Reject a batch whose run roots collide before any path is created."""

    paths = tuple(resolver.run_path(artifact, identity) for artifact, identity in requests)
    duplicates = {path for path in paths if paths.count(path) > 1}
    if duplicates:
        rendered = ", ".join(str(path) for path in sorted(duplicates))
        raise NamespaceCollisionError(f"artifact namespaces collide: {rendered}")
    return paths
