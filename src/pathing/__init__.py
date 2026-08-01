"""Versioned path resolution with legacy compatibility."""

from src.pathing.guards import (
    NamespaceCollisionError,
    assert_namespace_set_isolated,
    guard_write_target,
)
from src.pathing.layout import RunPathIdentity
from src.pathing.resolver import ArtifactPathResolver, resolve_legacy_paths

__all__ = [
    "ArtifactPathResolver",
    "NamespaceCollisionError",
    "RunPathIdentity",
    "assert_namespace_set_isolated",
    "guard_write_target",
    "resolve_legacy_paths",
]
