from __future__ import annotations

from pathlib import Path

import pytest

from src.pathing import (
    ArtifactPathResolver,
    NamespaceCollisionError,
    RunPathIdentity,
    assert_namespace_set_isolated,
    guard_write_target,
)


def _identity(track: str, mode: str, run_id: str = "run-1") -> RunPathIdentity:
    return RunPathIdentity(track, mode, "rtdetrv2_l", "baseline", 42, run_id)


def test_guard_accepts_exact_root_and_descendant_without_creating_them(
    tmp_path: Path,
) -> None:
    resolver = ArtifactPathResolver(tmp_path)
    identity = _identity("smoke", "smoke")
    root = resolver.run_path("predictions", identity)
    output = root / "predictions.json"
    assert guard_write_target(resolver, "predictions", identity, root) == root
    assert guard_write_target(resolver, "predictions", identity, output) == output
    assert not root.exists()


@pytest.mark.parametrize(
    ("identity", "wrong_identity"),
    [
        (_identity("smoke", "smoke"), _identity("controlled", "full")),
        (_identity("controlled", "full"), _identity("performance", "full")),
        (_identity("performance", "full"), _identity("performance", "sliced")),
    ],
)
def test_cross_namespace_targets_are_rejected_before_creation(
    tmp_path: Path,
    identity: RunPathIdentity,
    wrong_identity: RunPathIdentity,
) -> None:
    resolver = ArtifactPathResolver(tmp_path)
    wrong = resolver.run_path("predictions", wrong_identity)
    with pytest.raises(NamespaceCollisionError, match="outside expected namespace"):
        guard_write_target(resolver, "predictions", identity, wrong / "output.json")
    assert not wrong.exists()


def test_artifact_kind_collision_is_rejected(tmp_path: Path) -> None:
    resolver = ArtifactPathResolver(tmp_path)
    identity = _identity("controlled", "full")
    checkpoint = resolver.run_path("checkpoints", identity) / "best.pth"
    with pytest.raises(NamespaceCollisionError):
        guard_write_target(resolver, "predictions", identity, checkpoint)


def test_batch_guard_rejects_duplicate_namespace_roots(tmp_path: Path) -> None:
    resolver = ArtifactPathResolver(tmp_path)
    identity = _identity("performance", "sliced")
    with pytest.raises(NamespaceCollisionError, match="collide"):
        assert_namespace_set_isolated(
            resolver,
            [("predictions", identity), ("predictions", identity)],
        )


def test_all_valid_track_mode_roots_are_distinct(tmp_path: Path) -> None:
    resolver = ArtifactPathResolver(tmp_path)
    requests = [
        ("predictions", _identity("smoke", "smoke", "smoke")),
        ("predictions", _identity("controlled", "full", "control")),
        ("predictions", _identity("performance", "full", "perf-full")),
        ("predictions", _identity("performance", "sliced", "perf-sliced")),
        ("predictions", _identity("performance", "ensemble", "perf-ensemble")),
    ]
    paths = assert_namespace_set_isolated(resolver, requests)
    assert len(paths) == len(set(paths)) == 5
    assert not any(path.exists() for path in paths)
