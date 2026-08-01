"""Central path resolver whose legacy mode is byte-for-byte path compatible."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.pathing.layout import RunPathIdentity
from src.paths import ProjectPaths
from src.utils.serialization import read_yaml


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "shared" / "paths.yaml"


def resolve_legacy_paths(root: str | Path | None = None) -> ProjectPaths:
    """Return the existing resolver unchanged for migrated notebook cells."""
    return ProjectPaths.from_value(root)


class ArtifactPathResolver:
    """Resolve new isolated namespaces without creating any directories."""

    def __init__(
        self,
        root: str | Path,
        config_path: str | Path = DEFAULT_CONFIG,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.config_path = Path(config_path).expanduser().resolve()
        self.config: dict[str, Any] = read_yaml(self.config_path)
        if self.config.get("schema_version") != 1:
            raise ValueError("unsupported path config schema")

    def run_path(self, artifact: str, identity: RunPathIdentity) -> Path:
        artifacts = self.config["namespaced"]["artifacts"]
        modes = self.config["namespaced"]["modes"]
        if artifact not in artifacts:
            raise ValueError(f"unsupported artifact namespace: {artifact}")
        return (
            self.root
            / str(artifacts[artifact])
            / identity.track
            / str(modes[identity.mode])
            / identity.model_id
            / identity.experiment
            / f"seed-{identity.seed}"
            / identity.run_id
        )

    def assert_isolated(
        self,
        first: tuple[str, RunPathIdentity],
        second: tuple[str, RunPathIdentity],
    ) -> None:
        first_path = self.run_path(*first)
        second_path = self.run_path(*second)
        if first_path == second_path:
            raise ValueError("artifact namespaces unexpectedly collide")
