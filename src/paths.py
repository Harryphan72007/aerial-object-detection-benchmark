"""Centralized path construction for local and Google Drive runs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.utils.serialization import read_yaml

DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/visdrone_architecture_benchmark")


def load_project_config(config_path: str | Path = "project_config.yaml") -> dict:
    """Load the repository configuration without resolving private paths."""
    return read_yaml(config_path)


def configured_drive_root(
    config_path: str | Path = "project_config.yaml",
    override: str | Path | None = None,
) -> Path:
    """Return the configured Drive root, honoring an explicit CLI override."""
    if override:
        return Path(override).expanduser().resolve()
    config = load_project_config(config_path)
    value = config.get("colab", {}).get("drive_root")
    return Path(value).expanduser().resolve() if value else DEFAULT_DRIVE_ROOT


@dataclass(frozen=True)
class ProjectPaths:
    """Build every research-artifact path from one root directory."""

    root: Path

    @classmethod
    def from_value(cls, root: str | Path | None = None) -> "ProjectPaths":
        return cls(Path(root).expanduser().resolve() if root else DEFAULT_DRIVE_ROOT)

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def visdrone(self) -> Path:
        return self.datasets / "VisDrone2019-DET"

    @property
    def archives(self) -> Path:
        return self.visdrone / "archives"

    @property
    def raw(self) -> Path:
        return self.visdrone / "raw"

    def official_split(self, split: str) -> Path:
        """Return the extracted official split without relying on Drive symlinks."""
        if split not in {"train", "val"}:
            raise ValueError(f"VisDrone split must be 'train' or 'val', got {split!r}")
        return self.raw / f"VisDrone2019-DET-{split}"

    def images(self, split: str) -> Path:
        return self.official_split(split) / "images"

    @property
    def processed(self) -> Path:
        return self.visdrone / "processed"

    @property
    def dataset_manifests(self) -> Path:
        return self.visdrone / "manifests"

    @property
    def lr_search_manifests(self) -> Path:
        return self.dataset_manifests / "lr_search"

    def coco(self, track: str) -> Path:
        self.validate_track(track)
        return self.processed / f"coco_{track}"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    def model_checkpoints(self, model_id: str) -> Path:
        return self.checkpoints / model_id

    @property
    def lr_search_checkpoints(self) -> Path:
        return self.checkpoints / "lr_search"

    @property
    def final_checkpoints(self) -> Path:
        return self.checkpoints / "final"

    def run_dir(self, model_id: str, run_id: str) -> Path:
        return self.model_checkpoints(model_id) / run_id

    @property
    def registry_dir(self) -> Path:
        return self.root / "experiment_registry"

    @property
    def checkpoint_registry(self) -> Path:
        return self.registry_dir / "checkpoint_registry.json"

    @property
    def runs_csv(self) -> Path:
        return self.registry_dir / "runs.csv"

    @property
    def predictions(self) -> Path:
        return self.root / "predictions"

    @property
    def evaluation(self) -> Path:
        return self.root / "evaluation"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def profiling(self) -> Path:
        return self.root / "profiling"

    @property
    def exports(self) -> Path:
        return self.root / "exports"

    @property
    def result_bundles(self) -> Path:
        return self.root / "result_bundles"

    @property
    def pretrained(self) -> Path:
        return self.root / "pretrained"

    def create(self) -> "ProjectPaths":
        """Create the canonical Drive layout without touching source data."""
        for path in (
            self.archives,
            self.raw,
            self.processed,
            self.dataset_manifests,
            self.lr_search_manifests,
            self.coco("2class"),
            self.coco("10class"),
            self.checkpoints,
            self.lr_search_checkpoints,
            self.final_checkpoints,
            self.registry_dir,
            self.predictions,
            self.evaluation,
            self.reports,
            self.logs,
            self.profiling,
            self.exports,
            self.result_bundles,
            self.pretrained,
            self.cache,
        ):
            path.mkdir(parents=True, exist_ok=True)
        if not self.checkpoint_registry.exists():
            self.checkpoint_registry.write_text(
                '{"schema_version": 1, "runs": {}}\n', encoding="utf-8"
            )
        if not self.runs_csv.exists():
            self.runs_csv.write_text("run_id,model_id,status\n", encoding="utf-8")
        lock = self.registry_dir / "registry.lock"
        lock.touch(exist_ok=True)
        return self

    @staticmethod
    def validate_track(track: str) -> None:
        if track not in {"2class", "10class"}:
            raise ValueError(f"dataset track must be '2class' or '10class', got {track!r}")
