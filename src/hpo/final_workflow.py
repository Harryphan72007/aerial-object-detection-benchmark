"""Automatic baseline and tuned final runs for the full HPO protocol."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from src.hpo.workflow import HPO_PROTOCOL_ID, _cleanup_accelerator_memory
from src.hpo.rtdetr_v2 import RTDETR_HPO_PROTOCOL_ID
from src.models.registry import load_model_config
from src.optional_outputs import run_optional_output
from src.models.rtdetrv2.optimizer import checked_in_recipe
from src.paths import ProjectPaths
from src.training.checkpointing import make_run_id, materialize_checkpoint_alias
from src.training.trainer import TrainingOrchestrator
from src.utils.serialization import read_json, read_yaml, write_json

FINAL_SEEDS = (17, 42, 3407)
FINAL_EPOCHS = 25
IMAGE_SIZE = 640
EFFECTIVE_BATCH_SIZE = 8


def configuration_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class FinalExperimentWorkflow:
    """Discover HPO output and run resumable full-train experiments."""

    def __init__(
        self,
        repo_root: str | Path,
        drive_root: str | Path,
        model_id: str,
        dataset_track: str,
        *,
        orchestrator: TrainingOrchestrator | None = None,
    ):
        if dataset_track not in {"2class", "10class"}:
            raise ValueError(f"unsupported dataset track: {dataset_track}")
        self.repo_root = Path(repo_root).resolve()
        self.paths = ProjectPaths.from_value(drive_root).create()
        self.model_id = model_id
        self.dataset_track = dataset_track
        self.protocol_id = (
            RTDETR_HPO_PROTOCOL_ID if model_id == "rtdetrv2_l" else HPO_PROTOCOL_ID
        )
        self.selected_protocol_id = self.protocol_id
        load_model_config(model_id, self.repo_root)
        self.orchestrator = orchestrator or TrainingOrchestrator(
            self.repo_root, self.paths.root
        )
        self.hpo_root = (
            self.paths.root
            / "hpo"
            / self.protocol_id
            / model_id
            / dataset_track
        )

    @property
    def best_config_path(self) -> Path:
        current = self.hpo_root / "best_config.yaml"
        legacy = (
            self.paths.root
            / "hpo"
            / HPO_PROTOCOL_ID
            / self.model_id
            / self.dataset_track
            / "best_config.yaml"
        )
        return current if current.is_file() or not legacy.is_file() else legacy

    def _load_tuned_parameters(self) -> dict[str, Any]:
        if not self.best_config_path.is_file():
            raise FileNotFoundError(
                f"Complete the matching HPO notebook first: "
                f"{self.best_config_path}"
            )
        selected = read_yaml(self.best_config_path)
        expected_protocol = (
            HPO_PROTOCOL_ID
            if HPO_PROTOCOL_ID in self.best_config_path.parts
            else self.protocol_id
        )
        expected = {
            "model_id": self.model_id,
            "dataset_track": self.dataset_track,
            "protocol_id": expected_protocol,
            "search_seed": 42,
        }
        changed = {
            key: (selected.get(key), value)
            for key, value in expected.items()
            if selected.get(key) != value
        }
        if changed:
            raise ValueError(f"incompatible best HPO configuration: {changed}")
        self.selected_protocol_id = expected_protocol
        parameters = selected.get("parameters")
        if not isinstance(parameters, dict) or not parameters:
            raise ValueError("best HPO configuration contains no parameters")
        return parameters

    def _contract(
        self,
        seed: int,
        recipe: str,
        parameters: dict[str, Any],
        batch_size: int,
        accumulation: int,
    ) -> dict[str, Any]:
        effective = (
            batch_size * accumulation * int(os.environ.get("WORLD_SIZE", "1"))
        )
        if effective != EFFECTIVE_BATCH_SIZE:
            raise ValueError(
                f"effective batch size must be {EFFECTIVE_BATCH_SIZE}, got "
                f"{effective}"
            )
        scheduler_horizon = FINAL_EPOCHS
        if self.model_id == "rtdetrv2_l":
            scheduler_horizon = int(
                checked_in_recipe(self.repo_root)["scheduler_horizon_epochs"]
            )
        return {
            "model_id": self.model_id,
            "dataset_track": self.dataset_track,
            "protocol_id": self.selected_protocol_id,
            "seed": seed,
            "image_size": IMAGE_SIZE,
            "effective_batch_size": effective,
            "configuration_hash": configuration_hash(parameters),
            "scheduler_contract": {
                "epochs": FINAL_EPOCHS,
                "scheduler_horizon": scheduler_horizon,
            },
            "baseline_or_tuned": recipe,
            "restart_from_original_pretrained": True,
        }

    def _resumable(
        self, contract: dict[str, Any]
    ) -> tuple[str, Path, str | None]:
        model_root = self.paths.final_checkpoints / self.model_id
        if model_root.is_dir():
            for contract_path in sorted(
                model_root.glob("*/resume_contract.json"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            ):
                if read_json(contract_path) != contract:
                    continue
                run_dir = contract_path.parent
                manifest_path = run_dir / "run_manifest.json"
                if manifest_path.is_file():
                    manifest = read_json(manifest_path)
                    if manifest.get("status") == "completed":
                        return str(manifest["run_id"]), run_dir, "completed"
                if (run_dir / "last.pth").is_file():
                    return run_dir.name, run_dir, run_dir.name
        run_id = make_run_id(
            self.model_id, self.dataset_track, IMAGE_SIZE, int(contract["seed"])
        )
        return run_id, model_root / run_id, None

    def inspect(self) -> dict[str, Any]:
        tuned = self._load_tuned_parameters()
        return {
            "model_id": self.model_id,
            "dataset_track": self.dataset_track,
            "protocol_id": HPO_PROTOCOL_ID,
            "best_config": str(self.best_config_path),
            "tuned_parameters": tuned,
            "recipes": ("baseline", "tuned"),
            "seeds": FINAL_SEEDS,
            "full_official_train": True,
            "official_validation_used_for_tuning": False,
        }

    @staticmethod
    def _materialize_expected_aliases(run_dir: Path) -> None:
        """Keep legacy notebook consumers compatible with canonical checkpoints."""
        aliases = (
            (run_dir / "last.pth", run_dir / "latest.pt"),
            (run_dir / "best_map.pth", run_dir / "best.pt"),
        )
        for source, destination in aliases:
            if source.is_file():
                run_optional_output(
                    f"materialize_legacy_checkpoint_alias:{destination.name}",
                    run_dir,
                    lambda source=source, destination=destination: (
                        materialize_checkpoint_alias(source, destination)
                    ),
                )

    def run(
        self,
        *,
        start_expensive_stage: bool = False,
        batch_size: int = 1,
        accumulation: int = 8,
    ) -> dict[str, Any]:
        preview = self.inspect()
        if not start_expensive_stage:
            return {
                **preview,
                "preview": True,
                "message": (
                    "Set START_FINETUNING=True after reviewing this contract."
                ),
            }
        tuned = self._load_tuned_parameters()
        manifests: list[dict[str, Any]] = []
        annotation_root = self.paths.coco(self.dataset_track) / "annotations"
        for recipe, parameters in (("baseline", {}), ("tuned", tuned)):
            applied_parameters = dict(parameters)
            scheduler_horizon = FINAL_EPOCHS
            if self.model_id == "rtdetrv2_l":
                static_recipe = checked_in_recipe(self.repo_root)
                applied_parameters = {**static_recipe, **parameters}
                scheduler_horizon = int(static_recipe["scheduler_horizon_epochs"])
            for seed in FINAL_SEEDS:
                contract = self._contract(
                    seed, recipe, applied_parameters, batch_size, accumulation
                )
                run_id, run_dir, resume = self._resumable(contract)
                if resume == "completed":
                    self._materialize_expected_aliases(run_dir)
                    manifests.append(read_json(run_dir / "run_manifest.json"))
                    continue
                run_dir.mkdir(parents=True, exist_ok=True)
                write_json(run_dir / "resume_contract.json", contract)
                try:
                    manifest = self.orchestrator.run(
                        self.model_id,
                        dataset_track=self.dataset_track,
                        image_size=IMAGE_SIZE,
                        batch_size=batch_size,
                        gradient_accumulation_steps=accumulation,
                        epochs=FINAL_EPOCHS,
                        seed=seed,
                        use_amp=True,
                        resume_run_id=resume,
                        overrides=applied_parameters,
                        train_annotation_override=(
                            annotation_root / "instances_train.json"
                        ),
                        validation_annotation_override=(
                            annotation_root / "instances_val.json"
                        ),
                        train_images_override=self.paths.images("train"),
                        validation_images_override=self.paths.images("val"),
                        explicit_run_dir=run_dir,
                        explicit_run_id=run_id,
                        register_run=True,
                        scheduler_horizon=scheduler_horizon,
                        validation_interval=1,
                        run_kind="final_complete_official_train",
                        protocol_id=self.selected_protocol_id,
                        baseline_or_tuned=recipe,
                    )
                finally:
                    _cleanup_accelerator_memory()
                self._materialize_expected_aliases(run_dir)
                manifests.append(manifest)
        return {**preview, "preview": False, "runs": manifests}
