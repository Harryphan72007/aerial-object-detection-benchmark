"""Automatic baseline and tuned final runs for the full HPO protocol."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from src.config.benchmark_tracks import load_protocol, resolve_controlled_protocol
from src.evaluation.policy import detection_policy
from src.hpo.workflow import (
    HPO_PROTOCOL_ID,
    OBJECTIVE_CONTRACT_VERSION,
    _cleanup_accelerator_memory,
    source_tree_fingerprint,
)
from src.hpo.rtdetr_v2 import RTDETR_HPO_PROTOCOL_ID
from src.models.registry import load_model_config
from src.models.rtdetrv2.optimizer import checked_in_recipe
from src.paths import ProjectPaths
from src.reproducibility import git_commit
from src.training.checkpointing import enforce_completed_checkpoint_policy, make_run_id
from src.training.lr_search import (
    assert_selection_split_held_out,
    create_lr_search_manifests,
    validate_lr_search_manifests,
)
from src.training.trainer import TrainingOrchestrator
from src.utils.serialization import read_json, read_yaml, sha256_file, write_json
from src.workflows.adapter_gate import adapter_fingerprint

# Base controlled-protocol values, read from configs/controlled/benchmark.yaml
# (the single source of truth). Per-model resolution (e.g. the RT-DETR epoch
# deviation) is applied at runtime via resolve_controlled_protocol; these
# module-level names expose the shared base for imports and previews.
_CONTROLLED_PROTOCOL = load_protocol(Path(__file__).resolve().parents[2], "controlled")
FINAL_SEEDS = tuple(int(seed) for seed in _CONTROLLED_PROTOCOL["final_seeds"])
FINAL_TRAIN_EPOCHS = int(_CONTROLLED_PROTOCOL["final_train_epochs"])
IMAGE_SIZE = int(_CONTROLLED_PROTOCOL["image_size"])
EFFECTIVE_BATCH_SIZE = int(_CONTROLLED_PROTOCOL["effective_batch_size"])


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
        self.protocol = resolve_controlled_protocol(self.repo_root, model_id)
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
        self.manifest_dir = self.paths.dataset_manifests / "hpo" / dataset_track

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
        if self.model_id != "rtdetrv2_l":
            contract_version = selected.get("objective_contract_version")
            if contract_version != OBJECTIVE_CONTRACT_VERSION:
                raise ValueError(
                    "best HPO configuration predates the repaired objective "
                    f"contract ({contract_version!r}); archive it and rerun HPO"
                )
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
        expected_effective = int(self.protocol["effective_batch_size"])
        if effective != expected_effective:
            raise ValueError(
                f"effective batch size must be {expected_effective}, got "
                f"{effective}"
            )
        final_epochs = int(self.protocol["final_train_epochs"])
        scheduler_horizon = final_epochs
        if self.model_id == "rtdetrv2_l":
            scheduler_horizon = int(
                checked_in_recipe(self.repo_root)["scheduler_horizon_epochs"]
            )
        contract = {
            "model_id": self.model_id,
            "dataset_track": self.dataset_track,
            "protocol_id": self.selected_protocol_id,
            "seed": seed,
            "image_size": int(self.protocol["image_size"]),
            "effective_batch_size": effective,
            "configuration_hash": configuration_hash(parameters),
            "scheduler_contract": {
                "epochs": final_epochs,
                "scheduler_horizon": scheduler_horizon,
            },
            "baseline_or_tuned": recipe,
            "restart_from_original_pretrained": True,
        }
        if self.model_id != "rtdetrv2_l":
            # Hash the official source annotations. The final-train and
            # model-selection manifests are deterministic functions of these plus
            # the seed, so a change to the source data changes this contract; the
            # held-out selection itself is enforced by the run-time overrides and
            # validate_lr_search_manifests, not by the resume contract.
            annotation_root = self.paths.coco(self.dataset_track) / "annotations"
            annotation_paths = {
                "train": annotation_root / "instances_train.json",
                "validation": annotation_root / "instances_val.json",
            }
            missing = [str(path) for path in annotation_paths.values() if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "final-run contract cannot hash missing annotations: "
                    + ", ".join(missing)
                )
            contract.update(
                {
                    "objective_contract_version": OBJECTIVE_CONTRACT_VERSION,
                    "source_commit": git_commit(self.repo_root),
                    "source_tree_fingerprint": source_tree_fingerprint(
                        self.repo_root
                    ),
                    "environment_fingerprint": adapter_fingerprint(
                        self.model_id, self.repo_root
                    ),
                    "dataset_hashes": {
                        name: sha256_file(path)
                        for name, path in annotation_paths.items()
                    },
                    "best_config_sha256": sha256_file(self.best_config_path),
                    "evaluation_policy": detection_policy(),
                }
            )
        return contract

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
            self.model_id,
            self.dataset_track,
            int(self.protocol["image_size"]),
            int(contract["seed"]),
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
            "seeds": tuple(self.protocol["final_seeds"]),
            "final_train_split": "final_train_seed42.json",
            "model_selection_split": "model_selection_seed42.json",
            "full_official_train": False,
            "official_validation_used_for_tuning": False,
            "official_validation_used_for_selection": False,
        }

    @staticmethod
    def _finish_v2_cleanup(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
        """Recover a v2 run interrupted after its completed-manifest write."""
        if int(manifest.get("schema_version", 1)) < 2:
            return manifest
        best = Path(str(manifest.get("checkpoint_best", "")))
        if not best.is_file() or sha256_file(best) != manifest.get("checkpoint_sha256"):
            raise RuntimeError("completed v2 run has an invalid canonical checkpoint")
        removed = enforce_completed_checkpoint_policy(run_dir)
        manifest["checkpoint_resume"] = None
        manifest["checkpoint_cleanup"] = {
            "policy": "single_best_v2",
            "removed": removed,
        }
        write_json(run_dir / "run_manifest.json", manifest)
        return manifest

    def _ensure_selection_manifests(self) -> None:
        """Guarantee the held-out model-selection split exists and is valid.

        Final training reads ``final_train_seed42.json`` (official train minus the
        holdout) and selects ``best.pth`` on ``model_selection_seed42.json``. The
        HPO search normally creates these; rebuild them from the official
        annotations if a final run starts without a completed search directory.
        """
        annotation_root = self.paths.coco(self.dataset_track) / "annotations"
        train = annotation_root / "instances_train.json"
        validation = annotation_root / "instances_val.json"
        required = [
            self.manifest_dir / name
            for name in (
                "final_train_seed42.json",
                "model_selection_seed42.json",
                "search_train_seed42.json",
                "search_validation_seed42.json",
                "official_full_train.json",
                "official_validation.json",
                "split_summary.json",
            )
        ]
        if all(path.is_file() for path in required):
            validate_lr_search_manifests(
                self.manifest_dir,
                official_train_json=train,
                official_validation_json=validation,
            )
        else:
            create_lr_search_manifests(
                train,
                validation,
                self.manifest_dir,
                dataset_track=self.dataset_track,
                seed=int(self.protocol["search_seed"]),
            )
        assert_selection_split_held_out(self.manifest_dir)

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
        self._ensure_selection_manifests()
        manifests: list[dict[str, Any]] = []
        for recipe, parameters in (("baseline", {}), ("tuned", tuned)):
            applied_parameters = dict(parameters)
            final_epochs = int(self.protocol["final_train_epochs"])
            scheduler_horizon = final_epochs
            if self.model_id == "rtdetrv2_l":
                static_recipe = checked_in_recipe(self.repo_root)
                applied_parameters = {**static_recipe, **parameters}
                scheduler_horizon = int(static_recipe["scheduler_horizon_epochs"])
            for seed in self.protocol["final_seeds"]:
                contract = self._contract(
                    seed, recipe, applied_parameters, batch_size, accumulation
                )
                run_id, run_dir, resume = self._resumable(contract)
                if resume == "completed":
                    manifests.append(
                        self._finish_v2_cleanup(
                            run_dir, read_json(run_dir / "run_manifest.json")
                        )
                    )
                    continue
                run_dir.mkdir(parents=True, exist_ok=True)
                write_json(run_dir / "resume_contract.json", contract)
                try:
                    manifest = self.orchestrator.run(
                        self.model_id,
                        dataset_track=self.dataset_track,
                        image_size=int(self.protocol["image_size"]),
                        batch_size=batch_size,
                        gradient_accumulation_steps=accumulation,
                        epochs=final_epochs,
                        seed=seed,
                        use_amp=True,
                        resume_run_id=resume,
                        overrides=applied_parameters,
                        train_annotation_override=(
                            self.manifest_dir / "final_train_seed42.json"
                        ),
                        validation_annotation_override=(
                            self.manifest_dir / "model_selection_seed42.json"
                        ),
                        train_images_override=self.paths.images("train"),
                        validation_images_override=self.paths.images("train"),
                        explicit_run_dir=run_dir,
                        explicit_run_id=run_id,
                        register_run=True,
                        scheduler_horizon=scheduler_horizon,
                        validation_interval=1,
                        # Stable pipeline identifier (recognised by export /
                        # comparison / bundle machinery). The training set is now
                        # official train minus the held-out model-selection split;
                        # see the run manifest's dataset_hashes for the exact data.
                        run_kind="final_complete_official_train",
                        protocol_id=self.selected_protocol_id,
                        baseline_or_tuned=recipe,
                    )
                finally:
                    _cleanup_accelerator_memory()
                manifests.append(manifest)
        return {**preview, "preview": False, "runs": manifests}
