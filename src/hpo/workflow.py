"""Persistent two-phase Optuna random search on official-train-only subsets."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

from src.models.registry import load_model_config
from src.paths import ProjectPaths
from src.reproducibility import git_commit
from src.training.lr_search import (
    create_lr_search_manifests,
    validate_lr_search_manifests,
)
from src.training.trainer import TrainingOrchestrator
from src.utils.serialization import read_json, write_json, write_yaml
from src.workflows.adapter_gate import adapter_fingerprint

from src.hpo.search_spaces import broad_search_space, refined_search_space

HPO_PROTOCOL_ID = "two_stage_random_hpo_v1"
SEARCH_SEED = 42
PHASE_TRIALS = 5
MAX_ATTEMPT_MULTIPLIER = 4
DIVERGENCE_MARKERS = (
    "nan",
    "non-finite",
    "not finite",
    "infinite",
    "boxes1 must be",
    "boxes2 must be",
)
OOM_MARKERS = ("cuda out of memory", "out of memory")
TrialRunner = Callable[[str, int, dict[str, Any], Path], tuple[float, float]]


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sample(trial: Any, space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    parameters = {}
    for name, definition in space.items():
        if definition["kind"] == "categorical":
            parameters[name] = trial.suggest_categorical(name, definition["choices"])
        else:
            parameters[name] = trial.suggest_float(
                name,
                float(definition["low"]),
                float(definition["high"]),
                log=bool(definition.get("log")),
            )
    return parameters


def _failure_kind(error: BaseException) -> str | None:
    message = str(error).lower()
    if any(marker in message for marker in OOM_MARKERS):
        return "out_of_memory"
    if any(marker in message for marker in DIVERGENCE_MARKERS):
        return "numerical_divergence"
    return None


def _cleanup_accelerator_memory() -> None:
    """Release parent-process references after every isolated training run."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


class TwoStageRandomHPO:
    def __init__(
        self,
        repo_root: str | Path,
        drive_root: str | Path,
        model_id: str,
        dataset_track: str,
        *,
        trial_runner: TrialRunner | None = None,
    ):
        if dataset_track not in {"2class", "10class"}:
            raise ValueError(f"unsupported dataset track: {dataset_track}")
        self.repo_root = Path(repo_root).resolve()
        self.paths = ProjectPaths.from_value(drive_root).create()
        self.model_id = model_id
        self.dataset_track = dataset_track
        self.smoke_test = os.environ.get("SMOKE_TEST", "").lower() in {
            "1",
            "true",
            "yes",
        }
        load_model_config(model_id, self.repo_root)
        full_root = (
            self.paths.root
            / "hpo"
            / HPO_PROTOCOL_ID
            / model_id
            / dataset_track
        )
        self.root = full_root / "smoke_test" if self.smoke_test else full_root
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_dir = (
            self.paths.dataset_manifests / "hpo" / dataset_track
        )
        self.trial_runner = trial_runner or self._run_training_trial

    @property
    def study_path(self) -> Path:
        filename = "study_smoke.db" if self.smoke_test else "study.db"
        return self.root / filename

    def prepare_manifests(self) -> dict[str, Any]:
        train = (
            self.paths.coco(self.dataset_track)
            / "annotations"
            / "instances_train.json"
        )
        validation = (
            self.paths.coco(self.dataset_track)
            / "annotations"
            / "instances_val.json"
        )
        required = [
            self.manifest_dir / "search_train_seed42.json",
            self.manifest_dir / "search_validation_seed42.json",
            self.manifest_dir / "official_full_train.json",
            self.manifest_dir / "official_validation.json",
            self.manifest_dir / "split_summary.json",
        ]
        if all(path.is_file() for path in required):
            validate_lr_search_manifests(
                self.manifest_dir,
                official_train_json=train,
                official_validation_json=validation,
            )
            return read_json(self.manifest_dir / "split_summary.json")
        return create_lr_search_manifests(
            train,
            validation,
            self.manifest_dir,
            dataset_track=self.dataset_track,
            seed=SEARCH_SEED,
        )

    def _metadata(
        self,
        split_summary: dict[str, Any],
        search_space: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "dataset_track": self.dataset_track,
            "protocol_id": HPO_PROTOCOL_ID,
            "search_seed": SEARCH_SEED,
            "dataset_hashes": split_summary["hashes"],
            "source_commit": git_commit(self.repo_root),
            "environment_fingerprint": adapter_fingerprint(
                self.model_id, self.repo_root
            ),
            "search_space_hash": _json_hash(search_space),
            "objective": {
                "primary": "mAP50-95",
                "secondary_tiebreak": "APtiny",
                "directions": ["maximize", "maximize"],
            },
        }

    def _run_training_trial(
        self,
        phase: str,
        trial_number: int,
        parameters: dict[str, Any],
        run_dir: Path,
    ) -> tuple[float, float]:
        orchestrator = TrainingOrchestrator(self.repo_root, self.paths.root)
        epochs = 3 if phase == "phase_a" else 5
        manifest = orchestrator.run(
            self.model_id,
            dataset_track=self.dataset_track,
            image_size=640,
            batch_size=1,
            gradient_accumulation_steps=8,
            epochs=epochs,
            seed=SEARCH_SEED,
            use_amp=True,
            overrides=parameters,
            train_annotation_override=(
                self.manifest_dir / "search_train_seed42.json"
            ),
            validation_annotation_override=(
                self.manifest_dir / "search_validation_seed42.json"
            ),
            train_images_override=self.paths.images("train"),
            validation_images_override=self.paths.images("train"),
            explicit_run_dir=run_dir,
            explicit_run_id=(
                f"{self.model_id}__{self.dataset_track}__{HPO_PROTOCOL_ID}"
                f"__{phase}__trial{trial_number:02d}"
            ),
            resume_run_id=None,
            register_run=False,
            scheduler_horizon=epochs,
            validation_interval=1,
            run_kind=f"hpo_{phase}_trial",
            protocol_id=HPO_PROTOCOL_ID,
            baseline_or_tuned="search_trial",
        )
        return (
            float(manifest.get("best_validation_map", 0.0)),
            float(manifest.get("best_validation_aptiny", 0.0)),
        )

    def _study(self, metadata: dict[str, Any]):
        import optuna

        self.study_path.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{self.study_path.as_posix()}"
        suffix = "__smoke" if self.smoke_test else ""
        study = optuna.create_study(
            study_name=(
                f"{self.model_id}__{self.dataset_track}__{HPO_PROTOCOL_ID}{suffix}"
            ),
            storage=storage,
            sampler=optuna.samplers.RandomSampler(seed=SEARCH_SEED),
            directions=["maximize", "maximize"],
            load_if_exists=True,
        )
        existing = study.user_attrs.get("metadata")
        if existing and existing != metadata:
            immutable_keys = (
                "model_id",
                "dataset_track",
                "protocol_id",
                "search_seed",
                "dataset_hashes",
                "objective",
            )
            changed = {
                key: (existing.get(key), metadata.get(key))
                for key in immutable_keys
                if existing.get(key) != metadata.get(key)
            }
            if changed:
                raise RuntimeError(
                    "Persisted HPO study is incompatible with the current "
                    f"scientific contract: {changed}"
                )
            history = list(study.user_attrs.get("metadata_history", []))
            if existing not in history:
                history.append(existing)
            study.set_user_attr("metadata_history", history)
        study.set_user_attr("metadata", metadata)
        return study

    def _run_phase(
        self,
        study: Any,
        phase: str,
        space: dict[str, dict[str, Any]],
    ) -> None:
        def completed_count() -> int:
            return sum(
                1
                for trial in study.trials
                if trial.user_attrs.get("phase") == phase
                and str(trial.state.name) == "COMPLETE"
                and trial.values
                and all(math.isfinite(float(value)) for value in trial.values)
            )

        completed_before = completed_count()
        if completed_before >= PHASE_TRIALS:
            return

        missing = PHASE_TRIALS - completed_before
        attempt_limit = max(missing + 2, missing * MAX_ATTEMPT_MULTIPLIER)
        attempts = 0

        def objective(trial: Any) -> tuple[float, float]:
            trial.set_user_attr("phase", phase)
            trial.set_user_attr("resume", False)
            parameters = _sample(trial, space)
            learning_rate = float(parameters["learning_rate"])
            trial.set_user_attr("learning_rate", learning_rate)
            trial.set_user_attr("diverged", False)
            trial.set_user_attr("out_of_memory", False)
            run_dir = self.root / "trials" / phase / f"trial_{trial.number:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            try:
                values = self.trial_runner(
                    phase, trial.number, parameters, run_dir
                )
                if not all(math.isfinite(float(value)) for value in values):
                    raise FloatingPointError(
                        f"non-finite validation objectives: {values}"
                    )
            except (RuntimeError, ValueError, FloatingPointError) as error:
                failure_kind = _failure_kind(error)
                elapsed = time.perf_counter() - started
                trial.set_user_attr("failure_type", type(error).__name__)
                trial.set_user_attr("failure_reason", str(error))
                trial.set_user_attr("training_time_sec", elapsed)
                if failure_kind == "out_of_memory":
                    trial.set_user_attr("out_of_memory", True)
                elif (
                    failure_kind == "numerical_divergence"
                    and self.model_id == "rtdetrv2_l"
                ):
                    trial.set_user_attr("diverged", True)
                    trial.set_user_attr("divergence_learning_rate", learning_rate)
                else:
                    trial.set_user_attr("trial_status", "FAILED")
                    raise
                trial.set_user_attr("trial_status", "PRUNED")
                import optuna

                raise optuna.TrialPruned(str(error)) from error
            except Exception as error:
                trial.set_user_attr("trial_status", "FAILED")
                trial.set_user_attr("failure_type", type(error).__name__)
                trial.set_user_attr("failure_reason", str(error))
                raise
            finally:
                trial.set_user_attr(
                    "training_time_sec", time.perf_counter() - started
                )
                _cleanup_accelerator_memory()
            trial.set_user_attr("trial_status", "COMPLETE")
            trial.set_user_attr("run_dir", str(run_dir))
            trial.set_user_attr("validation_map", float(values[0]))
            trial.set_user_attr("validation_aptiny", float(values[1]))
            trial.set_user_attr("metrics_path", str(run_dir / "final_metrics.json"))
            trial.set_user_attr("checkpoint_path", str(run_dir / "best_map.pth"))
            return values

        while completed_count() < PHASE_TRIALS and attempts < attempt_limit:
            study.optimize(objective, n_trials=1)
            attempts += 1
        completed_after = completed_count()
        if completed_after < PHASE_TRIALS:
            raise RuntimeError(
                f"{phase} reached its attempt limit ({attempt_limit}) with "
                f"{completed_after}/{PHASE_TRIALS} finite completed trials. "
                "Inspect the persisted PRUNED/FAIL trials before resuming."
            )

    @staticmethod
    def _strongest_phase_a(study: Any) -> list[Any]:
        valid = [
            trial
            for trial in study.trials
            if trial.user_attrs.get("phase") == "phase_a"
            and str(trial.state.name) == "COMPLETE"
            and trial.values
        ]
        return sorted(
            valid,
            key=lambda trial: (float(trial.values[0]), float(trial.values[1])),
            reverse=True,
        )[:3]

    def _export(
        self,
        study: Any,
        broad: dict[str, dict[str, Any]],
        refined: dict[str, dict[str, Any]],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        import pandas as pd

        frame = study.trials_dataframe()
        for phase in ("phase_a", "phase_b"):
            numbers = {
                trial.number
                for trial in study.trials
                if trial.user_attrs.get("phase") == phase
            }
            frame[frame["number"].isin(numbers)].to_csv(
                self.root / f"{phase}_trials.csv", index=False
            )
        valid = [
            trial
            for trial in study.trials
            if str(trial.state.name) == "COMPLETE"
            and trial.values
            and all(math.isfinite(float(value)) for value in trial.values)
        ]
        if not valid:
            raise RuntimeError("HPO completed without a valid trial")
        best = max(
            valid,
            key=lambda trial: (float(trial.values[0]), float(trial.values[1])),
        )
        spaces = {"phase_a": broad, "phase_b": refined}
        search_space_hash = _json_hash(spaces)
        best_payload = {
            **metadata,
            "search_space_hash": search_space_hash,
            "trial_number": best.number,
            "objective_values": {
                "mAP50-95": float(best.values[0]),
                "APtiny": float(best.values[1]),
            },
            "parameters": best.params,
            "configuration_hash": _json_hash(best.params),
        }
        write_json(self.root / "search_space.json", spaces)
        write_json(self.root / "best_config.json", best_payload)
        write_yaml(self.root / "best_config.yaml", best_payload)
        application_reports = []
        for trial in valid:
            report = (
                Path(str(trial.user_attrs.get("run_dir", "")))
                / "applied_overrides.json"
            )
            if report.is_file():
                application_reports.append(
                    {"trial_number": trial.number, **read_json(report)}
                )
        write_json(
            self.root / "parameter_application_report.json",
            application_reports,
        )
        summary = {
            **metadata,
            "study_db": str(self.study_path),
            "search_space_hash": search_space_hash,
            "completed_trials": len(valid),
            "failed_trials": sum(
                str(trial.state.name) == "FAIL" for trial in study.trials
            ),
            "pruned_trials": sum(
                str(trial.state.name) == "PRUNED" for trial in study.trials
            ),
            "best_config": str(self.root / "best_config.yaml"),
            "resume_metadata": {
                "total_trials": len(study.trials),
                "study_name": study.study_name,
            },
        }
        write_json(self.root / "search_summary.json", summary)
        return summary

    def run(self, *, start_expensive_stage: bool = False) -> dict[str, Any]:
        split_summary = self.prepare_manifests()
        broad = broad_search_space(self.model_id)
        metadata = self._metadata(split_summary, broad)
        preview = {
            **metadata,
            "stage": "HPO",
            "study_db": str(self.study_path),
            "phase_a_trials": PHASE_TRIALS,
            "phase_b_trials": PHASE_TRIALS,
            "search_space": broad,
        }
        if not start_expensive_stage:
            return {
                **preview,
                "preview": True,
                "message": "Set START_HPO=True after reviewing this contract.",
            }
        study = self._study(metadata)
        self._run_phase(study, "phase_a", broad)
        strongest = self._strongest_phase_a(study)
        if not strongest:
            raise RuntimeError("Phase A produced no valid trials; Phase B not started")
        # The observed RT-DETR range is a controlled contract. Phase B samples
        # the same 1e-6..5e-4 interval rather than narrowing away divergent or
        # unexplored regions after only five short trials.
        refined = (
            broad
            if self.model_id == "rtdetrv2_l"
            else refined_search_space(
                broad, [dict(trial.params) for trial in strongest]
            )
        )
        self._run_phase(study, "phase_b", refined)
        return self._export(study, broad, refined, metadata)
