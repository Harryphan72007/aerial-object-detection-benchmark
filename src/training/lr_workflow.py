"""Executable LR-only search and complete-dataset fine-tuning workflow."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.git_utils import git_commit
from src.models.registry import load_model_config
from src.paths import ProjectPaths
from src.training.checkpointing import RunRegistry, make_run_id
from src.training.lr_search import (
    PROMOTION_RUNGS,
    BaselineOptimizerAudit,
    CandidateResult,
    FixedBenchmarkSettings,
    boundary_extension_candidates,
    boundary_status,
    candidate_checkpoint_dir,
    candidate_id,
    classify_candidate_failure,
    create_lr_search_manifests,
    estimate_workload,
    export_candidate_yaml,
    export_selected_yaml,
    generate_lr_candidates,
    rank_candidates,
    resolve_baseline_optimizer,
    selection_statistics,
    validate_lr_search_manifests,
)
from src.training.trainer import TrainingOrchestrator
from src.utils.environment import collect_environment
from src.utils.serialization import read_json, read_yaml, write_json


def _read_epoch_metrics(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "epoch_metrics.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _image_count(path: Path) -> int:
    return len(read_json(path).get("images", []))


class LRControlledBenchmark:
    """Coordinate the deterministic LR search without registering search runs."""

    def __init__(self, repo_root: str | Path, drive_root: str | Path):
        self.repo_root = Path(repo_root).resolve()
        self.paths = ProjectPaths.from_value(drive_root).create()
        self.settings = FixedBenchmarkSettings()
        self.orchestrator = TrainingOrchestrator(self.repo_root, self.paths.root)

    @property
    def manifest_dir(self) -> Path:
        return self.paths.lr_search_manifests

    @property
    def repository_config_dir(self) -> Path:
        return self.repo_root / "configs" / "lr_search"

    @property
    def persistent_config_dir(self) -> Path:
        return self.paths.root / "lr_search_configs"

    def prepare_manifests(self, *, force: bool = False) -> dict[str, Any]:
        source = self.paths.coco("2class") / "annotations"
        required = [
            self.manifest_dir / "search_train_seed42.json",
            self.manifest_dir / "search_validation_seed42.json",
            self.manifest_dir / "official_full_train.json",
            self.manifest_dir / "official_validation.json",
            self.manifest_dir / "split_summary.json",
        ]
        if force or not all(path.exists() for path in required):
            return create_lr_search_manifests(
                source / "instances_train.json",
                source / "instances_val.json",
                self.manifest_dir,
                seed=self.settings.seed,
            )
        validate_lr_search_manifests(self.manifest_dir)
        return read_json(self.manifest_dir / "split_summary.json")

    def resolve_baseline(self, model_id: str) -> BaselineOptimizerAudit:
        return resolve_baseline_optimizer(model_id, self.repo_root)

    def _record_model_failure(
        self, model_id: str, status: str, error: BaseException
    ) -> None:
        destination = self.paths.lr_search_checkpoints / model_id
        write_json(
            destination / "model_failure.json",
            {
                "model_id": model_id,
                "status": status,
                "error": repr(error),
                "environment": collect_environment(),
            },
        )

    def _fixed_training_arguments(self) -> dict[str, Any]:
        return {
            "dataset_track": self.settings.dataset_track,
            "image_size": self.settings.image_size,
            "seed": self.settings.seed,
            "use_amp": self.settings.amp,
        }

    @staticmethod
    def _benchmark_overrides(learning_rate: float) -> dict[str, Any]:
        return {
            "learning_rate": learning_rate,
            "max_detections": 500,
        }

    def _run_candidate(
        self,
        model_id: str,
        learning_rate: float,
        target_epoch: int,
        *,
        batch_size: int,
        accumulation: int,
    ) -> dict[str, Any]:
        run_id = candidate_id(model_id, learning_rate, self.settings.seed)
        run_dir = candidate_checkpoint_dir(
            self.paths.root, model_id, learning_rate, self.settings.seed
        )
        existing_epochs = _read_epoch_metrics(run_dir)
        if existing_epochs and max(int(row["epoch"]) for row in existing_epochs) >= target_epoch:
            return read_json(run_dir / "run_manifest.json")
        resume = run_id if (run_dir / "last.pth").exists() else None
        return self.orchestrator.run(
            model_id,
            batch_size=batch_size,
            gradient_accumulation_steps=accumulation,
            epochs=target_epoch,
            resume_run_id=resume,
            overrides=self._benchmark_overrides(learning_rate),
            train_annotation_override=self.manifest_dir / "search_train_seed42.json",
            validation_annotation_override=(
                self.manifest_dir / "search_validation_seed42.json"
            ),
            train_images_override=self.paths.coco("2class") / "train",
            validation_images_override=self.paths.coco("2class") / "train",
            explicit_run_dir=run_dir,
            explicit_run_id=run_id,
            register_run=False,
            scheduler_horizon=self.settings.search_max_epochs,
            validation_interval=1,
            run_kind="lr_search_candidate",
            **self._fixed_training_arguments(),
        )

    def run_lr_range_test(
        self,
        model_id: str,
        baseline: BaselineOptimizerAudit,
        *,
        batch_size: int,
        accumulation: int,
        optimizer_steps: int = 300,
    ) -> dict[str, Any]:
        model_root = self.paths.lr_search_checkpoints / model_id
        run_dir = model_root / "_lr_range_test_run"
        output = model_root / "lr_range_test"
        self.orchestrator.run(
            model_id,
            batch_size=batch_size,
            gradient_accumulation_steps=accumulation,
            epochs=1,
            overrides=self._benchmark_overrides(baseline.learning_rate),
            train_annotation_override=self.manifest_dir / "search_train_seed42.json",
            validation_annotation_override=(
                self.manifest_dir / "search_validation_seed42.json"
            ),
            train_images_override=self.paths.coco("2class") / "train",
            validation_images_override=self.paths.coco("2class") / "train",
            explicit_run_dir=run_dir,
            explicit_run_id=f"{model_id}__lr_range_test__seed42",
            register_run=False,
            scheduler_horizon=self.settings.search_max_epochs,
            validation_interval=0,
            run_kind="lr_range_test_non_promotable",
            lr_range_test_steps=optimizer_steps,
            lr_range_output=output,
            **self._fixed_training_arguments(),
        )
        return read_json(output / "summary.json")

    def calibrate_runtime(
        self,
        model_id: str,
        baseline: BaselineOptimizerAudit,
        *,
        batch_size: int,
        accumulation: int,
    ) -> dict[str, float]:
        output = self.paths.lr_search_checkpoints / model_id / "calibration.json"
        if output.exists():
            return read_json(output)
        run_dir = self.paths.lr_search_checkpoints / model_id / "_runtime_calibration"
        self.orchestrator.run(
            model_id,
            batch_size=batch_size,
            gradient_accumulation_steps=accumulation,
            epochs=1,
            overrides=self._benchmark_overrides(baseline.learning_rate),
            train_annotation_override=self.manifest_dir / "search_train_seed42.json",
            validation_annotation_override=(
                self.manifest_dir / "search_validation_seed42.json"
            ),
            train_images_override=self.paths.coco("2class") / "train",
            validation_images_override=self.paths.coco("2class") / "train",
            explicit_run_dir=run_dir,
            explicit_run_id=f"{model_id}__runtime_calibration__seed42",
            register_run=False,
            scheduler_horizon=self.settings.search_max_epochs,
            validation_interval=1,
            run_kind="runtime_calibration_non_promotable",
            **self._fixed_training_arguments(),
        )
        rows = _read_epoch_metrics(run_dir)
        if not rows:
            raise RuntimeError("calibration did not produce epoch timing metrics")
        row = rows[-1]
        validation_seconds = float(row.get("validation_seconds", 0.0))
        search_train_seconds = max(
            0.0, float(row["epoch_seconds"]) - validation_seconds
        )
        search_train_images = _image_count(
            self.manifest_dir / "search_train_seed42.json"
        )
        search_validation_images = _image_count(
            self.manifest_dir / "search_validation_seed42.json"
        )
        full_train_images = _image_count(self.manifest_dir / "official_full_train.json")
        official_validation_images = _image_count(
            self.manifest_dir / "official_validation.json"
        )
        calibration = {
            "search_train_epoch_seconds": search_train_seconds,
            "search_validation_epoch_seconds": validation_seconds,
            "final_train_epoch_seconds": (
                search_train_seconds * full_train_images / search_train_images
            ),
            "final_validation_epoch_seconds": (
                validation_seconds
                * official_validation_images
                / search_validation_images
            ),
        }
        write_json(output, calibration)
        return calibration

    def workload_estimate(
        self,
        calibration: dict[str, float],
        *,
        range_optimizer_steps: int = 0,
        batch_size: int = 2,
        accumulation: int = 4,
    ) -> dict[str, float]:
        estimate = estimate_workload(**calibration)
        range_seconds = 0.0
        if range_optimizer_steps:
            image_count = _image_count(
                self.manifest_dir / "search_train_seed42.json"
            )
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            batches = math.ceil(image_count / (batch_size * world_size))
            optimizer_steps_per_epoch = max(1, math.ceil(batches / accumulation))
            seconds_per_optimizer_step = (
                calibration["search_train_epoch_seconds"]
                / optimizer_steps_per_epoch
            )
            range_seconds = range_optimizer_steps * seconds_per_optimizer_step
            estimate["search_seconds"] += range_seconds
            estimate["total_seconds"] += range_seconds
            estimate["total_hours"] = estimate["total_seconds"] / 3600.0
        estimate["lr_range_test_seconds"] = range_seconds
        return estimate

    def _copy_config_artifact(self, source: Path) -> None:
        self.persistent_config_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, self.persistent_config_dir / source.name)

    def _candidate_state(
        self, model_id: str, candidates: list[float]
    ) -> tuple[Path, dict[str, Any]]:
        state_path = (
            self.paths.lr_search_checkpoints / model_id / "search_state.json"
        )
        if state_path.exists():
            state = read_json(state_path)
            stored = [float(value) for value in state["learning_rates"]]
            if stored != candidates:
                raise ValueError(
                    "stored candidate grid differs; use a fresh search directory"
                )
            return state_path, state
        state = {
            "model_id": model_id,
            "learning_rates": candidates,
            "candidates": {
                candidate_id(model_id, value): {
                    "learning_rate": value,
                    "status": "PENDING",
                }
                for value in candidates
            },
            "rung_decisions": [],
        }
        write_json(state_path, state)
        return state_path, state

    def _run_extension(
        self,
        model_id: str,
        learning_rates: list[float],
        *,
        batch_size: int,
        accumulation: int,
    ) -> list[CandidateResult]:
        for target in (2, 5, 10, 15):
            for learning_rate in learning_rates:
                self._run_candidate(
                    model_id,
                    learning_rate,
                    target,
                    batch_size=batch_size,
                    accumulation=accumulation,
                )
        return [
            CandidateResult(
                candidate_id(model_id, learning_rate),
                learning_rate,
                "COMPLETED",
                _read_epoch_metrics(
                    candidate_checkpoint_dir(
                        self.paths.root, model_id, learning_rate
                    )
                ),
            )
            for learning_rate in learning_rates
        ]

    def run_search(
        self,
        model_id: str,
        *,
        batch_size: int,
        accumulation: int,
        run_lr_range_test: bool,
        run_boundary_extension: bool,
        allow_over_budget_run: bool,
    ) -> dict[str, Any]:
        self.prepare_manifests()
        effective = batch_size * accumulation * int(os.environ.get("WORLD_SIZE", "1"))
        if effective != self.settings.effective_batch_size:
            raise ValueError(
                f"effective batch size must be 8, got {batch_size} * "
                f"{accumulation} * {os.environ.get('WORLD_SIZE', '1')} = "
                f"{effective}"
            )
        try:
            baseline = self.resolve_baseline(model_id)
        except (ImportError, FileNotFoundError, RuntimeError) as error:
            self._record_model_failure(model_id, "FAILED_ENVIRONMENT", error)
            raise
        try:
            calibration = self.calibrate_runtime(
                model_id,
                baseline,
                batch_size=batch_size,
                accumulation=accumulation,
            )
        except Exception as error:
            try:
                status = classify_candidate_failure(error)
            except Exception:
                raise
            self._record_model_failure(model_id, status, error)
            raise
        workload = self.workload_estimate(
            calibration,
            range_optimizer_steps=300 if run_lr_range_test else 0,
            batch_size=batch_size,
            accumulation=accumulation,
        )
        if workload["total_hours"] > 24 and not allow_over_budget_run:
            raise RuntimeError(
                f"Estimated total is {workload['total_hours']:.2f} hours. "
                "Set ALLOW_OVER_BUDGET_RUN=True explicitly to continue."
            )
        safe_interval = None
        range_summary = None
        if run_lr_range_test:
            try:
                range_summary = self.run_lr_range_test(
                    model_id,
                    baseline,
                    batch_size=batch_size,
                    accumulation=accumulation,
                )
            except Exception as error:
                status = classify_candidate_failure(error)
                if status not in {"FAILED_OOM", "FAILED_NUMERICAL"}:
                    self._record_model_failure(model_id, status, error)
                    raise
                range_summary = {
                    "valid_safe_interval": False,
                    "status": status,
                    "failure_reason": repr(error),
                    "fallback": "baseline_centered_range",
                    "model_state_promotable": False,
                }
                write_json(
                    self.paths.lr_search_checkpoints
                    / model_id
                    / "lr_range_test"
                    / "summary.json",
                    range_summary,
                )
            if range_summary.get("valid_safe_interval"):
                safe_interval = (
                    float(range_summary["safe_lower_learning_rate"]),
                    float(range_summary["safe_upper_learning_rate"]),
                )
        candidates = generate_lr_candidates(
            baseline.learning_rate, safe_interval=safe_interval
        )
        self.repository_config_dir.mkdir(parents=True, exist_ok=True)
        candidate_yaml = (
            self.repository_config_dir / f"{model_id}_2class_candidates.yaml"
        )
        export_candidate_yaml(
            candidate_yaml,
            model_id=model_id,
            baseline=baseline,
            candidates=candidates,
            settings=self.settings,
        )
        self._copy_config_artifact(candidate_yaml)
        state_path, state = self._candidate_state(model_id, candidates)
        active_ids = [
            candidate_id(model_id, learning_rate) for learning_rate in candidates
        ]
        all_results: dict[str, CandidateResult] = {}
        for rung in PROMOTION_RUNGS:
            target = int(rung["epoch"])
            existing = next(
                (
                    decision
                    for decision in state["rung_decisions"]
                    if int(decision["epoch"]) == target
                ),
                None,
            )
            if existing:
                active_ids = list(existing["promoted_candidate_ids"])
                continue
            for identifier in active_ids:
                entry = state["candidates"][identifier]
                learning_rate = float(entry["learning_rate"])
                entry["status"] = "RUNNING"
                write_json(state_path, state)
                try:
                    self._run_candidate(
                        model_id,
                        learning_rate,
                        target,
                        batch_size=batch_size,
                        accumulation=accumulation,
                    )
                except Exception as error:
                    status = classify_candidate_failure(error)
                    entry.update({"status": status, "failure_reason": repr(error)})
                    failure_dir = candidate_checkpoint_dir(
                        self.paths.root, model_id, learning_rate
                    )
                    write_json(
                        failure_dir / "failure_environment.json",
                        collect_environment(),
                    )
                    write_json(state_path, state)
                    continue
                entry["status"] = "COMPLETED"
                write_json(state_path, state)
            results: list[CandidateResult] = []
            for identifier in active_ids:
                entry = state["candidates"][identifier]
                learning_rate = float(entry["learning_rate"])
                result = CandidateResult(
                    identifier,
                    learning_rate,
                    str(entry["status"]),
                    _read_epoch_metrics(
                        candidate_checkpoint_dir(
                            self.paths.root, model_id, learning_rate
                        )
                    ),
                    entry.get("failure_reason"),
                )
                results.append(result)
                all_results[identifier] = result
            promoted, statistics_by_id = rank_candidates(
                results, rung_epoch=target, keep=int(rung["keep"])
            )
            promoted_ids = [item.candidate_id for item in promoted]
            if not promoted_ids:
                raise RuntimeError(f"no viable candidates remain at epoch {target}")
            for identifier in active_ids:
                entry = state["candidates"][identifier]
                if str(entry["status"]).startswith("FAILED"):
                    continue
                entry["status"] = (
                    "COMPLETED"
                    if target == self.settings.search_max_epochs
                    and identifier in promoted_ids
                    else "PROMOTED"
                    if identifier in promoted_ids
                    else "ELIMINATED"
                )
            state["rung_decisions"].append(
                {
                    "epoch": target,
                    "candidate_ids_started": active_ids,
                    "promoted_candidate_ids": promoted_ids,
                    "statistics": statistics_by_id,
                }
            )
            write_json(state_path, state)
            active_ids = promoted_ids
        winner_id = active_ids[0]
        winner_entry = state["candidates"][winner_id]
        winner = CandidateResult(
            winner_id,
            float(winner_entry["learning_rate"]),
            "COMPLETED",
            _read_epoch_metrics(
                candidate_checkpoint_dir(
                    self.paths.root, model_id, float(winner_entry["learning_rate"])
                )
            ),
        )
        initial_boundary = boundary_status(winner.learning_rate, candidates)
        extension_results: list[CandidateResult] = []
        if run_boundary_extension and initial_boundary != "interior":
            extension_seconds = 30 * (
                calibration["search_train_epoch_seconds"]
                + calibration["search_validation_epoch_seconds"]
            )
            extended_total_hours = (
                workload["total_seconds"] + extension_seconds
            ) / 3600.0
            if extended_total_hours > 24 and not allow_over_budget_run:
                raise RuntimeError(
                    f"Boundary extension raises the estimate to "
                    f"{extended_total_hours:.2f} hours. Set "
                    "ALLOW_OVER_BUDGET_RUN=True explicitly to continue."
                )
            extension_lrs = boundary_extension_candidates(
                winner.learning_rate, initial_boundary
            )
            extension_results = self._run_extension(
                model_id,
                extension_lrs,
                batch_size=batch_size,
                accumulation=accumulation,
            )
            winner = rank_candidates(
                [winner, *extension_results], rung_epoch=15, keep=1
            )[0][0]
            candidates = sorted([*candidates, *extension_lrs])
        selection = selection_statistics(winner.metrics, 15)
        environment = collect_environment()
        environment["framework"] = load_model_config(
            model_id, self.repo_root
        )["framework"]
        selected_yaml = (
            self.repository_config_dir / f"{model_id}_2class_selected.yaml"
        )
        selected_payload = export_selected_yaml(
            selected_yaml,
            model_id=model_id,
            baseline=baseline,
            candidates=candidates,
            selected=winner,
            selection=selection,
            manifest_dir=self.manifest_dir,
            git_commit=git_commit(self.repo_root),
            environment=environment,
            settings=self.settings,
        )
        self._copy_config_artifact(selected_yaml)
        summary = {
            "model_id": model_id,
            "baseline": asdict(baseline),
            "workload": workload,
            "range_test": range_summary,
            "candidates": candidates,
            "state": state,
            "selected": selected_payload["search"],
            "boundary_message": (
                "The optimum may lie below the tested range."
                if selected_payload["search"]["boundary_status"] == "lowest"
                else "The optimum may lie above the tested range."
                if selected_payload["search"]["boundary_status"] == "highest"
                else None
            ),
            "extension_candidate_ids": [
                result.candidate_id for result in extension_results
            ],
        }
        summary_path = (
            self.repository_config_dir / f"{model_id}_2class_search_summary.json"
        )
        write_json(summary_path, summary)
        self._copy_config_artifact(summary_path)
        return summary

    def run_final_training(
        self,
        model_id: str,
        selected_config: str | Path,
        *,
        batch_size: int,
        accumulation: int,
        allow_over_budget_run: bool,
        run_common_evaluation: bool = True,
    ) -> dict[str, Any]:
        self.prepare_manifests()
        selected_path = Path(selected_config)
        if not selected_path.is_absolute():
            selected_path = self.repo_root / selected_path
        selected = read_yaml(selected_path)
        if selected["experiment"]["model_id"] != model_id:
            raise ValueError("selected configuration model identity mismatch")
        final = selected["final_training"]
        if final.get("dataset") != "complete_official_train":
            raise ValueError("selected configuration does not use complete official train")
        if not bool(final.get("restart_from_pretrained")):
            raise ValueError("final training must restart from pretrained weights")
        fixed_expectations = {
            "epochs": self.settings.final_epochs,
            "seed": self.settings.seed,
            "image_size": self.settings.image_size,
            "effective_batch_size": self.settings.effective_batch_size,
        }
        for key, expected in fixed_expectations.items():
            if int(final.get(key, -1)) != expected:
                raise ValueError(
                    f"selected configuration changed fixed {key}: "
                    f"{final.get(key)!r} != {expected!r}"
                )
        source_train = (
            self.paths.coco("2class") / "annotations" / "instances_train.json"
        )
        final_ids = {
            int(image["id"])
            for image in read_json(self.manifest_dir / "official_full_train.json")[
                "images"
            ]
        }
        official_ids = {
            int(image["id"]) for image in read_json(source_train)["images"]
        }
        validation_ids = {
            int(image["id"])
            for image in read_json(self.manifest_dir / "official_validation.json")[
                "images"
            ]
        }
        assert final_ids == official_ids
        assert not (final_ids & validation_ids)
        effective = batch_size * accumulation * int(
            os.environ.get("WORLD_SIZE", "1")
        )
        if effective != self.settings.effective_batch_size:
            raise ValueError("effective batch size must remain 8")
        try:
            baseline = self.resolve_baseline(model_id)
        except (ImportError, FileNotFoundError, RuntimeError) as error:
            self._record_model_failure(model_id, "FAILED_ENVIRONMENT", error)
            raise
        try:
            calibration = self.calibrate_runtime(
                model_id,
                baseline,
                batch_size=batch_size,
                accumulation=accumulation,
            )
        except Exception as error:
            try:
                status = classify_candidate_failure(error)
            except Exception:
                raise
            self._record_model_failure(model_id, status, error)
            raise
        workload = self.workload_estimate(calibration)
        if workload["total_hours"] > 24 and not allow_over_budget_run:
            raise RuntimeError(
                f"Estimated total is {workload['total_hours']:.2f} hours. "
                "Set ALLOW_OVER_BUDGET_RUN=True explicitly to continue."
            )
        run_id = make_run_id(
            model_id,
            "2class",
            self.settings.image_size,
            self.settings.seed,
        )
        run_dir = self.paths.final_checkpoints / model_id / run_id
        manifest = self.orchestrator.run(
            model_id,
            batch_size=batch_size,
            gradient_accumulation_steps=accumulation,
            epochs=self.settings.final_epochs,
            overrides=self._benchmark_overrides(float(final["learning_rate"])),
            train_annotation_override=self.manifest_dir / "official_full_train.json",
            validation_annotation_override=(
                self.manifest_dir / "official_validation.json"
            ),
            train_images_override=self.paths.coco("2class") / "train",
            validation_images_override=self.paths.coco("2class") / "val",
            explicit_run_dir=run_dir,
            explicit_run_id=run_id,
            register_run=True,
            scheduler_horizon=self.settings.final_epochs,
            validation_interval=0,
            run_kind="final_complete_official_train",
            **self._fixed_training_arguments(),
        )
        if run_common_evaluation:
            subprocess.run(
                [
                    sys.executable,
                    str(self.repo_root / "scripts" / "evaluate.py"),
                    "--drive-root",
                    str(self.paths.root),
                    "--dataset-track",
                    "2class",
                    "--split",
                    "val",
                    "--run-id",
                    run_id,
                    "--resolutions",
                    str(self.settings.image_size),
                ],
                check=True,
                cwd=self.repo_root,
            )
            evaluation_path = (
                self.paths.evaluation
                / f"{run_id}__res{self.settings.image_size}__metrics.json"
            )
            evaluation = read_json(evaluation_path)
            training_summary = read_json(run_dir / "final_metrics.json")
            write_json(
                run_dir / "final_metrics.json",
                {"training": training_summary, "evaluation": evaluation},
            )
            manifest.update(
                {
                    "best_validation_map": float(evaluation["mAP"]),
                    "best_validation_aptiny": float(evaluation["APtiny"]),
                    "final_evaluation_metrics": str(evaluation_path),
                }
            )
            write_json(run_dir / "run_manifest.json", manifest)
            RunRegistry(self.paths).register_run(run_dir / "run_manifest.json")
        return manifest
