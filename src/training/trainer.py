"""Training orchestration and dependency-gated framework integrations."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.models.registry import load_model_config
from src.paths import ProjectPaths
from src.reproducibility import framework_versions, seed_everything
from src.drive_sync import validate_dataset, validate_drive_writable
from src.git_utils import write_git_provenance
from src.training.checkpointing import RunRegistry, initialize_run
from src.utils.environment import collect_environment
from src.utils.serialization import read_yaml, write_json, write_yaml


class TrainingOrchestrator:
    """Prepare run state, dispatch framework training, and register outputs."""

    def __init__(self, repo_root: str | Path, drive_root: str | Path):
        self.repo_root = Path(repo_root).resolve()
        candidate = ProjectPaths.from_value(drive_root)
        validate_drive_writable(candidate.root)
        self.paths = candidate.create()

    def run(
        self,
        model_id: str,
        dataset_track: str,
        image_size: int,
        batch_size: int,
        gradient_accumulation_steps: int,
        epochs: int,
        seed: int,
        use_amp: bool = True,
        resume_run_id: str | None = None,
        overrides: dict[str, Any] | None = None,
        *,
        train_annotation_override: str | Path | None = None,
        validation_annotation_override: str | Path | None = None,
        train_images_override: str | Path | None = None,
        validation_images_override: str | Path | None = None,
        explicit_run_dir: str | Path | None = None,
        explicit_run_id: str | None = None,
        register_run: bool = True,
        scheduler_horizon: int | None = None,
        validation_interval: int = 1,
        run_kind: str = "benchmark",
        lr_range_test_steps: int = 0,
        lr_range_output: str | Path | None = None,
    ) -> dict[str, Any]:
        validate_drive_writable(self.paths.root)
        validate_dataset(self.paths, dataset_track)
        seed_everything(seed)
        model_config = load_model_config(model_id, self.repo_root)
        scheduler_horizon = scheduler_horizon or epochs
        if scheduler_horizon < epochs:
            raise ValueError("scheduler_horizon must be greater than or equal to epochs")
        if validation_interval < 0:
            raise ValueError("validation_interval must be non-negative")
        if explicit_run_dir is not None:
            run_dir = Path(explicit_run_dir).expanduser().resolve()
            run_id = explicit_run_id or run_dir.name
            if resume_run_id and resume_run_id != run_id:
                raise ValueError("resume_run_id must match explicit_run_id")
            for subdirectory in ("tensorboard", "logs"):
                (run_dir / subdirectory).mkdir(parents=True, exist_ok=True)
        elif resume_run_id:
            candidates = RunRegistry(self.paths).list_available_runs(
                model_id, dataset_track, status=None
            )
            manifest = next(
                (run for run in candidates if run["run_id"] == resume_run_id),
                None,
            )
            if not manifest:
                raise KeyError(f"resume run not found: {resume_run_id}")
            run_id = resume_run_id
            run_dir = self.paths.run_dir(model_id, run_id)
        else:
            run_id, run_dir = initialize_run(
                self.paths, model_id, dataset_track, image_size, seed
            )

        dataset_root = self.paths.coco(dataset_track)
        train_annotation = Path(train_annotation_override or (
            dataset_root / "annotations" / "instances_train.json"
        )).resolve()
        validation_annotation = Path(validation_annotation_override or (
            dataset_root / "annotations" / "instances_val.json"
        )).resolve()
        train_images = Path(
            train_images_override or self.paths.images("train")
        ).resolve()
        validation_images = Path(
            validation_images_override or self.paths.images("val")
        ).resolve()
        for path in (
            train_annotation,
            validation_annotation,
            train_images,
            validation_images,
        ):
            if not path.exists():
                raise FileNotFoundError(
                    f"required converted dataset path missing: {path}"
                )

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        run_config = {
            "model_id": model_id,
            "dataset_track": dataset_track,
            "image_size": image_size,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "world_size": world_size,
            "effective_batch_size": (
                batch_size * gradient_accumulation_steps * world_size
            ),
            "epochs": epochs,
            "scheduler_horizon": scheduler_horizon,
            "validation_interval": validation_interval,
            "seed": seed,
            "use_amp": use_amp,
            "resume_run_id": resume_run_id,
            "overrides": overrides or {},
            "run_kind": run_kind,
            "register_run": register_run,
            "lr_range_test_steps": lr_range_test_steps,
            "lr_range_output": str(lr_range_output) if lr_range_output else None,
            "train_annotation": str(train_annotation),
            "validation_annotation": str(validation_annotation),
            "train_images": str(train_images),
            "validation_images": str(validation_images),
        }
        write_yaml(run_dir / "training_config.yaml", run_config)
        write_yaml(run_dir / "resolved_config.yaml", run_config)
        write_yaml(run_dir / "model_config.yaml", model_config)
        dataset_config = read_yaml(
            self.repo_root
            / "configs"
            / "common"
            / f"dataset_{dataset_track}.yaml"
        )
        write_yaml(run_dir / "dataset_config.yaml", dataset_config)
        write_json(run_dir / "environment.json", collect_environment())
        provenance = write_git_provenance(self.repo_root, run_dir)
        write_json(run_dir / "git_provenance.json", provenance)
        try:
            frozen = subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"], text=True
            )
            (run_dir / "pip_freeze.txt").write_text(frozen, encoding="utf-8")
        except Exception:
            (run_dir / "pip_freeze.txt").write_text(
                "unavailable\n", encoding="utf-8"
            )
        write_json(
            run_dir / "label_mapping.json",
            {
                "track": dataset_track,
                "class_names": dataset_config["class_names"],
            },
        )

        started = time.perf_counter()
        manifest = self._base_manifest(
            run_id, run_dir, model_config, run_config, dataset_config
        )
        manifest.update({
            "git_commit": provenance["git_commit"],
            "git_dirty": provenance["git_dirty"],
            "source_patch": provenance["source_patch"],
        })
        manifest["status"] = "running"
        write_json(run_dir / "run_manifest.json", manifest)
        try:
            if model_config["framework"] in {
                "mmdetection",
                "vmamba_mmdetection",
            }:
                summary = self._run_mmdetection(
                    run_dir,
                    model_config,
                    run_config,
                    train_annotation,
                    validation_annotation,
                    train_images,
                    validation_images,
                )
            elif model_config["framework"] == "transformers":
                summary = self._run_rtdetr(
                    run_dir,
                    model_config,
                    run_config,
                    train_annotation,
                    validation_annotation,
                    train_images,
                    validation_images,
                )
            else:
                raise RuntimeError(
                    "training integration is not enabled for "
                    f"{model_config['framework']}"
                )
            manifest.update(summary)
            manifest["status"] = "completed"
        except Exception as error:
            manifest["status"] = "failed"
            manifest["failure_reason"] = repr(error)
            write_json(run_dir / "run_manifest.json", manifest)
            raise
        finally:
            manifest["total_training_seconds"] = time.perf_counter() - started

        write_json(run_dir / "run_manifest.json", manifest)
        if register_run:
            RunRegistry(self.paths).register_run(run_dir / "run_manifest.json")
        return manifest

    def _base_manifest(
        self,
        run_id: str,
        run_dir: Path,
        model_config: dict[str, Any],
        run_config: dict[str, Any],
        dataset_config: dict[str, Any],
    ) -> dict[str, Any]:
        versions = framework_versions()
        framework = str(model_config["framework"])
        framework_version = (
            versions.get("mmdet_version")
            if "mmdetection" in framework
            else versions.get("transformers_version")
        )
        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "model_id": run_config["model_id"],
            "architecture_family": model_config["architecture_family"],
            "dataset_track": run_config["dataset_track"],
            "class_names": dataset_config["class_names"],
            "seed": run_config["seed"],
            "input_resolution": run_config["image_size"],
            "checkpoint_best_map": str(run_dir / "best_map.pth"),
            "checkpoint_best_aptiny": str(run_dir / "best_aptiny.pth"),
            "checkpoint_last": str(run_dir / "last.pth"),
            "config_path": str(run_dir / "training_config.yaml"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "framework": framework,
            "framework_version": framework_version,
            "pytorch_version": versions.get("pytorch_version"),
            "cuda_version": versions.get("cuda_version"),
            "gpu_name": versions.get("gpu_name"),
            "total_parameters": 0,
            "trainable_parameters": 0,
            "frozen_parameters": 0,
            "best_validation_map": 0.0,
            "best_validation_aptiny": 0.0,
            "best_epoch": 0,
            "total_training_seconds": 0.0,
            "status": "created",
        }

    @staticmethod
    def _resolve_upstream_config(model_config: dict[str, Any]) -> Path:
        environment_name = model_config.get("external_root_env")
        root = os.environ.get(str(environment_name)) if environment_name else None
        if not root:
            raise RuntimeError(
                f"Set {environment_name} to the official repository root before "
                f"training {model_config['model_id']}."
            )
        path = Path(root) / str(model_config["framework_config"])
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    @staticmethod
    def _validate_controlled_overrides(
        run_dir: Path, run_config: dict[str, Any]
    ) -> None:
        if run_config.get("run_kind") not in {
            "lr_search_candidate",
            "lr_range_test_non_promotable",
            "runtime_calibration_non_promotable",
            "final_complete_official_train",
        }:
            return
        report_path = run_dir / "applied_overrides.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        unsupported = report.get("unsupported", {})
        if unsupported:
            raise RuntimeError(
                f"controlled benchmark override was not applied: {unsupported}"
            )

    @staticmethod
    def _run_backend_process(
        command: list[str], cwd: Path, log_path: Path
    ) -> None:
        """Stream backend output while retaining enough context to classify failures."""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        tail: deque[str] = deque(maxlen=200)
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
                log.flush()
                tail.append(line.rstrip())
            return_code = process.wait()
        if return_code:
            raise RuntimeError(
                f"backend process exited with code {return_code}\n"
                + "\n".join(tail)
            )

    def _run_mmdetection(
        self,
        run_dir: Path,
        model_config: dict[str, Any],
        run_config: dict[str, Any],
        train_annotation: Path,
        validation_annotation: Path,
        train_images: Path,
        validation_images: Path,
    ) -> dict[str, Any]:
        upstream_config = self._resolve_upstream_config(model_config)
        command = [
            sys.executable,
            str(self.repo_root / "scripts" / "run_mmdetection.py"),
            "--base-config",
            str(upstream_config),
            "--run-dir",
            str(run_dir),
            "--train-ann",
            str(train_annotation),
            "--val-ann",
            str(validation_annotation),
            "--train-images",
            str(train_images),
            "--val-images",
            str(validation_images),
            "--classes",
            json.dumps(
                read_yaml(run_dir / "dataset_config.yaml")["class_names"]
            ),
            "--image-size",
            str(run_config["image_size"]),
            "--batch-size",
            str(run_config["batch_size"]),
            "--accumulation",
            str(run_config["gradient_accumulation_steps"]),
            "--epochs",
            str(run_config["epochs"]),
            "--scheduler-horizon",
            str(run_config["scheduler_horizon"]),
            "--validation-interval",
            str(run_config["validation_interval"]),
            "--seed",
            str(run_config["seed"]),
            "--overrides",
            json.dumps(run_config.get("overrides", {})),
        ]
        if model_config.get("strip_mask_branch"):
            command.append("--strip-mask-branch")
        if model_config.get("registration_import"):
            command.extend(
                ["--registration-import", str(model_config["registration_import"])]
            )
        weight_env = model_config.get("pretrained_weight_env")
        pretrained_weight = os.environ.get(str(weight_env)) if weight_env else None
        if pretrained_weight:
            checkpoint = Path(pretrained_weight).expanduser().resolve()
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"{weight_env} points to a missing checkpoint: {checkpoint}"
                )
            command.extend(["--pretrained-weight", str(checkpoint)])
        elif model_config.get("allow_scratch_without_pretrained"):
            command.append("--clear-pretrained")
        if run_config["use_amp"]:
            command.append("--amp")
        if run_config.get("resume_run_id"):
            command.append("--resume")
        if run_config.get("lr_range_test_steps"):
            command.extend(
                [
                    "--lr-range-test-steps",
                    str(run_config["lr_range_test_steps"]),
                    "--lr-range-output",
                    str(
                        run_config.get("lr_range_output")
                        or (run_dir / "lr_range_test")
                    ),
                ]
            )
        self._run_backend_process(
            command, self.repo_root, run_dir / "logs" / "backend.log"
        )
        self._validate_controlled_overrides(run_dir, run_config)
        if run_config.get("lr_range_test_steps"):
            return json.loads(
                (
                    Path(
                        run_config.get("lr_range_output")
                        or (run_dir / "lr_range_test")
                    )
                    / "summary.json"
                ).read_text(encoding="utf-8")
            )
        final_metrics = run_dir / "final_metrics.json"
        return (
            json.loads(final_metrics.read_text(encoding="utf-8"))
            if final_metrics.exists()
            else {}
        )

    def _run_rtdetr(
        self,
        run_dir: Path,
        model_config: dict[str, Any],
        run_config: dict[str, Any],
        train_annotation: Path,
        validation_annotation: Path,
        train_images: Path,
        validation_images: Path,
    ) -> dict[str, Any]:
        command = [
            sys.executable,
            str(self.repo_root / "scripts" / "run_rtdetr_training.py"),
            "--run-dir",
            str(run_dir),
            "--model-name",
            str(model_config["pretrained_model_name_or_path"]),
            "--train-ann",
            str(train_annotation),
            "--val-ann",
            str(validation_annotation),
            "--train-images",
            str(train_images),
            "--val-images",
            str(validation_images),
            "--epochs",
            str(run_config["epochs"]),
            "--scheduler-horizon",
            str(run_config["scheduler_horizon"]),
            "--validation-interval",
            str(run_config["validation_interval"]),
            "--batch-size",
            str(run_config["batch_size"]),
            "--accumulation",
            str(run_config["gradient_accumulation_steps"]),
            "--image-size",
            str(run_config["image_size"]),
            "--seed",
            str(run_config["seed"]),
            "--overrides",
            json.dumps(run_config.get("overrides", {})),
        ]
        if run_config["use_amp"]:
            command.append("--amp")
        if run_config.get("resume_run_id"):
            command.append("--resume")
        if run_config.get("lr_range_test_steps"):
            command.extend(
                [
                    "--lr-range-test-steps",
                    str(run_config["lr_range_test_steps"]),
                    "--lr-range-output",
                    str(
                        run_config.get("lr_range_output")
                        or (run_dir / "lr_range_test")
                    ),
                ]
            )
        self._run_backend_process(
            command, self.repo_root, run_dir / "logs" / "backend.log"
        )
        self._validate_controlled_overrides(run_dir, run_config)
        if run_config.get("lr_range_test_steps"):
            return json.loads(
                (
                    Path(
                        run_config.get("lr_range_output")
                        or (run_dir / "lr_range_test")
                    )
                    / "summary.json"
                ).read_text(encoding="utf-8")
            )
        return json.loads(
            (run_dir / "final_metrics.json").read_text(encoding="utf-8")
        )
