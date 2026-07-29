from __future__ import annotations

import json
from pathlib import Path

from src.result_export import (
    REQUIRED_BUNDLE_DIRECTORIES,
    REQUIRED_BUNDLE_FILES,
    create_result_bundle,
    export_bundle,
    find_secret_like_content,
    sanitize_text,
    validate_bundle,
)
from src.paths import ProjectPaths
from src.training.lr_search import sha256_json
from src.utils.serialization import write_json, write_yaml
from scripts.validate_results import validate_repo_results


def _make_bundle(root: Path, track: str = "2class") -> Path:
    bundle = root / f"faster_rcnn_resnet50__{track}__20260726_120000"
    bundle.mkdir(parents=True)
    for directory in REQUIRED_BUNDLE_DIRECTORIES:
        (bundle / directory).mkdir(parents=True, exist_ok=True)
    run_id = "faster_rcnn_resnet50__2class__640__20260726_120000__seed42"
    manifest = {
        "schema_version": 2,
        "result_bundle_id": bundle.name,
        "created_at": "2026-07-26T00:00:00Z",
        "model_id": "faster_rcnn_resnet50",
        "architecture_family": "CNN",
        "dataset_track": track,
        "class_names": ["person", "vehicle"],
        "run_id": run_id,
        "seed": 42,
        "seed_status": "single-seed",
        "selected_learning_rate": 0.001,
        "checkpoint_sha256": "a" * 64,
        "annotation_sha256": "b" * 64,
        "official_full_train_verified": True,
        "evaluation_git_commit": "a1b2c3d",
        "training_git_commit": "d4e5f6a",
        "generated_files": sorted(REQUIRED_BUNDLE_FILES),
        "intentionally_excluded_files": [".pth"],
        "export_status": "created",
    }
    (bundle / "bundle_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (bundle / "README.md").write_text(
        f"Bundle `{bundle.name}`; single-seed measured result.\n", encoding="utf-8"
    )
    (bundle / "configs" / "selected_lr.yaml").write_text(
        "experiment:\n"
        "  model_id: faster_rcnn_resnet50\n"
        f"  dataset_track: {track}\n"
        "search:\n"
        "  selected_learning_rate: 0.001\n",
        encoding="utf-8",
    )
    (bundle / "configs" / "final_resolved_config.yaml").write_text(
        "model_id: faster_rcnn_resnet50\n"
        f"dataset_track: {track}\n"
        "image_size: 640\n"
        "seed: 42\n"
        "epochs: 25\n"
        "scheduler_horizon: 25\n"
        "run_kind: final_complete_official_train\n"
        "overrides:\n"
        "  learning_rate: 0.001\n",
        encoding="utf-8",
    )
    (bundle / "search" / "candidates.csv").write_text(
        f"model_id,dataset_track,candidate_id,learning_rate,status\n"
        f"faster_rcnn_resnet50,{track},candidate_01,0.001,COMPLETED\n",
        encoding="utf-8",
    )
    (bundle / "search" / "promotion_history.csv").write_text(
        f"model_id,dataset_track,candidate_id,epoch,promoted\n"
        f"faster_rcnn_resnet50,{track},candidate_01,15,True\n",
        encoding="utf-8",
    )
    (bundle / "search" / "search_summary.json").write_text(
        json.dumps(
            {
                "model_id": "faster_rcnn_resnet50",
                "dataset_track": track,
                "selected_learning_rate": 0.001,
            }
        ),
        encoding="utf-8",
    )
    result = {
        "model_id": "faster_rcnn_resnet50",
        "dataset_track": track,
        "run_id": run_id,
        "evaluations": [
            {
                "model_id": "faster_rcnn_resnet50",
                "dataset_track": track,
                "run_id": run_id,
                "mAP": 0.5,
            }
        ],
    }
    (bundle / "metrics" / "final_metrics.json").write_text(
        json.dumps(result), encoding="utf-8"
    )
    (bundle / "metrics" / "per_class_metrics.csv").write_text(
        f"model_id,dataset_track,run_id,class_name,AP\n"
        f"faster_rcnn_resnet50,{track},{run_id},person,0.5\n",
        encoding="utf-8",
    )
    (bundle / "metrics" / "profiling_summary.json").write_text(
        json.dumps(
            {
                "model_id": "faster_rcnn_resnet50",
                "dataset_track": track,
                "run_id": run_id,
                "measurements": [{
                    "mean_latency_ms": 10.0,
                    "total_parameters": 100,
                    "total_training_seconds": 60.0,
                }],
            }
        ),
        encoding="utf-8",
    )
    (bundle / "reports" / "model_report.md").write_text(
        f"# Result\n\nBundle `{bundle.name}`; single-seed measured result.\n",
        encoding="utf-8",
    )
    (bundle / "provenance" / "environment_summary.json").write_text(
        json.dumps({"gpu_name": "test-gpu"}), encoding="utf-8"
    )
    (bundle / "provenance" / "dataset_hashes.json").write_text(
        json.dumps(
            {
                "model_id": "faster_rcnn_resnet50",
                "dataset_track": track,
                "run_id": run_id,
                "official_train_sha256": "c" * 64,
                "official_validation_sha256": "d" * 64,
                "evaluation_annotation_sha256": "b" * 64,
                "official_full_train_verified": True,
                "official_validation_verified": True,
            }
        ),
        encoding="utf-8",
    )
    (bundle / "provenance" / "git_commit.txt").write_text(
        "evaluation_git_commit=a1b2c3d\ntraining_git_commit=d4e5f6a\n",
        encoding="utf-8",
    )
    return bundle


def test_bundle_rejects_mixed_track_and_secrets(tmp_path):
    bundle = _make_bundle(tmp_path)
    metrics = bundle / "metrics" / "final_metrics.json"
    metrics.write_text(
        json.dumps(
            [
                {
                    "model_id": "faster_rcnn_resnet50",
                    "dataset_track": "10class",
                    "mAP": 0.5,
                }
            ]
        ),
        encoding="utf-8",
    )
    (bundle / "reports" / "bad.md").write_text(
        "token=ghp_abcdefghijklmnopqrstuvwxyz123456 /content/drive/private",
        encoding="utf-8",
    )
    errors = validate_bundle(bundle)
    assert any("mixed dataset tracks" in error for error in errors)
    assert any("secret-like" in error for error in errors)
    assert any("private absolute path" in error for error in errors)
    assert find_secret_like_content("token=ghp_abcdefghijklmnopqrstuvwxyz123456")


def test_export_is_portable_and_dry_run_does_not_write(tmp_path, monkeypatch):
    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    bundle = _make_bundle(drive / "result_bundles")
    monkeypatch.setattr("src.result_export._verify_bundle_registry", lambda *_: [])

    preview = export_bundle(drive, bundle.name, repo, dry_run=True)
    assert preview["copied"]
    assert preview["target_branch"] == "experiment-results"
    assert not (repo / "results").exists()

    exported = export_bundle(drive, bundle.name, repo)
    assert exported["copied"]
    latest = repo / "results" / "manifests" / "latest_result_manifest.json"
    assert latest.exists()
    copied_metrics = (
        repo
        / "results"
        / "bundles"
        / bundle.name
        / "metrics"
        / "final_metrics.json"
    )
    assert "/content/drive/" not in copied_metrics.read_text(encoding="utf-8")


def test_path_sanitizer_replaces_drive_and_windows_paths():
    value = sanitize_text("/content/drive/MyDrive/private C:\\Users\\alice\\secret.txt")
    assert "/content/drive/" not in value
    assert "C:/Users/" not in value


def test_empty_results_scaffold_is_valid_but_unmanifested_artifacts_are_not(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / ".gitkeep").write_text("", encoding="utf-8")
    (results / "README.md").write_text("# Published results\n", encoding="utf-8")
    assert not validate_repo_results(results)
    (results / "unexpected.csv").write_text("metric\n0.5\n", encoding="utf-8")
    assert any(
        "latest_result_manifest.json" in error
        for error in validate_repo_results(results)
    )


def test_bundle_rejects_forbidden_and_oversized_artifacts(tmp_path):
    bundle = _make_bundle(tmp_path)
    (bundle / "model.pth").write_bytes(b"weights")
    (bundle / "reports" / "too-large.txt").write_bytes(b"x" * 2048)
    errors = validate_bundle(bundle, max_file_size_mb=0.001)
    assert any("excluded artifact present: model.pth" in error for error in errors)
    assert any("oversized file: reports/too-large.txt" in error for error in errors)


def test_create_strict_bundle_from_synthetic_measured_artifacts(tmp_path):
    paths = ProjectPaths.from_value(tmp_path / "drive")
    paths.create()
    model_id = "rtdetrv2_l"
    run_id = "rtdetrv2_l__2class__640__20260726_120000__seed42"
    run_dir = paths.final_checkpoints / model_id / run_id
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "best_map.pth"
    checkpoint.write_bytes(b"synthetic-checkpoint-bytes")
    final_config = {
        "model_id": model_id,
        "dataset_track": "2class",
        "image_size": 640,
        "seed": 42,
        "epochs": 25,
        "scheduler_horizon": 25,
        "run_kind": "final_complete_official_train",
        "overrides": {"learning_rate": 0.0001},
    }
    write_yaml(run_dir / "resolved_config.yaml", final_config)
    write_yaml(run_dir / "training_config.yaml", final_config)
    write_json(run_dir / "environment.json", {"gpu_name": "synthetic-test-gpu"})
    run = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "model_id": model_id,
        "architecture_family": "End-to-end Transformer",
        "dataset_track": "2class",
        "class_names": ["person", "vehicle"],
        "seed": 42,
        "status": "completed",
        "created_at": "2026-07-26T12:00:00Z",
        "checkpoint_best_map": str(checkpoint),
        "git_commit": "a1b2c3d",
    }
    write_json(
        paths.checkpoint_registry,
        {"schema_version": 1, "runs": {run_id: run}},
    )
    selected = paths.root / "lr_search_configs" / f"{model_id}_2class_selected.yaml"
    write_yaml(
        selected,
        {
            "experiment": {"model_id": model_id, "dataset_track": "2class"},
            "search": {
                "selected_learning_rate": 0.0001,
                "candidates": [0.00005, 0.0001],
            },
            "final_training": {"learning_rate": 0.0001},
        },
    )
    write_json(
        paths.root
        / "lr_search_configs"
        / f"{model_id}_2class_search_summary.json",
        {
            "model_id": model_id,
            "candidates": [0.00005, 0.0001],
            "state": {
                "candidates": {
                    "candidate-a": {
                        "learning_rate": 0.00005,
                        "status": "ELIMINATED",
                    },
                    "candidate-b": {
                        "learning_rate": 0.0001,
                        "status": "COMPLETED",
                    },
                },
                "rung_decisions": [
                    {
                        "epoch": 15,
                        "candidate_ids_started": ["candidate-a", "candidate-b"],
                        "promoted_candidate_ids": ["candidate-b"],
                    }
                ],
            },
        },
    )
    train = {
        "images": [{"id": 1, "file_name": "train.jpg"}],
        "annotations": [],
        "categories": [],
    }
    validation = {
        "images": [{"id": 2, "file_name": "val.jpg"}],
        "annotations": [],
        "categories": [],
    }
    official_train_source = (
        paths.coco("2class") / "annotations" / "instances_train.json"
    )
    official_validation_source = (
        paths.coco("2class") / "annotations" / "instances_val.json"
    )
    full_train_manifest = paths.lr_search_manifests / "official_full_train.json"
    validation_manifest = paths.lr_search_manifests / "official_validation.json"
    write_json(official_train_source, train)
    write_json(official_validation_source, validation)
    write_json(full_train_manifest, train)
    write_json(validation_manifest, validation)
    write_json(
        paths.lr_search_manifests / "split_summary.json",
        {
            "hashes": {
                "official_full_train.json": sha256_json(full_train_manifest),
                "official_validation.json": sha256_json(validation_manifest),
            },
            "sources": {
                "official_train": {
                    "path": str(official_train_source),
                    "sha256": sha256_json(official_train_source),
                },
                "official_validation": {
                    "path": str(official_validation_source),
                    "sha256": sha256_json(official_validation_source),
                },
            },
            "verification": {
                "official_train_validation_disjoint": True,
            },
            "statistics": {
                "official_full_train.json": {"images": 1},
                "official_validation.json": {"images": 1},
            },
        },
    )
    write_json(
        paths.evaluation / f"{run_id}__res640__metrics.json",
        {
            "run_id": run_id,
            "model_id": model_id,
            "architecture_family": "End-to-end Transformer",
            "dataset_track": "2class",
            "class_names": ["person", "vehicle"],
            "evaluation_resolution": 640,
            "mAP": 0.5,
            "AP50": 0.7,
            "APtiny": 0.3,
            "per_class": {
                "person": {"AP": 0.4},
                "vehicle": {"AP": 0.6},
            },
            "mean_latency_ms": 12.0,
            "total_parameters": 100,
            "total_training_seconds": 60.0,
        },
    )
    bundle = create_result_bundle(
        paths.root,
        "2class",
        Path.cwd(),
        "rtdetrv2_l__2class__20260726_120000",
        model_id=model_id,
        run_id=run_id,
    )
    assert not validate_bundle(bundle)
    assert (bundle / "configs" / "selected_lr.yaml").is_file()
    assert (bundle / "metrics" / "final_metrics.json").is_file()
