from __future__ import annotations

import json
from pathlib import Path

from src.result_export import (
    REQUIRED_BUNDLE_FILES,
    export_bundle,
    find_secret_like_content,
    sanitize_text,
    validate_bundle,
)


def _make_bundle(root: Path, track: str = "2class") -> Path:
    bundle = root / f"evaluation__{track}__20260726_120000"
    bundle.mkdir(parents=True)
    for directory in ("figures", "reports", "samples"):
        (bundle / directory).mkdir()
    manifest = {
        "result_bundle_id": bundle.name,
        "dataset_track": track,
        "selected_run_ids": ["run-1"],
        "model_ids": ["faster_rcnn_resnet50"],
        "architecture_families": ["CNN"],
        "class_names": ["person", "vehicle"],
        "evaluation_date": "2026-07-26T00:00:00Z",
        "evaluation_git_commit": "abc123",
        "checkpoint_sha256": {},
        "training_git_commits": {"run-1": "abc123"},
        "annotation_sha256": None,
        "hardware_information": [],
        "framework_versions": [],
        "resolutions": [1024],
        "confidence_settings": {"threshold": 0.001},
        "iou_settings": {"default": "COCO"},
        "metric_configuration": {"evaluator": "test"},
        "generated_files": sorted(REQUIRED_BUNDLE_FILES),
        "intentionally_excluded_files": ["*.pth"],
        "failed_models": [],
        "seed_status": "single-seed",
        "export_status": "created",
    }
    (bundle / "bundle_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (bundle / "selected_runs.csv").write_text("run_id\nrun-1\n", encoding="utf-8")
    result = (
        "model_id,dataset_track,class_names,mAP,precision,recall,latency_ms\n"
        "faster_rcnn_resnet50,2class,\"['person', 'vehicle']\",0.5,0.5,0.5,10\n"
    )
    (bundle / "final_results.csv").write_text(result, encoding="utf-8")
    (bundle / "final_results.json").write_text(
        json.dumps([{
            "model_id": "faster_rcnn_resnet50",
            "dataset_track": track,
            "class_names": ["person", "vehicle"],
            "mAP": 0.5,
        }]),
        encoding="utf-8",
    )
    for name in REQUIRED_BUNDLE_FILES - {
        "bundle_manifest.json", "selected_runs.csv", "final_results.csv", "final_results.json"
    }:
        (bundle / name).write_text("model_id,mAP\nfaster_rcnn_resnet50,0.5\n", encoding="utf-8")
    return bundle


def test_bundle_rejects_mixed_track_and_secrets(tmp_path):
    bundle = _make_bundle(tmp_path)
    (bundle / "final_results.json").write_text(
        json.dumps([{"model_id": "faster_rcnn_resnet50", "dataset_track": "10class"}]),
        encoding="utf-8",
    )
    (bundle / "reports" / "bad.md").write_text(
        "2class and 10class token=ghp_abcdefghijklmnopqrstuvwxyz123456 /content/drive/private",
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
    assert not (repo / "results").exists()

    exported = export_bundle(drive, bundle.name, repo)
    assert exported["copied"]
    assert (repo / "results" / "manifests" / "latest_result_manifest.json").exists()
    assert "/content/drive/" not in (
        repo / "results" / "tables" / "final_results.json"
    ).read_text(encoding="utf-8")


def test_path_sanitizer_replaces_drive_and_windows_paths():
    value = sanitize_text("/content/drive/MyDrive/private C:\\Users\\alice\\secret.txt")
    assert "/content/drive/" not in value
    assert "C:/Users/" not in value
