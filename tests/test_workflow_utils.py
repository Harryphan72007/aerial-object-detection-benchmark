import json

import pytest

from src.colab_setup import clone_or_update_repository
from src.drive_sync import initialize_drive_directories, validate_drive_writable
from src.result_export import (
    find_secret_like_content,
    sanitize_text,
    validate_metric_value,
    validate_bundle,
)
from src.training.checkpointing import atomic_torch_save
from src.paths import configured_drive_root, load_project_config


def test_project_config_and_drive_layout(tmp_path):
    config = load_project_config("project_config.yaml")
    assert config["project"]["results_branch"] == "experiment-results"
    assert configured_drive_root("project_config.yaml", tmp_path / "override") == (tmp_path / "override").resolve()
    paths = initialize_drive_directories(tmp_path / "drive")
    assert paths.coco("2class").exists()
    assert paths.result_bundles.exists()
    validate_drive_writable(tmp_path / "drive")


def test_dirty_repository_detection(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    with pytest.raises(Exception):
        clone_or_update_repository("https://example.invalid/repo.git", repo)


def test_atomic_checkpoint_replacement(tmp_path):
    torch = pytest.importorskip("torch")
    target = tmp_path / "last.pth"
    atomic_torch_save({"epoch": 1}, target)
    atomic_torch_save({"epoch": 2}, target)
    assert torch.load(target, map_location="cpu", weights_only=False)["epoch"] == 2


def test_sanitization_and_secret_detection():
    assert "/content/drive/" not in sanitize_text("/content/drive/MyDrive/private/file.json")
    assert find_secret_like_content("token=ghp_abcdefghijklmnopqrstuvwxyz123456")
    assert validate_metric_value("mAP", 1.2)
    assert not validate_metric_value("latency_ms", 2.0)


def test_result_bundle_validation_rejects_bad_files(tmp_path):
    root = tmp_path / "evaluation__2class__20260726_120000"
    root.mkdir()
    manifest = {
        "result_bundle_id": root.name,
        "dataset_track": "2class",
        "selected_run_ids": ["run"],
        "class_names": ["person", "vehicle"],
    }
    (root / "bundle_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    errors = validate_bundle(root)
    assert any("missing required bundle file" in error for error in errors)
