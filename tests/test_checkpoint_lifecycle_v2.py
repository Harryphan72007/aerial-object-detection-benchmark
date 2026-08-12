from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

from scripts.cleanup_checkpoints import cleanup_checkpoints
from src.hpo.final_workflow import FinalExperimentWorkflow
from src.paths import ProjectPaths
from src.training.checkpointing import (
    RunRegistry,
    enforce_completed_checkpoint_policy,
    materialize_canonical_best,
    model_checkpoint_files,
    resolve_manifest_checkpoint,
    validate_checkpoint_identity,
)
from src.utils.serialization import read_json, write_json
from src.utils.serialization import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def _identity(run_dir: Path, *, seed: int = 42) -> dict[str, Any]:
    return {
        "run_id": run_dir.name,
        "model_id": "rtdetrv2_l",
        "seed": seed,
        "configuration_hash": f"config-{seed}",
        "epoch": 2,
        "selection_metric": "validation_mAP",
        "selection_metric_value": 0.4,
        "weight_variant": "raw",
    }


def _loader(path: Path) -> Mapping[str, Any]:
    value = read_json(path)
    assert isinstance(value, dict)
    return value


def _saver(value: Mapping[str, Any], path: Path) -> None:
    write_json(path, dict(value))


def _resume_payload(run_dir: Path, *, seed: int = 42) -> dict[str, Any]:
    return {
        "model_state_dict": {"current": 2},
        "best_model_state_dict": {"selected": 1},
        "optimizer_state_dict": {"step": 9},
        "scheduler_state_dict": {"last_epoch": 2},
        "scaler_state_dict": {"scale": 1024},
        "ema_state_dict": None,
        "epoch": 2,
        "optimizer_updates": 9,
        "rng_state": {"python": "state"},
        "sampler_generator_state": "sampler-state",
        "early_stopping_state": {"best_epoch": 1},
        "checkpoint_selection_state": {"best_raw_epoch": 1},
        "configuration_hash": _identity(run_dir, seed=seed)["configuration_hash"],
    }


def _v2_manifest(
    run_dir: Path, *, status: str = "completed", seed: int = 42
) -> dict[str, Any]:
    identity = _identity(run_dir, seed=seed)
    return {
        "schema_version": 2,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "model_id": "rtdetrv2_l",
        "architecture_family": "End-to-end Transformer",
        "dataset_track": "2class",
        "class_names": ["person", "vehicle"],
        "seed": seed,
        "input_resolution": 640,
        "checkpoint_best": str(run_dir / "best.pth"),
        "checkpoint_resume": str(run_dir / "last.pth"),
        "checkpoint_sha256": "pending",
        "checkpoint_selection_metric": "validation_mAP",
        "checkpoint_identity": identity,
        "weight_variant": "raw",
        "config_path": str(run_dir / "training_config.yaml"),
        "created_at": "2026-08-02T00:00:00+00:00",
        "framework": "transformers",
        "framework_version": "test",
        "pytorch_version": "test",
        "cuda_version": None,
        "gpu_name": "test",
        "total_parameters": 1,
        "trainable_parameters": 1,
        "frozen_parameters": 0,
        "best_validation_map": 0.4,
        "best_validation_aptiny": 0.2,
        "best_epoch": 2,
        "total_training_seconds": 1.0,
        "status": status,
    }


def test_active_checkpoint_is_one_atomic_rolling_filename(tmp_path: Path) -> None:
    last = tmp_path / "last.pth"
    _saver(_resume_payload(tmp_path), last)
    first = last.read_bytes()
    payload = _resume_payload(tmp_path)
    payload["epoch"] = 3
    _saver(payload, last)

    assert last.read_bytes() != first
    assert read_json(last)["epoch"] == 3
    assert [path.name for path in model_checkpoint_files(tmp_path)] == ["last.pth"]
    assert not list(tmp_path.glob("epoch_*.pth"))


def test_resume_payload_carries_every_required_resume_component(tmp_path: Path) -> None:
    payload = _resume_payload(tmp_path)
    assert {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "ema_state_dict",
        "epoch",
        "optimizer_updates",
        "rng_state",
        "sampler_generator_state",
        "early_stopping_state",
        "checkpoint_selection_state",
        "configuration_hash",
    } <= set(payload)


def test_completed_run_contains_only_loadable_identity_checked_best(
    tmp_path: Path,
) -> None:
    last = tmp_path / "last.pth"
    write_json(last, _resume_payload(tmp_path))
    best, checksum = materialize_canonical_best(
        last,
        tmp_path / "best.pth",
        _identity(tmp_path),
        loader=_loader,
        saver=_saver,
    )
    removed = enforce_completed_checkpoint_policy(tmp_path)

    assert checksum
    assert removed == ["last.pth"]
    assert [path.name for path in model_checkpoint_files(tmp_path)] == ["best.pth"]
    payload = validate_checkpoint_identity(best, _identity(tmp_path), loader=_loader)
    assert payload["model"] == {"selected": 1}


def test_failed_best_validation_keeps_only_resume_checkpoint(tmp_path: Path) -> None:
    last = tmp_path / "last.pth"
    write_json(last, _resume_payload(tmp_path))

    def corrupt_saver(_value: Mapping[str, Any], path: Path) -> None:
        write_json(path, {"not": "a checkpoint"})

    with pytest.raises(ValueError, match="checkpoint_identity"):
        materialize_canonical_best(
            last,
            tmp_path / "best.pth",
            _identity(tmp_path),
            loader=_loader,
            saver=corrupt_saver,
        )
    assert last.is_file()
    assert read_json(last)["optimizer_state_dict"] == {"step": 9}


def test_legacy_resolution_order_and_no_migration_side_effect(tmp_path: Path) -> None:
    for filename in ("best_map.pth", "best_raw.pth", "best.pt", "last.pth"):
        (tmp_path / filename).write_text(filename, encoding="utf-8")
    manifest = {"run_dir": str(tmp_path)}
    assert resolve_manifest_checkpoint(manifest, allow_legacy_aliases=True).name == (
        "best_map.pth"
    )
    (tmp_path / "best.pth").write_text("canonical", encoding="utf-8")
    assert resolve_manifest_checkpoint(manifest, allow_legacy_aliases=True).name == (
        "best.pth"
    )
    assert (tmp_path / "best_map.pth").is_file()


def test_cleanup_dry_run_is_non_destructive_and_apply_keeps_one_best(
    tmp_path: Path,
) -> None:
    paths = ProjectPaths.from_value(tmp_path / "drive")
    run_dir = paths.checkpoints / "rtdetrv2_l" / "completed-run"
    run_dir.mkdir(parents=True)
    identity = _identity(run_dir)
    write_json(
        run_dir / "best_map.pth",
        {"checkpoint_identity": identity, "model": {"weight": 1}},
    )
    write_json(run_dir / "last.pth", _resume_payload(run_dir))
    write_json(run_dir / "best_aptiny.pth", {"model": {"weight": 2}})
    write_json(run_dir / "run_manifest.json", _v2_manifest(run_dir))

    dry = cleanup_checkpoints(paths.root, loader=_loader)
    assert dry["mode"] == "dry-run"
    assert (run_dir / "best_map.pth").is_file()
    assert (run_dir / "last.pth").is_file()
    assert not (run_dir / "best.pth").exists()
    assert dry["runs"][0]["removed_files"] == []

    applied = cleanup_checkpoints(paths.root, apply=True, loader=_loader)
    assert applied["runs"][0]["errors"] == []
    assert [path.name for path in model_checkpoint_files(run_dir)] == ["best.pth"]


def test_cleanup_refuses_incomplete_and_out_of_boundary_sources(
    tmp_path: Path,
) -> None:
    paths = ProjectPaths.from_value(tmp_path / "drive")
    incomplete = paths.checkpoints / "rtdetrv2_l" / "incomplete"
    incomplete.mkdir(parents=True)
    write_json(incomplete / "last.pth", _resume_payload(incomplete))
    write_json(
        incomplete / "run_manifest.json",
        _v2_manifest(incomplete, status="failed"),
    )
    outside = tmp_path / "outside.pth"
    write_json(outside, {"model": {"weight": 1}})
    escaped = paths.checkpoints / "rtdetrv2_l" / "escaped"
    escaped.mkdir(parents=True)
    manifest = _v2_manifest(escaped)
    manifest["checkpoint_best"] = str(outside)
    write_json(escaped / "run_manifest.json", manifest)

    report = cleanup_checkpoints(paths.root, apply=True, loader=_loader)
    by_id = {entry["run_id"]: entry for entry in report["runs"]}
    assert (incomplete / "last.pth").is_file()
    assert by_id["incomplete"]["removed_files"] == []
    assert by_id["incomplete"]["warnings"]
    assert outside.is_file()
    assert by_id["escaped"]["errors"]


def test_interrupted_run_keeps_last_and_never_creates_best(tmp_path: Path) -> None:
    paths = ProjectPaths.from_value(tmp_path / "drive")
    run_dir = paths.checkpoints / "rtdetrv2_l" / "interrupted"
    run_dir.mkdir(parents=True)
    write_json(run_dir / "last.pth", _resume_payload(run_dir))
    write_json(
        run_dir / "run_manifest.json",
        _v2_manifest(run_dir, status="interrupted"),
    )
    cleanup_checkpoints(paths.root, apply=True, loader=_loader)
    assert (run_dir / "last.pth").is_file()
    assert not (run_dir / "best.pth").exists()


def test_registry_evaluation_resolves_v2_manifest_best(tmp_path: Path) -> None:
    paths = ProjectPaths.from_value(tmp_path / "drive").create()
    run_dir = paths.checkpoints / "rtdetrv2_l" / "registered"
    run_dir.mkdir(parents=True)
    identity = _identity(run_dir)
    write_json(
        run_dir / "best.pth",
        {"checkpoint_identity": identity, "model": {"weight": 1}},
    )
    manifest = _v2_manifest(run_dir)
    write_json(run_dir / "run_manifest.json", manifest)
    registry = RunRegistry(paths)
    registry.register_run(run_dir / "run_manifest.json")
    assert registry.load_checkpoint_from_registry(run_dir.name).name == "best.pth"


def test_baseline_and_tuned_runs_finalize_independently(tmp_path: Path) -> None:
    for recipe, seed in (("baseline", 17), ("tuned", 42)):
        run_dir = tmp_path / recipe
        run_dir.mkdir()
        write_json(run_dir / "last.pth", _resume_payload(run_dir, seed=seed))
        materialize_canonical_best(
            run_dir / "last.pth",
            run_dir / "best.pth",
            _identity(run_dir, seed=seed),
            loader=_loader,
            saver=_saver,
        )
        enforce_completed_checkpoint_policy(run_dir)
    assert (tmp_path / "baseline" / "best.pth").is_file()
    assert (tmp_path / "tuned" / "best.pth").is_file()


def test_completed_manifest_boundary_cleanup_is_recoverable(tmp_path: Path) -> None:
    best = tmp_path / "best.pth"
    last = tmp_path / "last.pth"
    best.write_bytes(b"canonical")
    last.write_bytes(b"resume")
    manifest = {
        "schema_version": 2,
        "checkpoint_best": str(best),
        "checkpoint_resume": str(last),
        "checkpoint_sha256": sha256_file(best),
    }
    recovered = FinalExperimentWorkflow._finish_v2_cleanup(tmp_path, manifest)
    assert recovered["checkpoint_resume"] is None
    assert model_checkpoint_files(tmp_path) == [best]


def test_backend_sources_forbid_new_aliases_and_epoch_histories() -> None:
    rtdetr = (ROOT / "scripts" / "run_rtdetr_training.py").read_text(
        encoding="utf-8"
    )
    mmdet = (ROOT / "scripts" / "run_mmdetection.py").read_text(encoding="utf-8")
    for forbidden in (
        "best_raw.pth",
        "best_map.pth",
        "best_aptiny.pth",
        "best.pt",
        "latest.pt",
    ):
        assert f'"{forbidden}"' not in rtdetr
    assert '"max_keep_ckpts": 1' in mmdet
    assert '"save_best": None' in mmdet
    assert '"filename": "last.pth"' in mmdet
    # Popping the key is not enough: MMEngine re-inserts a stock CheckpointHook
    # for every default name it cannot find, so it has to be set to None.
    assert 'cfg.default_hooks.pop("checkpoint", None)' not in mmdet
    assert "disable_default_checkpoint_hook(cfg.default_hooks)" in mmdet
    assert "class AtomicRollingCheckpointHook" in mmdet
    final_workflow = (ROOT / "src" / "hpo" / "final_workflow.py").read_text(
        encoding="utf-8"
    )
    assert "latest.pt" not in final_workflow
    assert "best.pt" not in final_workflow
    assert "epoch_*.pth" not in mmdet


def test_aptiny_is_metric_only_in_new_backend_contracts() -> None:
    for relative in (
        "scripts/run_rtdetr_training.py",
        "scripts/run_mmdetection.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "best_validation_aptiny" in source
        assert "checkpoint_best_aptiny" not in source
