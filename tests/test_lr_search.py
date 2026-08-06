import json
import math
from pathlib import Path

import pytest
import yaml

from scripts.run_mmdetection import configure_scheduler_horizon
from src.training.lr_search import (
    BaselineOptimizerAudit,
    CandidateResult,
    assert_final_training_uses_official_train,
    assert_only_learning_rate_changes,
    assert_selection_split_held_out,
    boundary_extension_candidates,
    boundary_status,
    candidate_checkpoint_dir,
    candidate_id,
    create_lr_search_manifests,
    exponential_moving_average,
    export_selected_yaml,
    generate_lr_candidates,
    rank_candidates,
    resolve_batch_policy,
    validate_lr_search_manifests,
)
from src.training.lr_range import (
    exponential_lr_schedule,
    save_lr_range_artifacts,
    should_stop_range_test,
)


def _coco(image_count: int, *, validation: bool = False) -> dict:
    images = [
        {
            "id": index + 1,
            "file_name": f"{'val' if validation else 'train'}_{index:04d}.jpg",
            "width": 100,
            "height": 100,
        }
        for index in range(image_count)
    ]
    annotations = []
    annotation_id = 1
    for index, image in enumerate(images):
        if index % 10 == 0:
            continue
        for offset, category_id in enumerate((1, 2) if index % 3 == 0 else (1 if index % 2 else 2,)):
            width = 8 if offset == 0 else 20
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image["id"],
                    "category_id": category_id,
                    "bbox": [1, 1, width, width],
                    "area": width * width,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "vehicle"}],
    }


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    train = tmp_path / "train.json"
    val = tmp_path / "val.json"
    train.write_text(json.dumps(_coco(100)), encoding="utf-8")
    val.write_text(json.dumps(_coco(20, validation=True)), encoding="utf-8")
    return train, val


def test_lr_candidates_are_deterministic_positive_unique_and_include_baseline():
    baseline = 1e-4
    first = generate_lr_candidates(baseline)
    second = generate_lr_candidates(baseline)
    assert first == second
    assert len(first) == len(set(first)) == 9
    assert all(value > 0 for value in first)
    assert first == sorted(first)
    assert baseline in first
    log_steps = [
        math.log(first[index + 1]) - math.log(first[index])
        for index in range(len(first) - 1)
    ]
    assert max(log_steps) - min(log_steps) < 1e-10
    range_test_grid = generate_lr_candidates(
        baseline, safe_interval=(baseline / 4, baseline * 4)
    )
    assert range_test_grid[4] == pytest.approx(baseline)
    with pytest.raises(ValueError):
        generate_lr_candidates(
            baseline, safe_interval=(baseline * 2, baseline * 32)
        )


def test_manifest_reproducibility_and_split_contract(tmp_path):
    train, val = _write_inputs(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    create_lr_search_manifests(train, val, first)
    create_lr_search_manifests(train, val, second)
    for filename in (
        "search_train_seed42.json",
        "search_validation_seed42.json",
        "official_full_train.json",
        "official_validation.json",
    ):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    checks = validate_lr_search_manifests(first)
    assert all(checks.values())
    search_train = json.loads((first / "search_train_seed42.json").read_text())
    search_validation = json.loads(
        (first / "search_validation_seed42.json").read_text()
    )
    assert len(search_train["images"]) == 20
    assert len(search_validation["images"]) == 5
    assert_final_training_uses_official_train(first)


def test_model_selection_split_is_held_out_from_search_and_official_val(tmp_path):
    """PR-05: best.pth is selected on a held-out split, never on official val."""
    train, val = _write_inputs(tmp_path)
    out = tmp_path / "manifests"
    create_lr_search_manifests(train, val, out)
    checks = validate_lr_search_manifests(out)
    for key in (
        "model_selection_subset_official_train",
        "model_selection_disjoint_search_train",
        "model_selection_disjoint_search_validation",
        "model_selection_disjoint_final_train",
        "final_train_union_model_selection_equals_official_train",
        "model_selection_filenames_disjoint_official_validation",
        "final_train_filenames_disjoint_official_validation",
    ):
        assert checks[key], key

    ids = lambda name: {
        int(image["id"])
        for image in json.loads((out / name).read_text())["images"]
    }
    search_train = ids("search_train_seed42.json")
    search_validation = ids("search_validation_seed42.json")
    selection = ids("model_selection_seed42.json")
    final_train = ids("final_train_seed42.json")
    official_train = ids("official_full_train.json")

    # Zero image-ID overlap between search subsets and the selection split.
    assert selection.isdisjoint(search_train)
    assert selection.isdisjoint(search_validation)
    # Final-train and selection partition the complete official train exactly.
    assert final_train.isdisjoint(selection)
    assert final_train | selection == official_train
    assert len(selection) == round(len(official_train) * 0.05)

    assert_selection_split_held_out(out)


def test_only_learning_rate_changes_and_effective_batch_is_fixed():
    base = {"image_size": 640, "batch": 2, "accumulation": 4, "learning_rate": 1e-4}
    candidates = [{**base, "learning_rate": value} for value in (1e-5, 1e-4, 1e-3)]
    assert_only_learning_rate_changes(candidates)
    resolve_batch_policy(2, 4, 1, 8)
    with pytest.raises(ValueError):
        resolve_batch_policy(2, 8, 1, 8)
    broken = [dict(candidates[0]), {**candidates[1], "image_size": 1024}]
    with pytest.raises(AssertionError):
        assert_only_learning_rate_changes(broken)


def _metrics(last_epoch: int, base_map: float, variance: float = 0.0):
    return [
        {
            "epoch": epoch,
            "mAP": base_map + (variance if epoch % 2 else -variance),
            "APtiny": base_map / 2,
            "validation_loss": 1.0 / epoch,
            "gradient_norm": 1.0,
            "training_loss": 2.0 / epoch,
        }
        for epoch in range(1, last_epoch + 1)
    ]


def test_promotion_ranking_uses_window_means_and_variance():
    varying = _metrics(15, 0.40, 0.0)
    for epoch, value in zip((13, 14, 15), (0.38, 0.40, 0.42), strict=True):
        varying[epoch - 1]["mAP"] = value
    candidates = [
        CandidateResult("a", 1e-5, "RUNNING", varying),
        CandidateResult("b", 1e-4, "RUNNING", _metrics(15, 0.40, 0.0)),
        CandidateResult("c", 1e-3, "RUNNING", _metrics(15, 0.35, 0.0)),
    ]
    promoted, stats = rank_candidates(candidates, rung_epoch=15, keep=1)
    assert promoted[0].candidate_id == "b"
    assert stats["b"]["map_standard_deviation"] < stats["a"]["map_standard_deviation"]


def test_moving_average_and_boundary_detection():
    assert exponential_moving_average([1.0, 1.0, 1.0]) == pytest.approx([1, 1, 1])
    candidates = [1e-5, 1e-4, 1e-3]
    assert boundary_status(1e-5, candidates) == "lowest"
    assert boundary_status(1e-4, candidates) == "interior"
    assert boundary_status(1e-3, candidates) == "highest"
    assert boundary_extension_candidates(1e-5, "lowest") == [2.5e-6, 5e-6]
    assert boundary_extension_candidates(1e-3, "highest") == [2e-3, 4e-3]


def test_candidate_checkpoint_isolation():
    first = candidate_checkpoint_dir("/drive", "rtdetrv2_l", 1e-4)
    second = candidate_checkpoint_dir("/drive", "rtdetrv2_l", 2e-4)
    assert first != second
    assert first.name == candidate_id("rtdetrv2_l", 1e-4)


def test_selected_yaml_round_trip_and_restart_from_pretrained(tmp_path):
    manifest_dir = tmp_path / "manifests"
    train, val = _write_inputs(tmp_path)
    create_lr_search_manifests(train, val, manifest_dir)
    baseline = BaselineOptimizerAudit(
        "rtdetrv2_l",
        1e-4,
        "configs/rtdetrv2_l/model.yaml",
        "AdamW",
        "CosineAnnealingLR",
        0.05,
        {"epochs": 0},
        "PekingU/rtdetr_v2_r50vd",
    )
    selected = CandidateResult(
        candidate_id("rtdetrv2_l", 1e-4),
        1e-4,
        "COMPLETED",
        _metrics(15, 0.4),
    )
    destination = tmp_path / "selected.yaml"
    export_selected_yaml(
        destination,
        model_id="rtdetrv2_l",
        baseline=baseline,
        candidates=generate_lr_candidates(1e-4),
        selected=selected,
        selection={
            "mean_map": 0.4,
            "mean_aptiny": 0.2,
            "map_standard_deviation": 0.0,
        },
        manifest_dir=manifest_dir,
        git_commit="abc",
        environment={},
    )
    loaded = yaml.safe_load(destination.read_text())
    assert loaded["search"]["selected_learning_rate"] == 1e-4
    assert loaded["final_training"]["dataset"] == "complete_official_train"
    assert loaded["final_training"]["restart_from_pretrained"] is True
    assert "checkpoint" not in loaded["final_training"]


def test_range_test_schedule_and_artifacts(tmp_path):
    schedule = exponential_lr_schedule(1e-4, optimizer_steps=10)
    assert schedule[0] == pytest.approx(1e-6)
    assert schedule[-1] == pytest.approx(2e-3)
    ratios = [right / left for left, right in zip(schedule, schedule[1:])]
    assert ratios == pytest.approx([ratios[0]] * len(ratios))
    history = [
        {
            "optimizer_step": index + 1,
            "learning_rate": learning_rate,
            "raw_loss": 2.0 - index * 0.1,
            "gradient_norm": 1.0,
        }
        for index, learning_rate in enumerate(schedule)
    ]
    summary = save_lr_range_artifacts(
        tmp_path, history, baseline_learning_rate=1e-4
    )
    assert (tmp_path / "history.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert summary["model_state_promotable"] is False
    assert should_stop_range_test([1.0, float("nan")]) == (
        True,
        "non_finite_loss",
    )


def test_scheduler_horizon_and_resume_match_uninterrupted():
    torch = pytest.importorskip("torch")

    def build():
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=15
        )
        return optimizer, scheduler

    full_optimizer, full_scheduler = build()
    full_lrs = []
    for _ in range(15):
        full_optimizer.step()
        full_scheduler.step()
        full_lrs.append(full_optimizer.param_groups[0]["lr"])

    first_optimizer, first_scheduler = build()
    resumed_lrs = []
    for _ in range(5):
        first_optimizer.step()
        first_scheduler.step()
        resumed_lrs.append(first_optimizer.param_groups[0]["lr"])
    optimizer_state = first_optimizer.state_dict()
    scheduler_state = first_scheduler.state_dict()

    resumed_optimizer, resumed_scheduler = build()
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_scheduler.load_state_dict(scheduler_state)
    assert resumed_scheduler.T_max == 15
    for _ in range(5, 15):
        resumed_optimizer.step()
        resumed_scheduler.step()
        resumed_lrs.append(resumed_optimizer.param_groups[0]["lr"])

    assert resumed_lrs == pytest.approx(full_lrs)
    assert resumed_scheduler.state_dict() == full_scheduler.state_dict()


def test_mmdetection_scheduler_is_retargeted_to_fixed_horizon():
    schedulers = [
        {
            "type": "LinearLR",
            "by_epoch": False,
            "begin": 0,
            "end": 500,
        },
        {
            "type": "MultiStepLR",
            "by_epoch": True,
            "begin": 0,
            "end": 12,
            "milestones": [8, 11],
        },
    ]
    configure_scheduler_horizon(schedulers, 12, 15)
    assert schedulers[0]["end"] == 500
    assert schedulers[1]["end"] == 15
    assert schedulers[1]["milestones"] == [10, 14]
