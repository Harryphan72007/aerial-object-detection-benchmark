from __future__ import annotations

import ast
import json
from pathlib import Path

import nbformat
import pytest

from src.paths import ProjectPaths
from src.utils.serialization import write_json, write_yaml
from src.workflows.contract import BENCHMARK_CONTRACT, validate_final_config

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "rtdetrv2_l"
RUN_ID = "rtdetrv2_l__2class__640__20260729_120000__seed42"
PRIMARY = {
    "10_resnet50.ipynb",
    "11_swin_t.ipynb",
    "12_vmamba_t.ipynb",
    "13_rtdetrv2.ipynb",
    "30_report.ipynb",
}
LEGACY = {
    "00_colab_repository_setup.ipynb",
    "00_environment_and_data_setup.ipynb",
    "00_visdrone_colab_setup.ipynb",
    "01_dataset_analysis.ipynb",
    "02_train_resnet50_faster_rcnn.ipynb",
    "03_train_swin_t_faster_rcnn.ipynb",
    "04_train_vmamba_t_faster_rcnn.ipynb",
    "05_train_rtdetrv2_l.ipynb",
    "06_train_yolox_s_optional.ipynb",
    "07_evaluate_all_models.ipynb",
    "08_architecture_visualization.ipynb",
    "09_error_analysis.ipynb",
    "10_generate_final_report.ipynb",
    "11_sync_results_to_github.ipynb",
    "12_learning_rate_search.ipynb",
    "13_full_dataset_finetune.ipynb",
    # The retired lr_controlled_v1 entry points, deleted with the protocol.
    "00_bootstrap_colab.ipynb",
    "01_run_model_day.ipynb",
    "02_publish_results.ipynb",
    "03_compare_all_models.ipynb",
    # Folded into the per-model pipeline and the report notebook.
    "00_prepare_visdrone.ipynb",
    "10_hpo_resnet50.ipynb",
    "11_hpo_swin_t.ipynb",
    "12_hpo_vmamba_t.ipynb",
    "13_hpo_rtdetrv2.ipynb",
    "20_finetune_resnet50.ipynb",
    "21_finetune_swin_t.ipynb",
    "22_finetune_vmamba_t.ipynb",
    "23_finetune_rtdetrv2.ipynb",
    "30_evaluate_all_models.ipynb",
    "31_publish_results.ipynb",
}


def _parameter_names(path: Path) -> set[str]:
    notebook = nbformat.read(path, as_version=4)
    cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code" and "parameters" in cell.metadata.get("tags", [])
    ]
    assert len(cells) == 1
    tree = ast.parse(cells[0].source)
    return {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


def _ready_dataset(paths: ProjectPaths) -> None:
    payload = {
        "images": [{"id": 1, "file_name": "one.jpg", "width": 1, "height": 1}],
        "annotations": [],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "vehicle"}],
    }
    for split in ("train", "val"):
        write_json(
            paths.coco("2class") / "annotations" / f"instances_{split}.json",
            payload,
        )
        paths.images(split).mkdir(parents=True, exist_ok=True)


def _selected(paths: ProjectPaths) -> None:
    write_yaml(
        paths.root / "lr_search_configs" / f"{MODEL_ID}_2class_selected.yaml",
        {
            "experiment": {"model_id": MODEL_ID},
            "search": {"selected_learning_rate": 0.0001},
            "final_training": {"learning_rate": 0.0001},
        },
    )


def _run(paths: ProjectPaths, status: str = "completed") -> Path:
    run_dir = paths.final_checkpoints / MODEL_ID / RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "last.pth").write_bytes(b"checkpoint")
    config = {
        "model_id": MODEL_ID,
        "dataset_track": "2class",
        "image_size": 640,
        "seed": 42,
        "epochs": 25,
        "scheduler_horizon": 25,
        "effective_batch_size": 8,
        "use_amp": True,
        "run_kind": "final_complete_official_train",
        "overrides": {"learning_rate": 0.0001},
    }
    write_yaml(run_dir / "training_config.yaml", config)
    manifest = {
        "run_id": RUN_ID,
        "run_dir": str(run_dir),
        "model_id": MODEL_ID,
        "dataset_track": "2class",
        "status": status,
    }
    write_json(run_dir / "run_manifest.json", manifest)
    write_json(paths.checkpoint_registry, {"schema_version": 1, "runs": {RUN_ID: manifest}})
    return run_dir


def test_expected_primary_notebooks_and_no_legacy_notebooks():
    actual = {path.name for path in (ROOT / "notebooks").glob("*.ipynb")}
    assert actual == PRIMARY
    assert not any((ROOT / "notebooks" / name).exists() for name in LEGACY)


def test_notebooks_are_clean_valid_and_config_cells_are_minimal():
    model_parameters = {
        "DATASET_TRACK",
        "START",
        "FULL_MATRIX",
        "USE_GOOGLE_DRIVE",
    }
    expected = {
        "10_resnet50.ipynb": model_parameters,
        "11_swin_t.ipynb": model_parameters,
        "12_vmamba_t.ipynb": model_parameters,
        "13_rtdetrv2.ipynb": model_parameters,
        "30_report.ipynb": {
            "DATASET_TRACK",
            "EVALUATE_MISSING",
            "PUBLISH_RESULTS",
            "DRY_RUN",
            "USE_GOOGLE_DRIVE",
        },
    }
    for path in (ROOT / "notebooks").rglob("*.ipynb"):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        assert not {"colab", "varInspector", "widgets"}.intersection(
            notebook.metadata
        )
        for cell in notebook.cells:
            assert not {
                "collapsed",
                "colab",
                "execution",
                "jupyter",
                "outputId",
                "scrolled",
            }.intersection(cell.metadata)
            if cell.cell_type == "code":
                assert cell.outputs == []
                assert cell.execution_count is None
    for name, variables in expected.items():
        assert _parameter_names(ROOT / "notebooks" / name) == variables


def test_final_contract_rejects_incompatible_resume_configuration():
    config = {
        "dataset_track": "2class",
        "image_size": 640,
        "seed": 42,
        "epochs": 25,
        "scheduler_horizon": 25,
        "effective_batch_size": 8,
        "use_amp": True,
        "run_kind": "final_complete_official_train",
        "overrides": {"learning_rate": 0.0001},
    }
    validate_final_config(config, selected_lr=0.0001)
    invalid = dict(config, image_size=1024)
    with pytest.raises(ValueError, match="Incompatible final run"):
        validate_final_config(invalid, selected_lr=0.0001)
    assert BENCHMARK_CONTRACT["effective_batch_size"] == 8


def test_publishing_configuration_defaults_to_dry_run():
    """Publishing is a push to a public repository; it never happens by default."""
    checked = 0
    for name in sorted(PRIMARY):
        notebook = nbformat.read(ROOT / "notebooks" / name, as_version=4)
        parameter = next(
            cell
            for cell in notebook.cells
            if "parameters" in cell.metadata.get("tags", [])
        )
        values: dict[str, object] = {}
        exec(compile(parameter.source, "<parameters>", "exec"), {}, values)
        if "PUBLISH_RESULTS" not in values:
            continue
        checked += 1
        assert values["PUBLISH_RESULTS"] is False, name
        assert values["DRY_RUN"] is True, name
    assert checked, "no notebook exposes PUBLISH_RESULTS"
