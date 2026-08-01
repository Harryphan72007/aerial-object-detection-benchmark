"""Build a lightweight validated multi-seed HPO result bundle."""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.hpo.workflow import HPO_PROTOCOL_ID
from src.models.registry import load_model_config
from src.paths import ProjectPaths
from src.reproducibility import git_commit
from src.result_export import (
    HPO_REQUIRED_BUNDLE_FILES,
    sanitize_value,
    validate_bundle,
)
from src.training.checkpointing import RunRegistry
from src.utils.serialization import (
    read_json,
    read_yaml,
    sha256_file,
    write_json,
    write_yaml,
)
from src.workflows.hpo_comparison import aggregate_hpo_results


def create_hpo_result_bundle(
    drive_root: str | Path,
    repo_root: str | Path,
    model_id: str,
    dataset_track: str,
) -> Path:
    paths = ProjectPaths.from_value(drive_root)
    repo = Path(repo_root).resolve()
    comparison = aggregate_hpo_results(paths.root, dataset_track)
    selected = [
        group
        for group in comparison["groups"]
        if group["model_id"] == model_id
    ]
    if len(selected) != 2 or any(
        group["status"] != "COMPLETE" for group in selected
    ):
        raise RuntimeError(
            f"Complete baseline and tuned three-seed results are required for "
            f"{model_id} {dataset_track}"
        )
    run_ids = [
        run["run_id"] for group in selected for run in group["runs"]
    ]
    registry = {
        run["run_id"]: run
        for run in RunRegistry(paths).list_available_runs(
            model_id, dataset_track, status="completed"
        )
    }
    runs = [registry[run_id] for run_id in run_ids]
    checkpoint_hashes = []
    official_train = (
        paths.coco(dataset_track) / "annotations" / "instances_train.json"
    ).resolve()
    official_validation = (
        paths.coco(dataset_track) / "annotations" / "instances_val.json"
    ).resolve()
    for run in runs:
        checkpoint = Path(str(run["checkpoint_best_map"]))
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_hashes.append(sha256_file(checkpoint))
        config = read_yaml(Path(str(run["run_dir"])) / "training_config.yaml")
        if (
            config.get("protocol_id") != HPO_PROTOCOL_ID
            or config.get("run_kind") != "final_complete_official_train"
            or config.get("dataset_track") != dataset_track
        ):
            raise RuntimeError(f"incompatible final run: {run['run_id']}")
        if Path(str(config.get("train_annotation", ""))).resolve() != official_train:
            raise RuntimeError(
                f"final run does not use complete official train: {run['run_id']}"
            )
        if (
            Path(str(config.get("validation_annotation", ""))).resolve()
            != official_validation
        ):
            raise RuntimeError(
                f"final run does not use official validation: {run['run_id']}"
            )
    annotation = official_validation
    if not annotation.is_file():
        raise FileNotFoundError(annotation)
    hpo_root = (
        paths.root / "hpo" / HPO_PROTOCOL_ID / model_id / dataset_track
    )
    search_summary = read_json(hpo_root / "search_summary.json")
    best_config = read_yaml(hpo_root / "best_config.yaml")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    bundle_id = f"{model_id}__{dataset_track}__hpo__{stamp}"
    output = paths.result_bundles / bundle_id
    for directory in (
        "configs",
        "search",
        "metrics",
        "reports",
        "provenance",
    ):
        (output / directory).mkdir(parents=True, exist_ok=False)
    write_yaml(output / "configs" / "best_config.yaml", sanitize_value(best_config))
    write_json(
        output / "search" / "search_summary.json",
        sanitize_value(search_summary),
    )
    comparison_payload = read_json(comparison["output"])
    write_json(
        output / "metrics" / "comparison.json",
        sanitize_value(comparison_payload),
    )
    model_config = load_model_config(model_id, repo)
    environment = {
        "environment_fingerprint": search_summary.get(
            "environment_fingerprint"
        ),
        "source_revision": model_config.get("source_revision")
        or model_config.get("pretrained_revision"),
        "source_license": model_config.get("source_license"),
    }
    write_json(
        output / "provenance" / "environment_summary.json",
        sanitize_value(environment),
    )
    write_json(
        output / "provenance" / "dataset_hashes.json",
        sanitize_value(search_summary.get("dataset_hashes", {})),
    )
    evaluation_commit = git_commit(repo)
    training_commits = sorted(
        {str(run.get("git_commit", "")) for run in runs}
    )
    (output / "provenance" / "git_commit.txt").write_text(
        f"evaluation_git_commit={evaluation_commit}\n"
        + "\n".join(
            f"training_git_commit={value}" for value in training_commits
        )
        + "\n",
        encoding="utf-8",
    )
    report = [
        f"# {model_id} multi-seed HPO result",
        "",
        f"Bundle `{bundle_id}` contains measured baseline and tuned summaries.",
        f"Dataset track: `{dataset_track}`.",
        f"Protocol: `{HPO_PROTOCOL_ID}`.",
        "Seeds: 17, 42, 3407.",
        "",
        "Runtime datasets, checkpoints, raw predictions, studies, and logs are "
        "intentionally excluded.",
    ]
    (output / "reports" / "model_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    manifest: dict[str, Any] = {
        "schema_version": 3,
        "result_bundle_id": bundle_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "architecture_family": model_config["architecture_family"],
        "dataset_track": dataset_track,
        "class_names": [
            category["name"] for category in read_json(annotation)["categories"]
        ],
        "protocol_id": HPO_PROTOCOL_ID,
        "run_ids": run_ids,
        "seeds": [17, 42, 3407],
        "seed_status": "multi-seed",
        "checkpoint_sha256": checkpoint_hashes,
        "annotation_sha256": sha256_file(annotation),
        "official_full_train_verified": True,
        "evaluation_git_commit": evaluation_commit,
        "training_git_commits": training_commits,
        "generated_files": sorted(HPO_REQUIRED_BUNDLE_FILES),
        "intentionally_excluded_files": [
            "study.db",
            "checkpoints",
            "raw predictions",
            "logs",
        ],
        "export_status": "created",
    }
    write_json(output / "bundle_manifest.json", manifest)
    errors = validate_bundle(output)
    if errors:
        shutil.rmtree(output)
        raise RuntimeError("HPO bundle validation failed:\n" + "\n".join(errors))
    return output
