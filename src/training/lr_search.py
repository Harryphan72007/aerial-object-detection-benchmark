"""Deterministic learning-rate search protocol and dataset-manifest utilities."""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.paths import ProjectPaths
from src.training.recipes import (
    RTDETR_BASELINE_LR,
    RTDETR_OPTIMIZER_TYPE,
    RTDETR_SCHEDULER_TYPE,
    RTDETR_WARMUP_EPOCHS,
    RTDETR_WEIGHT_DECAY,
)
from src.utils.serialization import read_json, read_yaml, write_json, write_yaml

EXPERIMENT_NAME = "visdrone_lr_controlled_benchmark"
EXPERIMENT_DISPLAY_NAME = "VisDrone Learning-Rate-Controlled Architecture Benchmark"
EXPERIMENT_CLAIM = (
    "Each model receives the same learning-rate search protocol, search data policy, "
    "promotion rules, final dataset (official train minus a fixed held-out "
    "model-selection split), model-selection policy, input resolution, effective "
    "batch size, seed policy, evaluation protocol, and final training budget. The "
    "final checkpoint is selected on the held-out model-selection split; official "
    "validation is evaluated exactly once, at the end, and never drives selection. "
    "Only the learning rate differs between candidates."
)
SUPPORTED_PRIMARY_MODELS = {
    "faster_rcnn_resnet50",
    "faster_rcnn_swin_t",
    "faster_rcnn_vmamba_t",
    "rtdetrv2_l",
}
CANDIDATE_STATUSES = {
    "PENDING",
    "RUNNING",
    "PROMOTED",
    "ELIMINATED",
    "COMPLETED",
    "FAILED_OOM",
    "FAILED_NUMERICAL",
    "FAILED_ENVIRONMENT",
    "FAILED_ADAPTER",
}
PROMOTION_RUNGS = (
    {"epoch": 2, "keep": 5},
    {"epoch": 5, "keep": 3},
    {"epoch": 10, "keep": 2},
    {"epoch": 15, "keep": 1},
)
METRIC_WINDOWS = {2: (1, 2), 5: (4, 5), 10: (8, 10), 15: (13, 15)}


@dataclass(frozen=True)
class FixedBenchmarkSettings:
    dataset_track: str = "2class"
    image_size: int = 640
    seed: int = 42
    amp: bool = True
    effective_batch_size: int = 8
    search_max_epochs: int = 15
    final_epochs: int = 25
    primary_metric: str = "mAP_50_95"
    secondary_metric: str = "APtiny"


@dataclass(frozen=True)
class BaselineOptimizerAudit:
    model_id: str
    learning_rate: float
    baseline_config_path: str
    optimizer_type: str
    scheduler_type: str
    weight_decay: float | None
    warmup_configuration: Any
    pretrained_checkpoint: str | None
    parameter_group_configuration: Any = None


@dataclass
class CandidateResult:
    candidate_id: str
    learning_rate: float
    status: str
    metrics: list[dict[str, Any]]
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in CANDIDATE_STATUSES:
            raise ValueError(f"invalid candidate status: {self.status}")


def sha256_json(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate_lr_candidates(
    baseline_learning_rate: float,
    *,
    count: int = 9,
    safe_interval: tuple[float, float] | None = None,
) -> list[float]:
    """Return a sorted deterministic logarithmic grid containing the baseline."""
    if not math.isfinite(baseline_learning_rate) or baseline_learning_rate <= 0:
        raise ValueError("baseline_learning_rate must be finite and positive")
    if count != 9:
        raise ValueError("the controlled benchmark requires exactly 9 LR candidates")
    lower, upper = (
        safe_interval
        if safe_interval is not None
        else (baseline_learning_rate / 8.0, baseline_learning_rate * 8.0)
    )
    if not (math.isfinite(lower) and math.isfinite(upper) and 0 < lower < upper):
        raise ValueError("safe LR interval must contain finite positive increasing values")
    values = np.geomspace(lower, upper, num=count).astype(float).tolist()
    if len(set(values)) != count or any(value <= 0 for value in values):
        raise ValueError("LR candidates must be 9 unique positive values")
    log_steps = np.diff(np.log(values))
    if not np.allclose(log_steps, log_steps[0], rtol=1e-10):
        raise AssertionError("LR grid is not logarithmically spaced")
    closest_log_distance = min(
        abs(math.log(value / baseline_learning_rate)) for value in values
    )
    half_log_step = abs(float(log_steps[0])) / 2
    if closest_log_distance > half_log_step + 1e-12:
        raise ValueError(
            "safe LR interval is not centered closely enough on the baseline LR"
        )
    return values


def candidate_id(model_id: str, learning_rate: float, seed: int = 42) -> str:
    if model_id not in SUPPORTED_PRIMARY_MODELS:
        raise ValueError(f"unsupported primary model: {model_id}")
    if learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    return f"{model_id}__2class__lr_{learning_rate:.3e}__seed{seed}"


def candidate_checkpoint_dir(
    drive_root: str | Path, model_id: str, learning_rate: float, seed: int = 42
) -> Path:
    return (
        Path(drive_root)
        / "checkpoints"
        / "lr_search"
        / model_id
        / candidate_id(model_id, learning_rate, seed)
    )


def resolve_batch_policy(
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int = 1,
    required_effective_batch_size: int = 8,
) -> dict[str, int]:
    values = (per_device_batch_size, gradient_accumulation_steps, world_size)
    if any(value <= 0 for value in values):
        raise ValueError("batch policy values must be positive")
    effective = math.prod(values)
    if effective != required_effective_batch_size:
        raise ValueError(
            f"effective batch size must be {required_effective_batch_size}, got "
            f"{per_device_batch_size} * {gradient_accumulation_steps} * {world_size} "
            f"= {effective}"
        )
    return {
        "per_device_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "world_size": world_size,
        "effective_batch_size": effective,
    }


def assert_only_learning_rate_changes(configs: Sequence[Mapping[str, Any]]) -> None:
    if not configs:
        raise ValueError("at least one candidate config is required")
    reference = {key: value for key, value in configs[0].items() if key != "learning_rate"}
    for index, config in enumerate(configs):
        if set(config) != set(configs[0]):
            raise AssertionError(f"candidate {index} has different config keys")
        fixed = {key: value for key, value in config.items() if key != "learning_rate"}
        if fixed != reference:
            raise AssertionError(f"candidate {index} changes a non-LR setting")


def resolve_baseline_optimizer(
    model_id: str,
    repo_root: str | Path,
) -> BaselineOptimizerAudit:
    """Resolve the actual optimizer recipe; MMDetection configs load at runtime."""
    if model_id not in SUPPORTED_PRIMARY_MODELS:
        raise ValueError(f"unsupported primary model: {model_id}")
    repo_root = Path(repo_root).resolve()
    model_config_path = repo_root / "configs" / model_id / "model.yaml"
    model_config = read_yaml(model_config_path)
    if model_id == "rtdetrv2_l":
        return BaselineOptimizerAudit(
            model_id=model_id,
            learning_rate=RTDETR_BASELINE_LR,
            baseline_config_path=str(model_config_path),
            optimizer_type=RTDETR_OPTIMIZER_TYPE,
            scheduler_type=RTDETR_SCHEDULER_TYPE,
            weight_decay=RTDETR_WEIGHT_DECAY,
            warmup_configuration={"epochs": RTDETR_WARMUP_EPOCHS},
            pretrained_checkpoint=str(model_config["pretrained_model_name_or_path"]),
            parameter_group_configuration=None,
        )
    import os

    environment_name = str(model_config["external_root_env"])
    external_root = os.environ.get(environment_name)
    if not external_root:
        raise RuntimeError(
            f"Set {environment_name}; the baseline LR must be read from the final "
            "loaded upstream MMDetection configuration."
        )
    config_path = Path(external_root) / str(model_config["framework_config"])
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    try:
        from mmengine.config import Config
    except ImportError as exc:
        raise RuntimeError("MMEngine is required to resolve the optimizer config") from exc
    cfg = Config.fromfile(str(config_path))
    optimizer = dict(cfg.optim_wrapper.optimizer)
    schedulers = cfg.get("param_scheduler", [])
    if not isinstance(schedulers, list):
        schedulers = [schedulers]
    warmup = [
        dict(scheduler)
        for scheduler in schedulers
        if "linear" in str(scheduler.get("type", "")).lower()
        or int(scheduler.get("begin", 0)) == 0 and not bool(scheduler.get("by_epoch", True))
    ]
    scheduler_types = [str(scheduler.get("type", "unknown")) for scheduler in schedulers]
    pretrained = model_config.get("pretrained")
    weight_env = model_config.get("pretrained_weight_env")
    if weight_env:
        pretrained = os.environ.get(str(weight_env))
        if not pretrained:
            raise RuntimeError(
                f"Set {weight_env} to the verified original pretrained checkpoint. "
                "The LR-controlled benchmark does not permit a scratch substitute."
            )
    return BaselineOptimizerAudit(
        model_id=model_id,
        learning_rate=float(optimizer["lr"]),
        baseline_config_path=str(config_path.resolve()),
        optimizer_type=str(optimizer.get("type", "unknown")),
        scheduler_type="+".join(scheduler_types) or "none",
        weight_decay=float(optimizer["weight_decay"])
        if optimizer.get("weight_decay") is not None
        else None,
        warmup_configuration=warmup,
        pretrained_checkpoint=str(pretrained) if pretrained else None,
        parameter_group_configuration=cfg.optim_wrapper.get("paramwise_cfg"),
    )


def _density_edges(counts: Sequence[int]) -> tuple[float, float, float]:
    if not counts:
        return (0.0, 0.0, 0.0)
    return tuple(float(value) for value in np.quantile(counts, [0.25, 0.5, 0.75]))


def _bucket(value: float, edges: Sequence[float]) -> int:
    return sum(value > edge for edge in edges)


def _image_features(data: dict[str, Any]) -> tuple[dict[int, tuple[Any, ...]], dict[int, list[dict[str, Any]]]]:
    annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in data.get("annotations", []):
        annotations[int(annotation["image_id"])].append(annotation)
    counts = [len(annotations[int(image["id"])]) for image in data["images"]]
    density_edges = _density_edges(counts)
    features: dict[int, tuple[Any, ...]] = {}
    for image in data["images"]:
        image_id = int(image["id"])
        rows = annotations[image_id]
        person = sum(int(row["category_id"]) == 1 for row in rows)
        vehicle = sum(int(row["category_id"]) == 2 for row in rows)
        tiny = sum(float(row.get("area", 0)) < 16**2 for row in rows)
        small = sum(16**2 <= float(row.get("area", 0)) < 32**2 for row in rows)
        total = len(rows)
        presence = "empty" if not total else "both" if person and vehicle else "person" if person else "vehicle"
        person_ratio = person / total if total else 0.0
        tiny_ratio = tiny / total if total else 0.0
        small_ratio = small / total if total else 0.0
        features[image_id] = (
            presence,
            _bucket(person_ratio, (0.0, 0.25, 0.5, 0.75)),
            _bucket(total, density_edges),
            _bucket(tiny_ratio, (0.0, 0.5, 0.9)),
            _bucket(small_ratio, (0.0, 0.25, 0.75)),
        )
    return features, annotations


def _stratified_take(
    image_ids: Sequence[int],
    features: Mapping[int, tuple[Any, ...]],
    count: int,
    seed: int,
) -> list[int]:
    if not 0 <= count <= len(image_ids):
        raise ValueError("invalid stratified sample size")
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for image_id in image_ids:
        groups[features[image_id]].append(image_id)
    rng = random.Random(seed)
    for ids in groups.values():
        ids.sort()
        rng.shuffle(ids)
    total = len(image_ids)
    allocations: dict[tuple[Any, ...], int] = {}
    remainders: list[tuple[float, tuple[Any, ...]]] = []
    for key, ids in groups.items():
        exact = len(ids) * count / total if total else 0.0
        allocations[key] = min(len(ids), int(math.floor(exact)))
        remainders.append((exact - math.floor(exact), key))
    remaining = count - sum(allocations.values())
    for _, key in sorted(remainders, key=lambda item: (-item[0], repr(item[1]))):
        if remaining <= 0:
            break
        if allocations[key] < len(groups[key]):
            allocations[key] += 1
            remaining -= 1
    if remaining:
        for key in sorted(groups, key=repr):
            while remaining and allocations[key] < len(groups[key]):
                allocations[key] += 1
                remaining -= 1
    selected = [
        image_id
        for key in sorted(groups, key=repr)
        for image_id in groups[key][: allocations[key]]
    ]
    return sorted(selected)


def _subset_coco(data: dict[str, Any], image_ids: set[int], description: str) -> dict[str, Any]:
    payload = {
        "info": {**data.get("info", {}), "description": description},
        "licenses": data.get("licenses", []),
        "categories": data.get("categories", []),
        "images": [image for image in data["images"] if int(image["id"]) in image_ids],
        "annotations": [
            annotation
            for annotation in data["annotations"]
            if int(annotation["image_id"]) in image_ids
        ],
    }
    return payload


def _remap_validation_ids(validation: dict[str, Any], first_id: int) -> dict[str, Any]:
    mapping: dict[int, int] = {}
    images: list[dict[str, Any]] = []
    for offset, image in enumerate(sorted(validation["images"], key=lambda item: int(item["id"]))):
        source_id = int(image["id"])
        new_id = first_id + offset
        mapping[source_id] = new_id
        images.append({**image, "id": new_id, "source_image_id": source_id})
    annotations = [
        {**annotation, "image_id": mapping[int(annotation["image_id"])]}
        for annotation in validation["annotations"]
    ]
    return {**validation, "images": images, "annotations": annotations}


def summarize_coco(data: dict[str, Any]) -> dict[str, Any]:
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in data.get("annotations", []):
        by_image[int(annotation["image_id"])].append(annotation)
    object_count = len(data.get("annotations", []))
    person = sum(int(row["category_id"]) == 1 for row in data.get("annotations", []))
    vehicle = sum(int(row["category_id"]) == 2 for row in data.get("annotations", []))
    tiny = sum(float(row.get("area", 0)) < 16**2 for row in data.get("annotations", []))
    small = sum(16**2 <= float(row.get("area", 0)) < 32**2 for row in data.get("annotations", []))
    images = data.get("images", [])
    empty = sum(not by_image[int(image["id"])] for image in images)
    both = sum(
        {int(row["category_id"]) for row in by_image[int(image["id"])]} >= {1, 2}
        for image in images
    )
    counts = [len(by_image[int(image["id"])]) for image in images]
    return {
        "images": len(images),
        "objects": object_count,
        "person_objects": person,
        "vehicle_objects": vehicle,
        "person_to_vehicle_ratio": person / vehicle if vehicle else None,
        "tiny_object_ratio": tiny / object_count if object_count else 0.0,
        "small_object_ratio": small / object_count if object_count else 0.0,
        "empty_image_ratio": empty / len(images) if images else 0.0,
        "images_containing_both_classes_ratio": both / len(images) if images else 0.0,
        "objects_per_image_mean": statistics.fmean(counts) if counts else 0.0,
    }


def _ensure_image_identity(
    data: dict[str, Any],
    *,
    original_split: str,
    source_archive_identity: str,
) -> None:
    for image in data.get("images", []):
        recorded_split = image.setdefault("original_split", original_split)
        recorded_archive = image.setdefault(
            "source_archive_sha256", source_archive_identity
        )
        if recorded_split != original_split:
            raise ValueError(
                f"image {image.get('file_name')} records original split "
                f"{recorded_split!r}, expected {original_split!r}"
            )
        if not recorded_archive:
            raise ValueError(
                f"image {image.get('file_name')} has no source archive identity"
            )


def create_lr_search_manifests(
    official_train_json: str | Path,
    official_validation_json: str | Path,
    output_dir: str | Path,
    *,
    dataset_track: str = "2class",
    seed: int = 42,
    search_train_fraction: float = 0.20,
    search_validation_fraction: float = 0.05,
    model_selection_fraction: float = 0.05,
    source_archive_identities: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    official_train_json = Path(official_train_json).resolve()
    official_validation_json = Path(official_validation_json).resolve()
    train = json.loads(official_train_json.read_text(encoding="utf-8"))
    validation = json.loads(official_validation_json.read_text(encoding="utf-8"))
    source_archive_identities = dict(source_archive_identities or {})
    train_identity = source_archive_identities.get(
        "train", f"coco-source:{sha256_json(official_train_json)}"
    )
    validation_identity = source_archive_identities.get(
        "val", f"coco-source:{sha256_json(official_validation_json)}"
    )
    _ensure_image_identity(
        train,
        original_split="train",
        source_archive_identity=train_identity,
    )
    _ensure_image_identity(
        validation,
        original_split="val",
        source_archive_identity=validation_identity,
    )
    expected_category_ids = (
        [1, 2] if dataset_track == "2class" else list(range(1, 11))
    )
    if dataset_track not in {"2class", "10class"}:
        raise ValueError(f"unsupported dataset track: {dataset_track}")
    if [
        int(category["id"]) for category in train.get("categories", [])
    ] != expected_category_ids:
        raise ValueError(
            f"{dataset_track} search requires category IDs "
            f"{expected_category_ids}"
        )
    features, _ = _image_features(train)
    all_ids = sorted(int(image["id"]) for image in train["images"])
    train_count = round(len(all_ids) * search_train_fraction)
    validation_count = round(len(all_ids) * search_validation_fraction)
    search_train_ids = _stratified_take(all_ids, features, train_count, seed)
    remaining = sorted(set(all_ids) - set(search_train_ids))
    search_validation_ids = _stratified_take(remaining, features, validation_count, seed + 1)
    # Held-out model-selection split: drawn from official train but disjoint from
    # both search subsets, so the final checkpoint is selected on data that never
    # informed the learning-rate search and is never in the official validation
    # set. Final training uses official train MINUS this holdout; official
    # validation is evaluated exactly once, at the end, and never drives selection.
    selection_pool = sorted(
        set(all_ids) - set(search_train_ids) - set(search_validation_ids)
    )
    selection_count = round(len(all_ids) * model_selection_fraction)
    selection_count = min(selection_count, len(selection_pool))
    model_selection_ids = _stratified_take(
        selection_pool, features, selection_count, seed + 2
    )
    final_train_ids = sorted(set(all_ids) - set(model_selection_ids))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    maximum_train_id = max(all_ids, default=0)
    canonical_validation = _remap_validation_ids(validation, maximum_train_id + 1)
    payloads = {
        "search_train_seed42.json": _subset_coco(
            train, set(search_train_ids), "VisDrone LR search train subset (seed 42)"
        ),
        "search_validation_seed42.json": _subset_coco(
            train,
            set(search_validation_ids),
            "VisDrone LR search validation subset drawn only from official train (seed 42)",
        ),
        "model_selection_seed42.json": _subset_coco(
            train,
            set(model_selection_ids),
            "VisDrone held-out model-selection split from official train, disjoint "
            "from the search subsets (seed 42); drives final best.pth selection",
        ),
        "final_train_seed42.json": _subset_coco(
            train,
            set(final_train_ids),
            "VisDrone final training set: official train minus the model-selection "
            "holdout (seed 42)",
        ),
        "official_full_train.json": {
            **train,
            "info": {**train.get("info", {}), "description": "Complete official VisDrone train"},
        },
        "official_validation.json": {
            **canonical_validation,
            "info": {
                **canonical_validation.get("info", {}),
                "description": "Complete official VisDrone validation (globally remapped IDs)",
            },
        },
    }
    paths: dict[str, str] = {}
    for filename, payload in payloads.items():
        path = output_dir / filename
        write_json(path, payload)
        paths[filename] = str(path)
    verification = validate_lr_search_manifests(output_dir)
    summary = {
        "dataset_track": dataset_track,
        "seed": seed,
        "fractions": {
            "search_train": search_train_fraction,
            "search_validation": search_validation_fraction,
            "model_selection": model_selection_fraction,
        },
        "statistics": {name: summarize_coco(payload) for name, payload in payloads.items()},
        "verification": verification,
        "hashes": {name: sha256_json(path) for name, path in paths.items()},
        "sources": {
            "official_train": {
                "path": str(official_train_json),
                "sha256": sha256_json(official_train_json),
                "source_archive_sha256": train_identity,
            },
            "official_validation": {
                "path": str(official_validation_json),
                "sha256": sha256_json(official_validation_json),
                "source_archive_sha256": validation_identity,
            },
        },
    }
    write_json(output_dir / "split_summary.json", summary)
    return summary


def validate_lr_search_manifests(
    manifest_dir: str | Path,
    *,
    official_train_json: str | Path | None = None,
    official_validation_json: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(manifest_dir)
    search_train = json.loads((root / "search_train_seed42.json").read_text(encoding="utf-8"))
    search_validation = json.loads(
        (root / "search_validation_seed42.json").read_text(encoding="utf-8")
    )
    model_selection = json.loads(
        (root / "model_selection_seed42.json").read_text(encoding="utf-8")
    )
    final_train = json.loads(
        (root / "final_train_seed42.json").read_text(encoding="utf-8")
    )
    official_train = json.loads((root / "official_full_train.json").read_text(encoding="utf-8"))
    official_validation = json.loads(
        (root / "official_validation.json").read_text(encoding="utf-8")
    )
    ids = lambda data: {int(image["id"]) for image in data["images"]}
    search_train_ids = ids(search_train)
    search_validation_ids = ids(search_validation)
    model_selection_ids = ids(model_selection)
    final_train_ids = ids(final_train)
    official_train_ids = ids(official_train)
    official_validation_ids = ids(official_validation)
    filenames = lambda data: {str(image["file_name"]) for image in data["images"]}
    identities = lambda data: {
        (
            str(image["file_name"]),
            str(image.get("original_split", "")),
            str(image.get("source_archive_sha256", "")),
        )
        for image in data["images"]
    }
    search_train_filenames = filenames(search_train)
    search_validation_filenames = filenames(search_validation)
    model_selection_filenames = filenames(model_selection)
    final_train_filenames = filenames(final_train)
    official_train_filenames = filenames(official_train)
    official_validation_filenames = filenames(official_validation)
    search_train_identities = identities(search_train)
    search_validation_identities = identities(search_validation)
    official_train_identities = identities(official_train)
    official_validation_identities = identities(official_validation)
    checks = {
        "search_numeric_ids_disjoint": not (
            search_train_ids & search_validation_ids
        ),
        "search_train_subset_official_train": search_train_ids <= official_train_ids,
        "search_validation_subset_official_train": search_validation_ids <= official_train_ids,
        "official_train_validation_numeric_ids_disjoint": not (
            official_train_ids & official_validation_ids
        ),
        "search_train_validation_filenames_disjoint": not (
            search_train_filenames & search_validation_filenames
        ),
        "search_train_filenames_subset_official_train": (
            search_train_filenames <= official_train_filenames
        ),
        "search_validation_filenames_subset_official_train": (
            search_validation_filenames <= official_train_filenames
        ),
        "official_train_validation_filenames_disjoint": not (
            official_train_filenames & official_validation_filenames
        ),
        "model_selection_subset_official_train": (
            model_selection_ids <= official_train_ids
        ),
        "model_selection_disjoint_search_train": not (
            model_selection_ids & search_train_ids
        ),
        "model_selection_disjoint_search_validation": not (
            model_selection_ids & search_validation_ids
        ),
        "model_selection_disjoint_final_train": not (
            model_selection_ids & final_train_ids
        ),
        "final_train_union_model_selection_equals_official_train": (
            (final_train_ids | model_selection_ids) == official_train_ids
        ),
        "model_selection_filenames_disjoint_official_validation": not (
            model_selection_filenames & official_validation_filenames
        ),
        "final_train_filenames_disjoint_official_validation": not (
            final_train_filenames & official_validation_filenames
        ),
        "search_train_stable_identities_subset_official_train": (
            search_train_identities <= official_train_identities
        ),
        "search_validation_stable_identities_subset_official_train": (
            search_validation_identities <= official_train_identities
        ),
        "official_train_validation_stable_identities_disjoint": not (
            official_train_identities & official_validation_identities
        ),
        "official_train_original_split_is_train": all(
            image.get("original_split") == "train"
            for image in official_train["images"]
        ),
        "official_validation_original_split_is_val": all(
            image.get("original_split") == "val"
            for image in official_validation["images"]
        ),
        "source_archive_identity_present": all(
            image.get("source_archive_sha256")
            for data in (official_train, official_validation)
            for image in data["images"]
        ),
    }
    summary_path = root / "split_summary.json"
    if official_train_json is not None:
        checks["official_train_source_hash_current"] = (
            sha256_json(official_train_json)
            == json.loads(summary_path.read_text(encoding="utf-8"))["sources"][
                "official_train"
            ]["sha256"]
        )
    if official_validation_json is not None:
        checks["official_validation_source_hash_current"] = (
            sha256_json(official_validation_json)
            == json.loads(summary_path.read_text(encoding="utf-8"))["sources"][
                "official_validation"
            ]["sha256"]
        )
    if not all(checks.values()):
        raise AssertionError(f"invalid LR-search manifests: {checks}")
    return checks


def lr_search_manifests_current(
    manifest_dir: str | Path,
    *,
    official_train_json: str | Path,
    official_validation_json: str | Path,
) -> bool:
    """Whether the persisted manifests still describe this exact dataset."""
    root = Path(manifest_dir)
    required = (
        "search_train_seed42.json",
        "search_validation_seed42.json",
        "official_full_train.json",
        "official_validation.json",
        "split_summary.json",
    )
    if not all((root / name).is_file() for name in required):
        return False
    try:
        validate_lr_search_manifests(
            root,
            official_train_json=official_train_json,
            official_validation_json=official_validation_json,
        )
    except (AssertionError, FileNotFoundError, KeyError, json.JSONDecodeError, ValueError):
        return False
    return True


def ensure_lr_search_manifests(
    paths: ProjectPaths, *, seed: int = 42, force: bool = False
) -> tuple[dict[str, Any], str]:
    """Build the held-out split manifests once and reuse them while they verify.

    These manifests fix which images the search may see and which are held out
    for model selection, so rebuilding them when they are already valid would
    silently move the held-out split under a run that has already started.
    Returns the split summary and whether it was ``reused`` or ``created``.
    """
    manifest_dir = paths.lr_search_manifests
    annotations = paths.coco("2class") / "annotations"
    official_train_json = annotations / "instances_train.json"
    official_validation_json = annotations / "instances_val.json"
    if not force and lr_search_manifests_current(
        manifest_dir,
        official_train_json=official_train_json,
        official_validation_json=official_validation_json,
    ):
        return read_json(manifest_dir / "split_summary.json"), "reused"
    # The archive identity travels into the manifests so a rebuilt dataset can
    # never be mistaken for the one a stored manifest was cut from.
    source_archive_identities = {
        split: str(
            read_json(paths.dataset_manifests / f"{split}_extraction.json")[
                "archive_sha256"
            ]
        )
        for split in ("train", "val")
    }
    summary = create_lr_search_manifests(
        official_train_json,
        official_validation_json,
        manifest_dir,
        seed=seed,
        source_archive_identities=source_archive_identities,
    )
    return summary, "created"


def assert_final_training_uses_official_train(
    manifest_dir: str | Path,
    official_train_source: str | Path | None = None,
) -> None:
    root = Path(manifest_dir)
    final_train = json.loads((root / "official_full_train.json").read_text(encoding="utf-8"))
    official_validation = json.loads(
        (root / "official_validation.json").read_text(encoding="utf-8")
    )
    final_train_image_ids = {int(image["id"]) for image in final_train["images"]}
    if official_train_source is None:
        summary_path = root / "split_summary.json"
        if summary_path.exists():
            official_train_source = json.loads(
                summary_path.read_text(encoding="utf-8")
            ).get("sources", {}).get("official_train", {}).get("path")
    if official_train_source is None:
        raise ValueError("official_train_source is required to prove dataset identity")
    source_payload = json.loads(
        Path(official_train_source).read_text(encoding="utf-8")
    )
    official_train_image_ids = {
        int(image["id"]) for image in source_payload["images"]
    }
    official_validation_image_ids = {
        int(image["id"]) for image in official_validation["images"]
    }
    assert final_train_image_ids == official_train_image_ids
    assert not (final_train_image_ids & official_validation_image_ids)
    summary = json.loads(
        (root / "split_summary.json").read_text(encoding="utf-8")
    )
    train_archive_identity = str(
        summary["sources"]["official_train"]["source_archive_sha256"]
    )

    def stable_identity(
        image: Mapping[str, Any],
        *,
        default_split: str = "",
        default_archive: str = "",
    ) -> tuple[str, str, str]:
        return (
            str(image["file_name"]),
            str(image.get("original_split", default_split)),
            str(image.get("source_archive_sha256", default_archive)),
        )

    final_train_identities = {
        stable_identity(image) for image in final_train["images"]
    }
    official_train_identities = {
        stable_identity(
            image,
            default_split="train",
            default_archive=train_archive_identity,
        )
        for image in source_payload["images"]
    }
    official_validation_identities = {
        stable_identity(image) for image in official_validation["images"]
    }
    assert final_train_identities == official_train_identities
    assert not (final_train_identities & official_validation_identities)


def assert_selection_split_held_out(manifest_dir: str | Path) -> None:
    """Prove the two-stage HPO final protocol never selects on official val.

    The final training set (``final_train_seed42.json``) is official train minus
    the held-out ``model_selection_seed42.json`` split. Selection runs on the
    holdout, which is disjoint from both search subsets and from official
    validation. This is the invariant that makes the reported number free of the
    checkpoint-selection bias that the earlier per-epoch official-val selection
    introduced.
    """
    root = Path(manifest_dir)
    load = lambda name: json.loads((root / name).read_text(encoding="utf-8"))
    ids = lambda data: {int(image["id"]) for image in data["images"]}
    names = lambda data: {str(image["file_name"]) for image in data["images"]}

    final_train = load("final_train_seed42.json")
    model_selection = load("model_selection_seed42.json")
    search_train = load("search_train_seed42.json")
    search_validation = load("search_validation_seed42.json")
    official_train = load("official_full_train.json")
    official_validation = load("official_validation.json")

    final_ids = ids(final_train)
    selection_ids = ids(model_selection)
    if final_ids & selection_ids:
        raise AssertionError("final-train and model-selection splits overlap")
    if (final_ids | selection_ids) != ids(official_train):
        raise AssertionError(
            "final-train + model-selection must partition official train"
        )
    if selection_ids & ids(search_train):
        raise AssertionError("model-selection overlaps the search-train subset")
    if selection_ids & ids(search_validation):
        raise AssertionError("model-selection overlaps the search-validation subset")
    if names(model_selection) & names(official_validation):
        raise AssertionError("model-selection overlaps official validation")
    if names(final_train) & names(official_validation):
        raise AssertionError("final-train overlaps official validation")


def exponential_moving_average(values: Iterable[float], beta: float = 0.98) -> list[float]:
    if not 0 <= beta < 1:
        raise ValueError("beta must be in [0, 1)")
    result: list[float] = []
    average = 0.0
    for index, value in enumerate(values, start=1):
        average = beta * average + (1 - beta) * float(value)
        result.append(average / (1 - beta**index))
    return result


def _metric(row: Mapping[str, Any], names: Sequence[str], default: float = float("nan")) -> float:
    for name in names:
        value = row.get(name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return default


def _present_metrics(
    rows: Sequence[Mapping[str, Any]], names: Sequence[str]
) -> list[float]:
    values: list[float] = []
    for row in rows:
        if not any(name in row and row.get(name) is not None for name in names):
            continue
        values.append(_metric(row, names))
    return values


def selection_statistics(metrics: Sequence[Mapping[str, Any]], rung_epoch: int) -> dict[str, float]:
    start, end = METRIC_WINDOWS[rung_epoch]
    window = [row for row in metrics if start <= int(row.get("epoch", 0)) <= end]
    maps = [
        _metric(row, ("mAP_50_95", "mAP", "coco/bbox_mAP", "bbox_mAP"))
        for row in window
    ]
    aptiny = [
        _metric(row, ("APtiny", "aerial_coco/APtiny", "aptiny")) for row in window
    ]
    losses = [
        _metric(row, ("validation_loss", "val_loss", "training_loss", "loss"))
        for row in window
    ]
    gradients = [_metric(row, ("gradient_norm", "grad_norm")) for row in window]
    finite_maps = [value for value in maps if math.isfinite(value)]
    finite_tiny = [value for value in aptiny if math.isfinite(value)]
    finite_losses = [value for value in losses if math.isfinite(value)]
    finite_gradients = [value for value in gradients if math.isfinite(value)]
    return {
        "mean_map": statistics.fmean(finite_maps) if finite_maps else float("-inf"),
        "mean_aptiny": statistics.fmean(finite_tiny) if finite_tiny else float("-inf"),
        "map_standard_deviation": statistics.pstdev(finite_maps) if len(finite_maps) > 1 else 0.0,
        "mean_validation_loss": statistics.fmean(finite_losses) if finite_losses else float("inf"),
        "gradient_standard_deviation": statistics.pstdev(finite_gradients)
        if len(finite_gradients) > 1
        else 0.0,
        "valid_epochs": float(len(finite_maps)),
    }


def rank_candidates(
    candidates: Sequence[CandidateResult],
    *,
    rung_epoch: int,
    keep: int,
) -> tuple[list[CandidateResult], dict[str, dict[str, float]]]:
    if rung_epoch not in METRIC_WINDOWS:
        raise ValueError(f"unsupported rung epoch: {rung_epoch}")
    viable: list[CandidateResult] = []
    stats: dict[str, dict[str, float]] = {}
    for candidate in candidates:
        candidate_stats = selection_statistics(candidate.metrics, rung_epoch)
        stats[candidate.candidate_id] = candidate_stats
        rows = [
            row
            for row in candidate.metrics
            if int(row.get("epoch", 0)) <= rung_epoch
        ]
        losses = _present_metrics(rows, ("training_loss", "loss"))
        gradients = _present_metrics(rows, ("gradient_norm", "grad_norm"))
        numerically_stable = all(math.isfinite(value) for value in losses + gradients)
        severe_loss_increase = (
            len(losses) >= 2
            and losses[-1] > 4.0 * min(losses[:-1])
        )
        finite_gradients = [value for value in gradients if math.isfinite(value)]
        exploding_gradients = bool(
            finite_gradients
            and (
                max(finite_gradients) > 1_000.0
                or (
                    len(finite_gradients) >= 3
                    and finite_gradients[-1]
                    > 100.0 * max(statistics.median(finite_gradients[:-1]), 1e-12)
                )
            )
        )
        if (
            candidate.status.startswith("FAILED")
            or not numerically_stable
            or severe_loss_increase
            or exploding_gradients
        ):
            continue
        viable.append(candidate)
    if rung_epoch == 2 and any(stats[item.candidate_id]["mean_map"] > 0 for item in viable):
        viable = [item for item in viable if stats[item.candidate_id]["mean_map"] > 0]
    ranked = sorted(
        viable,
        key=lambda item: (
            -stats[item.candidate_id]["mean_map"],
            -stats[item.candidate_id]["mean_aptiny"],
            stats[item.candidate_id]["map_standard_deviation"],
            stats[item.candidate_id]["mean_validation_loss"],
            stats[item.candidate_id]["gradient_standard_deviation"],
            item.learning_rate,
        ),
    )
    return ranked[: min(keep, len(ranked))], stats


def boundary_status(selected_lr: float, candidates: Sequence[float]) -> str:
    ordered = sorted(candidates)
    if math.isclose(selected_lr, ordered[0], rel_tol=1e-12):
        return "lowest"
    if math.isclose(selected_lr, ordered[-1], rel_tol=1e-12):
        return "highest"
    if any(math.isclose(selected_lr, value, rel_tol=1e-12) for value in ordered):
        return "interior"
    raise ValueError("selected LR is not one of the candidates")


def boundary_extension_candidates(selected_lr: float, status: str) -> list[float]:
    if status == "lowest":
        return sorted([selected_lr / 4.0, selected_lr / 2.0])
    if status == "highest":
        return sorted([selected_lr * 2.0, selected_lr * 4.0])
    if status == "interior":
        return []
    raise ValueError(f"unsupported boundary status: {status}")


def classify_candidate_failure(error: BaseException) -> str:
    message = str(error).lower()
    if "out of memory" in message or "cuda oom" in message:
        return "FAILED_OOM"
    if any(token in message for token in ("nan", "infinite", "non-finite", "overflow")):
        return "FAILED_NUMERICAL"
    if any(token in message for token in ("environment", "cuda", "mmcv", "mmengine")):
        return "FAILED_ENVIRONMENT"
    if "adapter" in message or "not implemented" in message:
        return "FAILED_ADAPTER"
    # Programming errors are intentionally not converted into candidate failures.
    raise error


def estimate_workload(
    *,
    search_train_epoch_seconds: float,
    search_validation_epoch_seconds: float,
    final_train_epoch_seconds: float,
    final_validation_epoch_seconds: float,
) -> dict[str, float]:
    search_epoch_equivalents = 9 * 2 + 5 * 3 + 3 * 5 + 2 * 5
    search_validation_passes = 9 * 2 + 5 * 3 + 3 * 5 + 2 * 5
    search_seconds = (
        search_epoch_equivalents * search_train_epoch_seconds
        + search_validation_passes * search_validation_epoch_seconds
    )
    final_seconds = 25 * final_train_epoch_seconds + final_validation_epoch_seconds
    calibration_seconds = (
        search_train_epoch_seconds + search_validation_epoch_seconds
    )
    return {
        "seconds_per_search_training_epoch": search_train_epoch_seconds,
        "seconds_per_search_validation_epoch": search_validation_epoch_seconds,
        "seconds_per_final_training_epoch": final_train_epoch_seconds,
        "seconds_per_final_validation_epoch": final_validation_epoch_seconds,
        "search_epoch_equivalents": float(search_epoch_equivalents),
        "calibration_seconds": calibration_seconds,
        "search_seconds": search_seconds,
        "final_seconds": final_seconds,
        "total_seconds": calibration_seconds + search_seconds + final_seconds,
        "total_hours": (
            calibration_seconds + search_seconds + final_seconds
        )
        / 3600.0,
    }


def export_candidate_yaml(
    destination: str | Path,
    *,
    model_id: str,
    baseline: BaselineOptimizerAudit,
    candidates: Sequence[float],
    settings: FixedBenchmarkSettings | None = None,
) -> dict[str, Any]:
    settings = settings or FixedBenchmarkSettings()
    payload = {
        "experiment": {
            "name": EXPERIMENT_NAME,
            "display_name": EXPERIMENT_DISPLAY_NAME,
            "claim": EXPERIMENT_CLAIM,
            "model_id": model_id,
            "dataset_track": settings.dataset_track,
            "search_seed": settings.seed,
        },
        "baseline": asdict(baseline),
        "fixed_settings": asdict(settings),
        "search": {
            "method": "logarithmic_grid_successive_halving",
            "candidates": [float(value) for value in candidates],
            "rungs": [dict(rung) for rung in PROMOTION_RUNGS],
        },
    }
    write_yaml(destination, payload)
    return payload


def export_selected_yaml(
    destination: str | Path,
    *,
    model_id: str,
    baseline: BaselineOptimizerAudit,
    candidates: Sequence[float],
    selected: CandidateResult,
    selection: Mapping[str, float],
    manifest_dir: str | Path,
    git_commit: str,
    environment: Mapping[str, Any],
    settings: FixedBenchmarkSettings | None = None,
) -> dict[str, Any]:
    settings = settings or FixedBenchmarkSettings()
    manifest_dir = Path(manifest_dir)
    status = boundary_status(selected.learning_rate, candidates)
    payload = {
        "experiment": {
            "name": EXPERIMENT_NAME,
            "model_id": model_id,
            "dataset_track": settings.dataset_track,
            "search_seed": settings.seed,
        },
        "baseline": {
            "learning_rate": baseline.learning_rate,
            "baseline_config_path": baseline.baseline_config_path,
            "optimizer": baseline.optimizer_type,
            "weight_decay": baseline.weight_decay,
            "scheduler": baseline.scheduler_type,
            "warmup": baseline.warmup_configuration,
            "pretrained_checkpoint": baseline.pretrained_checkpoint,
            "parameter_groups": baseline.parameter_group_configuration,
        },
        "search_data": {
            "search_train_manifest": str(manifest_dir / "search_train_seed42.json"),
            "search_validation_manifest": str(
                manifest_dir / "search_validation_seed42.json"
            ),
            "search_train_hash": sha256_json(manifest_dir / "search_train_seed42.json"),
            "search_validation_hash": sha256_json(
                manifest_dir / "search_validation_seed42.json"
            ),
        },
        "search": {
            "method": "logarithmic_grid_successive_halving",
            "candidates": [float(value) for value in candidates],
            "rungs": [2, 5, 10, 15],
            "primary_metric": settings.primary_metric,
            "secondary_metric": settings.secondary_metric,
            "selected_learning_rate": selected.learning_rate,
            "selected_candidate_id": selected.candidate_id,
            "selection_statistics": {
                "mean_map_last_three": selection["mean_map"],
                "mean_aptiny_last_three": selection["mean_aptiny"],
                "map_standard_deviation": selection["map_standard_deviation"],
            },
            "boundary_status": status,
        },
        "final_training": {
            "dataset": "complete_official_train",
            "train_manifest": str(manifest_dir / "official_full_train.json"),
            "validation_manifest": str(manifest_dir / "official_validation.json"),
            "restart_from_pretrained": True,
            "learning_rate": selected.learning_rate,
            "epochs": settings.final_epochs,
            "seed": settings.seed,
            "image_size": settings.image_size,
            "effective_batch_size": settings.effective_batch_size,
        },
        "provenance": {
            "git_commit": git_commit,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "gpu": environment.get("gpu_name"),
            "cuda": environment.get("cuda_version"),
            "pytorch": environment.get("pytorch_version"),
            "framework": environment.get("framework"),
        },
    }
    write_yaml(destination, payload)
    return payload
