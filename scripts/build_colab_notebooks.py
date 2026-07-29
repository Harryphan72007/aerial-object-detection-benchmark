#!/usr/bin/env python
"""Build the standalone setup notebook and apply reproducible notebook guards."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def markdown(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str, tags: list[str] | None = None) -> dict[str, Any]:
    metadata = {"tags": tags} if tags else {}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": source.splitlines(True),
    }


def write_notebook(path: Path, cells: list[dict[str, Any]]) -> None:
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:03d}"
    payload = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": path.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


BOOTSTRAP = """\
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

SMOKE_TEST = os.environ.get("SMOKE_TEST", "0").lower() in {"1", "true", "yes", "on"}
try:
    IS_COLAB = importlib.util.find_spec("google.colab") is not None
except ModuleNotFoundError:
    IS_COLAB = False
REPOSITORY_URL = os.environ.get(
    "BENCHMARK_REPOSITORY_URL",
    "https://github.com/Harryphan72007/aerial-object-detection-benchmark.git",
)
REPOSITORY_BRANCH = os.environ.get("BENCHMARK_REPOSITORY_BRANCH", "main")
if IS_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    REPO_DIR = Path("/content/aerial-object-detection-benchmark")
    if not (REPO_DIR / ".git").is_dir():
        subprocess.run(
            ["git", "clone", "--branch", REPOSITORY_BRANCH, REPOSITORY_URL, str(REPO_DIR)],
            check=True,
        )
else:
    REPO_DIR = Path(os.environ.get("BENCHMARK_REPO_ROOT", Path.cwd())).resolve()
if not (REPO_DIR / "pyproject.toml").is_file():
    raise RuntimeError(f"Repository root is invalid: {REPO_DIR}")
os.chdir(REPO_DIR)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if IS_COLAB:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-dataset-colab.txt"],
        check=True,
    )
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)
DRIVE_ROOT = os.environ.get(
    "VISDRONE_DRIVE_ROOT",
    "/content/drive/MyDrive/visdrone_architecture_benchmark"
    if IS_COLAB
    else str(REPO_DIR / ".notebook-smoke"),
)
from src.paths import ProjectPaths
from src.reproducibility import seed_everything
from src.utils.environment import collect_environment
paths = ProjectPaths.from_value(DRIVE_ROOT).create()
seed_everything(42)
print({"repo": str(REPO_DIR), "storage": str(paths.root), "smoke_test": SMOKE_TEST})
collect_environment()
"""


def build_setup_notebook() -> None:
    cells = [
        markdown(
            """# VisDrone2019-DET Colab setup

This standalone notebook prepares the benchmark dataset without installing any model
framework. A normal run persists verified archives, unchanged raw data, Track A
(`person`, `vehicle`) COCO, Track B (official classes 1–10) COCO, manifests,
statistics, validation reports, visualizations, and a DataLoader smoke result.

Expected persistent root:
`/content/drive/MyDrive/visdrone_architecture_benchmark/datasets/VisDrone2019-DET`.
"""
        ),
        markdown("## 1. Runtime diagnostics\n"),
        code(
            """\
import os, platform, shutil, sys
print("Python:", sys.version)
print("Platform:", platform.platform())
try:
    import psutil
    print("RAM GiB:", round(psutil.virtual_memory().total / 2**30, 2))
except ImportError:
    print("RAM: install cell has not run yet")
usage = shutil.disk_usage("/content" if os.path.isdir("/content") else ".")
print("Free disk GiB:", round(usage.free / 2**30, 2))
try:
    import torch
    print("PyTorch:", torch.__version__)
    print("torchvision:", __import__("torchvision").__version__)
    print("CUDA build:", torch.version.cuda, "available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print("GPU:", props.name, "VRAM GiB:", round(props.total_memory / 2**30, 2))
except ImportError:
    print("PyTorch: not installed (dataset conversion is CPU-compatible)")
"""
        ),
        markdown("## 2. User configuration\n\nEdit this one cell before running the notebook.\n"),
        code(
            """\
import os
REPOSITORY_URL = "https://github.com/Harryphan72007/aerial-object-detection-benchmark.git"
REPOSITORY_BRANCH = "main"
USE_GOOGLE_DRIVE = True
DATASET_SOURCE = "auto"       # auto | manual | kaggle
KAGGLE_DATASET_HANDLE = ""    # required only when DATASET_SOURCE == "kaggle"
REDOWNLOAD = False
TRACKS = ("2class", "10class")  # choose either or both
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0").lower() in {"1", "true", "yes", "on"}
SMOKE_TEST_SUBSET_SIZE = int(os.environ.get("SMOKE_TEST_SUBSET_SIZE", "8"))
RANDOM_SEED = 42
assert set(TRACKS) <= {"2class", "10class"} and TRACKS
assert DATASET_SOURCE in {"auto", "manual", "kaggle"}
"""
        ),
        markdown("## 3. Optional Google Drive mount\n"),
        code(
            """\
import importlib.util
from pathlib import Path
try:
    IS_COLAB = importlib.util.find_spec("google.colab") is not None
except ModuleNotFoundError:
    IS_COLAB = False
if USE_GOOGLE_DRIVE:
    if not IS_COLAB:
        print("Google Drive mount skipped outside Colab; using VISDRONE_DRIVE_ROOT.")
    else:
        from google.colab import drive
        drive.mount("/content/drive")
STORAGE_ROOT = Path(
    os.environ.get(
        "VISDRONE_DRIVE_ROOT",
        "/content/drive/MyDrive/visdrone_architecture_benchmark"
        if USE_GOOGLE_DRIVE and IS_COLAB
        else "/content/visdrone_architecture_benchmark",
    )
)
print("Persistent storage root:", STORAGE_ROOT)
"""
        ),
        markdown("## 4. Clone or detect the repository\n"),
        code(
            """\
import subprocess, sys
if IS_COLAB:
    REPO_DIR = Path("/content/aerial-object-detection-benchmark")
    if REPO_DIR.exists() and not (REPO_DIR / ".git").is_dir():
        raise RuntimeError(f"Refusing to overwrite non-Git directory: {REPO_DIR}")
    if not (REPO_DIR / ".git").is_dir():
        subprocess.run(
            ["git", "clone", "--branch", REPOSITORY_BRANCH, REPOSITORY_URL, str(REPO_DIR)],
            check=True,
        )
else:
    REPO_DIR = Path(os.environ.get("BENCHMARK_REPO_ROOT", Path.cwd())).resolve()
if not (REPO_DIR / "pyproject.toml").is_file():
    raise RuntimeError(f"Repository root not found: {REPO_DIR}")
os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR))
print("Repository:", REPO_DIR)
print("Commit:", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
"""
        ),
        markdown("## 5. Install dataset-pipeline dependencies only\n"),
        code(
            """\
if IS_COLAB:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-dataset-colab.txt"],
        check=True,
    )
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)
else:
    print("Local execution: using the active environment; no packages were changed.")
from src.paths import ProjectPaths
paths = ProjectPaths.from_value(STORAGE_ROOT).create()
print("Dataset root:", paths.visdrone)
"""
        ),
        markdown(
            """## 6. Acquire dataset archives

Order: validated cache → verified token-free GitHub release asset → Kaggle only
when explicitly selected and authenticated → interactive manual upload. No
credential is embedded or printed. Interrupted HTTP downloads resume from `.part`.
"""
        ),
        code(
            """\
import shutil
from src.data.download import VISDRONE_ARCHIVES, ensure_archive
archive_manifests = {}
if SMOKE_TEST:
    from src.data.smoke_dataset import create_smoke_archives
    create_smoke_archives(paths.archives, SMOKE_TEST_SUBSET_SIZE)
    print("Created deterministic local smoke archives.")
elif DATASET_SOURCE == "kaggle":
    if not KAGGLE_DATASET_HANDLE:
        raise RuntimeError(
            "Set KAGGLE_DATASET_HANDLE to a verified dataset you are authorized to use. "
            "Do not paste credentials into this notebook."
        )
    try:
        import kagglehub
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub==0.3.12"], check=True)
        import kagglehub
    downloaded_root = Path(kagglehub.dataset_download(KAGGLE_DATASET_HANDLE))
    for spec in VISDRONE_ARCHIVES.values():
        matches = list(downloaded_root.rglob(str(spec["filename"])))
        if len(matches) != 1:
            raise RuntimeError(
                f"Kaggle download must contain exactly one {spec['filename']}; found {len(matches)}"
            )
        shutil.copy2(matches[0], paths.archives / str(spec["filename"]))
elif DATASET_SOURCE == "manual":
    print("Manual archives requested.")
else:
    try:
        for split in ("train", "val"):
            archive_manifests[split] = ensure_archive(
                split, paths.archives, paths.dataset_manifests, redownload=REDOWNLOAD
            )
            print(archive_manifests[split])
    except Exception as error:
        print("Automatic download failed:", type(error).__name__, error)
        DATASET_SOURCE = "manual"

if DATASET_SOURCE == "manual" and not SMOKE_TEST:
    expected = {str(spec["filename"]) for spec in VISDRONE_ARCHIVES.values()}
    present = {path.name for path in paths.archives.glob("*.zip")}
    missing = expected - present
    if missing and not IS_COLAB:
        raise RuntimeError(f"Copy these archives into {paths.archives}: {sorted(missing)}")
    if missing:
        from google.colab import files
        print("Upload the official archives:", sorted(missing))
        uploaded = files.upload()
        for name, content in uploaded.items():
            if name in expected:
                (paths.archives / name).write_bytes(content)
"""
        ),
        markdown("## 7. Archive integrity and manifest verification\n"),
        code(
            """\
import json
from datetime import datetime, timezone
from src.data.download import sha256_file, validate_zip
for split, spec in VISDRONE_ARCHIVES.items():
    archive = paths.archives / str(spec["filename"])
    validate_zip(
        archive,
        str(spec["folder"]),
        1 if SMOKE_TEST else int(spec["minimum_bytes"]),
    )
    manifest = {
        "split": split,
        "source_url": "synthetic://smoke-test" if SMOKE_TEST else str(spec["url"]),
        "archive_path": str(archive),
        "size_bytes": archive.stat().st_size,
        "sha256": sha256_file(archive),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = paths.dataset_manifests / f"{split}_archive.json"
    if manifest_path.exists() and not REDOWNLOAD and not SMOKE_TEST:
        previous = json.loads(manifest_path.read_text())
        if previous.get("sha256") != manifest["sha256"]:
            raise RuntimeError(f"Checksum changed for cached archive: {archive}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\\n")
    print(json.dumps(manifest, indent=2))
"""
        ),
        markdown("## 8. Idempotent extraction (raw data is never modified)\n"),
        code(
            """\
from src.data.download import extract_idempotent
for spec in VISDRONE_ARCHIVES.values():
    destination = extract_idempotent(
        paths.archives / str(spec["filename"]), paths.raw, str(spec["folder"])
    )
    print("Ready:", destination)
"""
        ),
        markdown("## 9. Parse and validate the official eight-column annotation rows\n"),
        code(
            """\
from src.data.convert_visdrone import parse_visdrone_line
example_annotation = next((paths.raw / "VisDrone2019-DET-train/annotations").glob("*.txt"))
example_row = next(row for row in example_annotation.read_text().splitlines() if row.strip())
parsed = parse_visdrone_line(example_row)
print(
    dict(
        zip(
            ["bbox_left", "bbox_top", "bbox_width", "bbox_height", "score",
             "object_category", "truncation", "occlusion"],
            parsed,
        )
    )
)
"""
        ),
        markdown("## 10. Convert Track A and Track B to deterministic COCO\n"),
        code(
            """\
prepare_command = [
    sys.executable, "-m", "scripts.prepare_data",
    "--drive-root", str(paths.root),
    "--tracks", *TRACKS,
    "--validate",
]
if SMOKE_TEST:
    prepare_command.extend(["--max-images-per-split", str(SMOKE_TEST_SUBSET_SIZE)])
subprocess.run(prepare_command, check=True)
"""
        ),
        markdown("## 11. Validation reports and dataset statistics\n"),
        code(
            """\
from src.data.statistics import compute_statistics
from src.data.validate_annotations import validate_coco
all_statistics = {}
for track in TRACKS:
    for split in ("train", "val"):
        annotation_file = paths.coco(track) / "annotations" / f"instances_{split}.json"
        image_dir = paths.coco(track) / split
        report = validate_coco(annotation_file, image_dir)
        report.raise_for_errors()
        stats = compute_statistics(annotation_file)
        all_statistics[f"{track}/{split}"] = stats
        print(track, split, json.dumps(stats, indent=2))
"""
        ),
        markdown(
            "## 12. Visualize six annotated images\n\n"
            "Selection is deterministic and favors crowded, tiny, class-diverse, occluded, and truncated examples.\n"
        ),
        code(
            """\
import matplotlib.pyplot as plt
from PIL import ImageDraw
from src.data.dataloaders import CocoDetectionRecords
visual_track = "10class" if "10class" in TRACKS else "2class"
records = CocoDetectionRecords(
    paths.coco(visual_track) / "train",
    paths.coco(visual_track) / "annotations/instances_train.json",
)
def interest(record):
    annotations = record["annotations"]
    tiny = sum(float(item["area"]) < 32**2 for item in annotations)
    classes = len({int(item["category_id"]) for item in annotations})
    attributes = [item.get("attributes", {}) for item in annotations]
    difficult = sum(
        int(item.get("occlusion", 0)) > 0 or int(item.get("truncation", 0)) > 0
        for item in attributes
    )
    return (len(annotations), tiny, classes, difficult, -record["image_id"])
selected = sorted((records[index] for index in range(len(records))), key=interest, reverse=True)[:6]
if len(selected) < 6:
    raise RuntimeError("At least six converted images are required for visualization")
figure, axes = plt.subplots(2, 3, figsize=(18, 8))
for axis, record in zip(axes.flat, selected):
    canvas = record["image"].copy()
    draw = ImageDraw.Draw(canvas)
    for annotation in record["annotations"]:
        x, y, width, height = annotation["bbox"]
        label = records.categories[int(annotation["category_id"])]
        draw.rectangle((x, y, x + width, y + height), outline="red", width=2)
        draw.text((x, y), label, fill="yellow")
    axis.imshow(canvas)
    axis.set_title(f"{record['file_name']} ({len(record['annotations'])} objects)")
    axis.axis("off")
plt.tight_layout()
plt.show()
"""
        ),
        markdown("## 13. DataLoader smoke test\n"),
        code(
            """\
from src.data.dataloaders import detection_collate
try:
    from torch.utils.data import DataLoader
    loader = DataLoader(records, batch_size=min(2, len(records)), shuffle=False, collate_fn=detection_collate)
    batch = next(iter(loader))
except ImportError:
    print("PyTorch is absent; using the identical collate path for this local CPU verification.")
    batch = detection_collate([records[index] for index in range(min(2, len(records)))])
assert batch and all(item["image"].mode == "RGB" and item["original_size"] for item in batch)
print("Valid batch:", [(item["image_id"], item["original_size"], len(item["annotations"])) for item in batch])
"""
        ),
        markdown("## 14. Final summary and next notebook\n"),
        code(
            """\
print("VisDrone setup complete.")
print("Persistent root:", paths.root)
print("Raw data:", paths.raw)
for track in TRACKS:
    print(track, "COCO:", paths.coco(track))
print("Manifests:", paths.dataset_manifests)
print("Next notebook: notebooks/01_dataset_analysis.ipynb")
"""
        ),
    ]
    write_notebook(NOTEBOOKS / "00_visdrone_colab_setup.ipynb", cells)


def build_lr_workflow_notebooks() -> None:
    search_cells = [
        markdown(
            """# VisDrone learning-rate search

This shared notebook runs the same LR-only protocol for one supported primary
model. Search manifests are drawn exclusively from official train. Search
checkpoints are isolated and are never registered as final benchmark runs.
"""
        ),
        code(BOOTSTRAP, ["bootstrap"]),
        markdown("## Configuration\n\nChange `MODEL_ID` for the model-day being run.\n"),
        code(
            """\
MODEL_ID = "rtdetrv2_l"
START_EXPENSIVE_STAGE = False
RUN_LR_RANGE_TEST = True
RUN_BOUNDARY_EXTENSION = False
ALLOW_OVER_BUDGET_RUN = False
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4

# Locked benchmark controls; do not edit for comparable runs.
DATASET_TRACK = "2class"
SEARCH_SEED = 42
IMAGE_SIZE = 640
EFFECTIVE_BATCH_SIZE = 8
SEARCH_MAX_EPOCHS = 15
if SMOKE_TEST:
    START_EXPENSIVE_STAGE = False
assert DATASET_TRACK == "2class"
assert EFFECTIVE_BATCH_SIZE == PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
"""
        ),
        markdown("## Install and validate the selected model environment\n"),
        code(
            """\
if IS_COLAB and not SMOKE_TEST:
    requirement = (
        "requirements-rtdetr-colab.txt"
        if MODEL_ID == "rtdetrv2_l"
        else "requirements-openmmlab-py310-cu118.txt"
    )
    if MODEL_ID != "rtdetrv2_l" and sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            "OpenMMLab models require the documented Python 3.10 custom/local "
            "Colab runtime; the current hosted runtime is not supported."
        )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", requirement],
        check=True,
    )
if MODEL_ID != "rtdetrv2_l":
    upstream = Path("/content/VMamba" if MODEL_ID == "faster_rcnn_vmamba_t" else "/content/mmdetection")
    if IS_COLAB and not upstream.joinpath(".git").is_dir() and not SMOKE_TEST:
        url = (
            "https://github.com/MzeroMiko/VMamba.git"
            if MODEL_ID == "faster_rcnn_vmamba_t"
            else "https://github.com/open-mmlab/mmdetection.git"
        )
        clone_command = ["git", "clone"]
        if MODEL_ID != "faster_rcnn_vmamba_t":
            clone_command.extend(["--depth", "1", "--branch", "v3.3.0"])
        clone_command.extend([url, str(upstream)])
        subprocess.run(clone_command, check=True)
    if MODEL_ID == "faster_rcnn_vmamba_t":
        if upstream.joinpath(".git").is_dir():
            subprocess.run(
                ["git", "-C", str(upstream), "checkout",
                 "2ed52ead062a51a64521ed3871d52914bf532876"],
                check=True,
            )
        os.environ["VMAMBA_ROOT"] = str(upstream)
        pretrained = paths.pretrained / "vmamba_t.pth"
        if not pretrained.is_file() and not SMOKE_TEST:
            raise FileNotFoundError(
                f"Place the verified official VMamba-T checkpoint at {pretrained}"
            )
        os.environ["VMAMBA_T_PRETRAINED"] = str(pretrained)
        if not SMOKE_TEST:
            try:
                import selective_scan_cuda
            except ImportError:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install",
                     str(upstream / "kernels" / "selective_scan"),
                     "--no-build-isolation"],
                    check=True,
                )
            import selective_scan_cuda
    else:
        os.environ["MMDET_ROOT"] = str(upstream)
print("Environment selected for:", MODEL_ID)
print("If pip changed core packages, restart the runtime once, then rerun from the top.")
"""
        ),
        markdown("## Validate manifests, model identity, and baseline optimizer\n"),
        code(
            """\
from src.models.registry import create_adapter, load_model_config
from src.training.lr_search import (
    SUPPORTED_PRIMARY_MODELS,
    generate_lr_candidates,
)
from src.training.lr_workflow import LRControlledBenchmark
from src.benchmark_status import discover_model_status, format_preflight_summary
from src.utils.serialization import read_json

assert MODEL_ID in SUPPORTED_PRIMARY_MODELS
workflow = LRControlledBenchmark(REPO_DIR, DRIVE_ROOT)
split_summary = workflow.prepare_manifests()
model_config = load_model_config(MODEL_ID, REPO_DIR)
adapter = create_adapter(MODEL_ID, "cpu")
baseline = workflow.resolve_baseline(MODEL_ID)
default_candidates = generate_lr_candidates(baseline.learning_rate)
print("Framework:", model_config["framework"])
print("Adapter:", type(adapter).__name__)
print("Baseline audit:", baseline)
print("Search split checks:", split_summary["verification"])
print("Default LR candidates:", default_candidates)
status = discover_model_status(DRIVE_ROOT, MODEL_ID, REPO_DIR)
calibration_path = paths.lr_search_checkpoints / MODEL_ID / "calibration.json"
estimate = (
    workflow.workload_estimate(
        read_json(calibration_path),
        range_optimizer_steps=300 if RUN_LR_RANGE_TEST else 0,
        batch_size=PER_DEVICE_BATCH_SIZE,
        accumulation=GRADIENT_ACCUMULATION_STEPS,
    )
    if calibration_path.exists()
    else None
)
train_stats = split_summary["statistics"]["search_train_seed42.json"]
val_stats = split_summary["statistics"]["search_validation_seed42.json"]
try:
    import torch
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NOT DETECTED"
except ImportError:
    gpu_name = "NOT DETECTED"
print(format_preflight_summary({
    "model": MODEL_ID,
    "dataset_track": DATASET_TRACK,
    "mode": "LR SEARCH",
    "train_manifest": workflow.manifest_dir / "search_train_seed42.json",
    "validation_manifest": workflow.manifest_dir / "search_validation_seed42.json",
    "training_images": train_stats["images"],
    "validation_images": val_stats["images"],
    "full_official_train": "NO (official-train subset only)",
    "image_size": IMAGE_SIZE,
    "batch_size": PER_DEVICE_BATCH_SIZE,
    "gradient_accumulation": GRADIENT_ACCUMULATION_STEPS,
    "effective_batch_size": EFFECTIVE_BATCH_SIZE,
    "learning_rate": f"{len(default_candidates)} candidates around {baseline.learning_rate:.6g}",
    "epoch_budget": "successive-halving rungs 2/5/10/15",
    "gpu": gpu_name,
    "estimated_runtime": f"{estimate['total_hours']:.2f} h" if estimate else "calculated after one-epoch calibration",
    "output_directory": paths.lr_search_checkpoints / MODEL_ID,
    "resume_status": (
        f"resume after rungs {status['search_completed_rungs']}"
        if status["lr_search_status"] == "IN_PROGRESS"
        else status["lr_search_status"]
    ),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
}))
"""
        ),
        markdown("## Workload contract\n"),
        code(
            """\
SEARCH_EPOCH_EQUIVALENTS = 9 * 2 + 5 * 3 + 3 * 5 + 2 * 5
print("Search train epoch-equivalents:", SEARCH_EPOCH_EQUIVALENTS)
print("Search validation passes:", SEARCH_EPOCH_EQUIVALENTS)
calibration_path = paths.lr_search_checkpoints / MODEL_ID / "calibration.json"
if calibration_path.exists():
    import json
    calibration = json.loads(calibration_path.read_text())
    estimate = workflow.workload_estimate(
        calibration,
        range_optimizer_steps=300 if RUN_LR_RANGE_TEST else 0,
        batch_size=PER_DEVICE_BATCH_SIZE,
        accumulation=GRADIENT_ACCUMULATION_STEPS,
    )
    print("Measured workload estimate:", estimate)
    if estimate["total_hours"] > 24:
        print("WARNING: estimated protocol exceeds 24 hours.")
        print("Set ALLOW_OVER_BUDGET_RUN=True to opt in explicitly.")
else:
    print("A one-epoch calibration will run before the search starts.")
"""
        ),
        markdown("## Run or resume successive halving\n"),
        code(
            """\
if START_EXPENSIVE_STAGE:
    from src.notebook_utils import require_gpu, require_model_environment
    require_model_environment(
        "rtdetr" if MODEL_ID == "rtdetrv2_l" else "openmmlab"
    )
    require_gpu(MODEL_ID)
    search_result = workflow.run_search(
        MODEL_ID,
        batch_size=PER_DEVICE_BATCH_SIZE,
        accumulation=GRADIENT_ACCUMULATION_STEPS,
        run_lr_range_test=RUN_LR_RANGE_TEST,
        run_boundary_extension=RUN_BOUNDARY_EXTENSION,
        allow_over_budget_run=ALLOW_OVER_BUDGET_RUN,
    )
    print("Promotion decisions:")
    for decision in search_result["state"]["rung_decisions"]:
        print(decision)
    print("Selected:", search_result["selected"])
    selected_path = workflow.persistent_config_dir / f"{MODEL_ID}_2class_selected.yaml"
    summary_path = workflow.persistent_config_dir / f"{MODEL_ID}_2class_search_summary.json"
    print("\\nLR SEARCH COMPLETE")
    print("\\nModel:", MODEL_ID)
    print("Selected learning rate:", search_result["selected"]["selected_learning_rate"])
    print("Selected configuration:", selected_path)
    print("Search summary:", summary_path)
    print("Candidate ranking:", paths.lr_search_checkpoints / MODEL_ID / "search_state.json")
    print("Next notebook:", REPO_DIR / "notebooks" / "13_full_dataset_finetune.ipynb")
else:
    print("Expensive stage is OFF. Validation and candidate preview completed.")
    print("Set START_EXPENSIVE_STAGE=True only for the selected model-day.")
"""
        ),
        markdown(
            """## Output

The selected YAML is written to `configs/lr_search/` and copied to persistent
storage. A boundary winner is reported as a finite-range warning, not as a
global optimum.
"""
        ),
    ]
    write_notebook(
        NOTEBOOKS / "12_learning_rate_search.ipynb", search_cells
    )

    final_cells = [
        markdown(
            """# Complete-official-train fine-tuning

This notebook consumes one selected LR, reloads the original pretrained model,
trains on every official training image for 25 epochs, and then invokes the
common repository evaluator once on complete official validation.
"""
        ),
        code(BOOTSTRAP, ["bootstrap"]),
        markdown("## Configuration\n"),
        code(
            """\
MODEL_ID = "rtdetrv2_l"
START_EXPENSIVE_STAGE = False
ALLOW_OVER_BUDGET_RUN = False
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4

# Locked benchmark controls; do not edit for comparable runs.
FINAL_EPOCHS = 25
FINAL_SEED = 42
if SMOKE_TEST:
    START_EXPENSIVE_STAGE = False
assert FINAL_EPOCHS == 25 and FINAL_SEED == 42
assert PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS == 8
"""
        ),
        markdown("## Install and validate the selected model environment\n"),
        code(
            """\
if IS_COLAB and not SMOKE_TEST:
    requirement = (
        "requirements-rtdetr-colab.txt"
        if MODEL_ID == "rtdetrv2_l"
        else "requirements-openmmlab-py310-cu118.txt"
    )
    if MODEL_ID != "rtdetrv2_l" and sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            "OpenMMLab models require the documented Python 3.10 custom/local "
            "Colab runtime; the current hosted runtime is not supported."
        )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", requirement],
        check=True,
    )
if MODEL_ID != "rtdetrv2_l":
    upstream = Path("/content/VMamba" if MODEL_ID == "faster_rcnn_vmamba_t" else "/content/mmdetection")
    if IS_COLAB and not upstream.joinpath(".git").is_dir() and not SMOKE_TEST:
        url = (
            "https://github.com/MzeroMiko/VMamba.git"
            if MODEL_ID == "faster_rcnn_vmamba_t"
            else "https://github.com/open-mmlab/mmdetection.git"
        )
        clone_command = ["git", "clone"]
        if MODEL_ID != "faster_rcnn_vmamba_t":
            clone_command.extend(["--depth", "1", "--branch", "v3.3.0"])
        clone_command.extend([url, str(upstream)])
        subprocess.run(clone_command, check=True)
    os.environ[
        "VMAMBA_ROOT" if MODEL_ID == "faster_rcnn_vmamba_t" else "MMDET_ROOT"
    ] = str(upstream)
    if MODEL_ID == "faster_rcnn_vmamba_t":
        subprocess.run(
            ["git", "-C", str(upstream), "checkout",
             "2ed52ead062a51a64521ed3871d52914bf532876"],
            check=True,
        )
        pretrained = paths.pretrained / "vmamba_t.pth"
        if not pretrained.is_file() and not SMOKE_TEST:
            raise FileNotFoundError(
                f"Place the verified official VMamba-T checkpoint at {pretrained}"
            )
        os.environ["VMAMBA_T_PRETRAINED"] = str(pretrained)
        if not SMOKE_TEST:
            try:
                import selective_scan_cuda
            except ImportError:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install",
                     str(upstream / "kernels" / "selective_scan"),
                     "--no-build-isolation"],
                    check=True,
                )
            import selective_scan_cuda
print("Environment selected for:", MODEL_ID)
"""
        ),
        markdown("## Load and validate the selected configuration\n"),
        code(
            """\
import json
from src.training.lr_workflow import LRControlledBenchmark
from src.utils.serialization import read_yaml
from src.benchmark_status import (
    discover_model_status,
    find_resumable_final_run,
    find_selected_config,
    format_preflight_summary,
)

workflow = LRControlledBenchmark(REPO_DIR, DRIVE_ROOT)
workflow.prepare_manifests()
selected_path = find_selected_config(DRIVE_ROOT, MODEL_ID, REPO_DIR)
if selected_path is None:
    if SMOKE_TEST:
        print("No selected YAML in smoke mode; run notebook 12 on a GPU first.")
        selected = None
    else:
        raise FileNotFoundError(
            f"Selected LR YAML for {MODEL_ID} is missing. Complete notebook 12 first."
        )
else:
    selected = read_yaml(selected_path)
    assert selected["experiment"]["model_id"] == MODEL_ID
    assert selected["final_training"]["dataset"] == "complete_official_train"
    assert selected["final_training"]["restart_from_pretrained"] is True
    print(selected)
"""
        ),
        markdown("## Prove complete-train identity and validation exclusion\n"),
        code(
            """\
from src.training.lr_search import assert_final_training_uses_official_train
assert_final_training_uses_official_train(workflow.manifest_dir)
summary = json.loads((workflow.manifest_dir / "split_summary.json").read_text())
print("Complete official train proof:", summary["sources"]["official_train"])
print("Split checks:", summary["verification"])
print("FULL OFFICIAL TRAINING SPLIT VERIFIED: YES")
if selected is not None:
    final = selected["final_training"]
    resumable = find_resumable_final_run(
        DRIVE_ROOT,
        MODEL_ID,
        selected_learning_rate=float(final["learning_rate"]),
    )
    calibration_path = paths.lr_search_checkpoints / MODEL_ID / "calibration.json"
    estimate = (
        workflow.workload_estimate(json.loads(calibration_path.read_text()))
        if calibration_path.exists()
        else None
    )
    try:
        import torch
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NOT DETECTED"
    except ImportError:
        gpu_name = "NOT DETECTED"
    print(format_preflight_summary({
        "model": MODEL_ID,
        "dataset_track": "2class",
        "mode": "FINAL FINE-TUNING",
        "train_manifest": workflow.manifest_dir / "official_full_train.json",
        "validation_manifest": workflow.manifest_dir / "official_validation.json",
        "training_images": summary["statistics"]["official_full_train.json"]["images"],
        "validation_images": summary["statistics"]["official_validation.json"]["images"],
        "full_official_train": "YES",
        "image_size": 640,
        "batch_size": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": 8,
        "learning_rate": final["learning_rate"],
        "epoch_budget": 25,
        "gpu": gpu_name,
        "estimated_runtime": f"{estimate['final_seconds'] / 3600:.2f} h" if estimate else "calculated after calibration",
        "output_directory": (
            resumable["run_dir"] if resumable
            else paths.final_checkpoints / MODEL_ID / "<new-run-id>"
        ),
        "resume_status": (
            f"AUTO-RESUME {resumable['run_id']}" if resumable else "NEW RUN"
        ),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }))
"""
        ),
        markdown("## Restart from pretrained and run final benchmark\n"),
        code(
            """\
if START_EXPENSIVE_STAGE:
    if selected is None:
        raise RuntimeError("A selected LR configuration is required.")
    from src.notebook_utils import require_gpu, require_model_environment
    require_model_environment(
        "rtdetr" if MODEL_ID == "rtdetrv2_l" else "openmmlab"
    )
    require_gpu(MODEL_ID)
    final_manifest = workflow.run_final_training(
        MODEL_ID,
        selected_path,
        batch_size=PER_DEVICE_BATCH_SIZE,
        accumulation=GRADIENT_ACCUMULATION_STEPS,
        allow_over_budget_run=ALLOW_OVER_BUDGET_RUN,
        run_common_evaluation=True,
    )
    print("Final registered run:", final_manifest["run_id"])
    print("Output directory:", final_manifest["run_dir"])
    print("Evaluation:", final_manifest.get("final_evaluation_metrics"))
    print("\\nFINAL TRAINING COMPLETE")
    print("\\nModel:", MODEL_ID)
    print("Run ID:", final_manifest["run_id"])
    print("Best checkpoint:", final_manifest["checkpoint_best_map"])
    print("Last checkpoint:", final_manifest["checkpoint_last"])
    print("Best mAP50-95:", final_manifest.get("best_validation_map"))
    print("Best APtiny:", final_manifest.get("best_validation_aptiny"))
    print("Training time:", final_manifest.get("total_training_seconds"), "seconds")
    print(
        "Evaluation command:",
        f'python scripts/evaluate.py --drive-root "{DRIVE_ROOT}" '
        f'--dataset-track 2class --split val --run-id "{final_manifest["run_id"]}" '
        "--resolutions 640",
    )
    print("Next notebook:", REPO_DIR / "notebooks" / "07_evaluate_all_models.ipynb")
else:
    print("Expensive stage is OFF; no model weights were changed.")
"""
        ),
    ]
    write_notebook(
        NOTEBOOKS / "13_full_dataset_finetune.ipynb", final_cells
    )


def patch_existing_notebooks() -> None:
    paths = sorted(NOTEBOOKS.glob("*.ipynb"))
    for path in paths:
        if path.name == "00_visdrone_colab_setup.ipynb":
            continue
        notebook = json.loads(path.read_text(encoding="utf-8"))
        first_code = next(i for i, cell in enumerate(notebook["cells"]) if cell["cell_type"] == "code")
        notebook["cells"][first_code] = code(BOOTSTRAP, ["bootstrap"])
        notebook.setdefault("metadata", {}).setdefault("kernelspec", {
            "display_name": "Python 3", "language": "python", "name": "python3"
        })
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
        name = path.name
        if name in {
            "02_train_resnet50_faster_rcnn.ipynb",
            "03_train_swin_t_faster_rcnn.ipynb",
            "04_train_vmamba_t_faster_rcnn.ipynb",
            "05_train_rtdetrv2_l.ipynb",
        }:
            config_cell = notebook["cells"][4]
            config_source = "".join(config_cell["source"])
            config_source = config_source.replace(
                "RUN_HYPERPARAMETER_SEARCH = False\n", ""
            )
            if "SMOKE_TEST stops before model construction/training" not in config_source:
                config_source += (
                    "\nif SMOKE_TEST:\n"
                    "    NUM_EPOCHS = 1\n"
                    "    BATCH_SIZE = 1\n"
                    "print('SMOKE_TEST stops before model construction/training:', SMOKE_TEST)\n"
                )
            config_cell["source"] = config_source.splitlines(True)
            notebook["cells"][6] = code(
                """\
from src.notebook_utils import preflight_dataset
report = preflight_dataset(paths, DATASET_TRACK, minimum_free_gb=0 if SMOKE_TEST else 5)
print(report)
report.raise_for_errors()
"""
            )
            notebook["cells"][9] = code(
                """\
import shlex
cmd = f"python scripts/train.py --drive-root '{DRIVE_ROOT}' --model-id {MODEL_ID} --dataset-track {DATASET_TRACK} --image-size {IMAGE_SIZE} --batch-size {BATCH_SIZE} --gradient-accumulation-steps {GRADIENT_ACCUMULATION_STEPS} --epochs {NUM_EPOCHS} --seed {SEED}"
if not USE_AMP:
    cmd += " --no-amp"
if RESUME_RUN_ID:
    cmd += f" --resume-run-id {RESUME_RUN_ID}"
print(cmd)
if SMOKE_TEST:
    print("SMOKE_TEST: command validated; expensive training not started.")
else:
    from src.notebook_utils import require_gpu, require_model_environment
    require_model_environment("rtdetr" if MODEL_ID == "rtdetrv2_l" else "openmmlab")
    require_gpu(MODEL_ID)
    subprocess.run(shlex.split(cmd), check=True)
"""
            )
            for index, cell in enumerate(notebook["cells"]):
                source = "".join(cell.get("source", []))
                if "## Optional Optuna search" in source:
                    notebook["cells"][index] = markdown(
                        "## Learning-rate selection\n\n"
                        "Use the shared `12_learning_rate_search.ipynb`, then "
                        "`13_full_dataset_finetune.ipynb`. The legacy "
                        "multidimensional Optuna search has been removed.\n"
                    )
                    if index + 1 < len(notebook["cells"]):
                        notebook["cells"][index + 1] = code(
                            "print('LR search: notebooks/12_learning_rate_search.ipynb')\n"
                            "print('Final training: notebooks/13_full_dataset_finetune.ipynb')\n"
                        )
            if name in {
                "02_train_resnet50_faster_rcnn.ipynb",
                "03_train_swin_t_faster_rcnn.ipynb",
            }:
                notebook["cells"][first_code]["source"].extend(
                    """\
if not SMOKE_TEST:
    MMDET_ROOT = Path(os.environ.get("MMDET_ROOT", "/content/mmdetection"))
    if not (MMDET_ROOT / ".git").is_dir():
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", "v3.3.0",
             "https://github.com/open-mmlab/mmdetection.git", str(MMDET_ROOT)],
            check=True,
        )
    os.environ["MMDET_ROOT"] = str(MMDET_ROOT)
""".splitlines(True)
                )
            elif name == "04_train_vmamba_t_faster_rcnn.ipynb":
                notebook["cells"][first_code]["source"].extend(
                    """\
if not SMOKE_TEST:
    VMAMBA_ROOT = Path(os.environ.get("VMAMBA_ROOT", "/content/VMamba"))
    VMAMBA_COMMIT = "2ed52ead062a51a64521ed3871d52914bf532876"
    if not (VMAMBA_ROOT / ".git").is_dir():
        subprocess.run(["git", "clone", "https://github.com/MzeroMiko/VMamba.git",
                        str(VMAMBA_ROOT)], check=True)
    subprocess.run(["git", "-C", str(VMAMBA_ROOT), "checkout", VMAMBA_COMMIT], check=True)
    os.environ["VMAMBA_ROOT"] = str(VMAMBA_ROOT)
    try:
        import selective_scan_cuda
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install",
             str(VMAMBA_ROOT / "kernels/selective_scan"), "--no-build-isolation"],
            check=True,
        )
""".splitlines(True)
                )
            elif name == "05_train_rtdetrv2_l.ipynb":
                notebook["cells"][first_code]["source"].extend(
                    """\
if IS_COLAB and not SMOKE_TEST:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r",
         "requirements-rtdetr-colab.txt"],
        check=True,
    )
""".splitlines(True)
                )
        elif name == "06_train_yolox_s_optional.ipynb":
            notebook["cells"][6] = code(
                """\
from src.notebook_utils import preflight_dataset
report = preflight_dataset(paths, DATASET_TRACK, minimum_free_gb=0 if SMOKE_TEST else 5)
print(report)
report.raise_for_errors()
"""
            )
            notebook["cells"][8] = code(
                """\
if SMOKE_TEST:
    print("SMOKE_TEST: optional YOLOX clone and integration are not executed.")
else:
    raise RuntimeError(
        "YOLOX-S is an optional control and its adapter is not implemented. "
        "Do not treat this notebook as a completed training path."
    )
"""
            )
        elif name == "07_evaluate_all_models.ipynb":
            notebook["cells"] = [
                markdown(
                    "# Evaluate one final benchmark run\n\n"
                    "Use the same model environment as training. The notebook discovers the "
                    "latest compatible completed final run and evaluates official validation.\n"
                ),
                code(BOOTSTRAP, ["bootstrap"]),
                markdown("## Configuration\n"),
                code(
                    """\
MODEL_ID = "rtdetrv2_l"
DATASET_TRACK = "2class"
RUN_ID = ""  # Leave blank unless the discovery table shows multiple compatible runs.
EVALUATION_RESOLUTION = 640
"""
                ),
                markdown("## Restore and validate the selected model environment\n"),
                code(
                    """\
if IS_COLAB and not SMOKE_TEST:
    requirement = (
        "requirements-rtdetr-colab.txt"
        if MODEL_ID == "rtdetrv2_l"
        else "requirements-openmmlab-py310-cu118.txt"
    )
    if MODEL_ID != "rtdetrv2_l" and sys.version_info[:2] != (3, 10):
        raise RuntimeError("Use the documented Python 3.10 custom/local runtime.")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", requirement],
        check=True,
    )
if MODEL_ID != "rtdetrv2_l":
    upstream = Path("/content/VMamba" if MODEL_ID == "faster_rcnn_vmamba_t" else "/content/mmdetection")
    if IS_COLAB and not upstream.joinpath(".git").is_dir() and not SMOKE_TEST:
        url = (
            "https://github.com/MzeroMiko/VMamba.git"
            if MODEL_ID == "faster_rcnn_vmamba_t"
            else "https://github.com/open-mmlab/mmdetection.git"
        )
        clone_command = ["git", "clone"]
        if MODEL_ID != "faster_rcnn_vmamba_t":
            clone_command.extend(["--depth", "1", "--branch", "v3.3.0"])
        clone_command.extend([url, str(upstream)])
        subprocess.run(clone_command, check=True)
    os.environ[
        "VMAMBA_ROOT" if MODEL_ID == "faster_rcnn_vmamba_t" else "MMDET_ROOT"
    ] = str(upstream)
    if MODEL_ID == "faster_rcnn_vmamba_t":
        subprocess.run(
            ["git", "-C", str(upstream), "checkout",
             "2ed52ead062a51a64521ed3871d52914bf532876"],
            check=True,
        )
        os.environ["VMAMBA_T_PRETRAINED"] = str(paths.pretrained / "vmamba_t.pth")
        if not Path(os.environ["VMAMBA_T_PRETRAINED"]).is_file() and not SMOKE_TEST:
            raise FileNotFoundError(os.environ["VMAMBA_T_PRETRAINED"])
        if not SMOKE_TEST:
            try:
                import selective_scan_cuda
            except ImportError:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install",
                     str(upstream / "kernels" / "selective_scan"),
                     "--no-build-isolation"],
                    check=True,
                )
from src.notebook_utils import require_gpu, require_model_environment
if not SMOKE_TEST:
    require_model_environment("rtdetr" if MODEL_ID == "rtdetrv2_l" else "openmmlab")
    require_gpu(MODEL_ID)
print("Model environment preflight:", "SMOKE_TEST" if SMOKE_TEST else "PASS")
"""
                ),
                markdown("## Discover the final run\n"),
                code(
                    """\
import pandas as pd
from src.training.checkpointing import RunRegistry
from src.utils.serialization import read_yaml

registry = RunRegistry(paths)
candidates = []
for run in registry.list_available_runs(MODEL_ID, DATASET_TRACK, status="completed"):
    run_dir = Path(run.get("run_dir") or paths.final_checkpoints / MODEL_ID / run["run_id"])
    training_config = run_dir / "training_config.yaml"
    if training_config.exists() and read_yaml(training_config).get("run_kind") == "final_complete_official_train":
        candidates.append({**run, "run_dir": str(run_dir)})
display(pd.DataFrame([
    {key: row.get(key) for key in ("run_id", "created_at", "status", "best_validation_map", "run_dir")}
    for row in candidates
]))
if RUN_ID:
    selected_runs = [row for row in candidates if row["run_id"] == RUN_ID]
elif len(candidates) == 1:
    selected_runs = candidates
elif len(candidates) > 1:
    raise RuntimeError("Multiple compatible final runs found. Copy one run_id into RUN_ID and rerun.")
elif SMOKE_TEST:
    selected_runs = []
else:
    raise RuntimeError("No compatible completed final run. Finish notebook 13 first.")
SELECTED_RUN_ID = selected_runs[0]["run_id"] if selected_runs else None
print("Selected final run:", SELECTED_RUN_ID or "SMOKE_TEST: none required")
"""
                ),
                markdown("## Evaluate complete official validation\n"),
                code(
                    """\
if SMOKE_TEST:
    print("SMOKE_TEST: final-run discovery passed; GPU evaluation was skipped.")
else:
    command = [
        sys.executable, "scripts/evaluate.py",
        "--drive-root", DRIVE_ROOT,
        "--dataset-track", DATASET_TRACK,
        "--split", "val",
        "--run-id", SELECTED_RUN_ID,
        "--resolutions", str(EVALUATION_RESOLUTION),
    ]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)
    evaluation_files = sorted(paths.evaluation.glob(f"{SELECTED_RUN_ID}__res*__metrics.json"))
    if not evaluation_files:
        raise RuntimeError("Evaluation completed without a metrics file.")
    recommended = f"{MODEL_ID}__{DATASET_TRACK}__" + __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    ).strftime("%Y%m%d_%H%M%S")
    print("\\nRESULTS READY FOR REVIEW")
    print("\\nEvaluation directory:", paths.evaluation)
    print("Report directory:", paths.reports)
    print("Recommended result bundle ID:", recommended)
    print("Next notebook:", REPO_DIR / "notebooks" / "10_generate_final_report.ipynb")
"""
                ),
            ]
        elif name == "08_architecture_visualization.ipynb":
            notebook["cells"][3] = code(
                """\
if SMOKE_TEST:
    print("SMOKE_TEST: activation hooks require a trained model and are skipped.")
else:
    import torch, numpy as np, matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    def list_modules(model, contains=None):
        for name, module in model.named_modules():
            if not contains or contains.lower() in name.lower():
                print(name, module.__class__.__name__)
"""
            )
        elif name == "10_generate_final_report.ipynb":
            notebook["cells"] = [
                markdown(
                    "# Generate final benchmark reports\n\n"
                    "This step is CPU-compatible and idempotent. It rebuilds report artifacts "
                    "from measured evaluation JSON files in Drive.\n"
                ),
                code(BOOTSTRAP, ["bootstrap"]),
                markdown("## Configuration\n"),
                code(
                    """\
MODEL_ID = "rtdetrv2_l"
DATASET_TRACK = "2class"
"""
                ),
                markdown("## Generate and verify reports\n"),
                code(
                    """\
from src.benchmark_status import discover_model_status
if SMOKE_TEST:
    print("SMOKE_TEST: report workflow imports passed; measured evaluation is not required.")
else:
    status = discover_model_status(DRIVE_ROOT, MODEL_ID, REPO_DIR)
    if status["evaluation_status"] != "COMPLETE":
        raise RuntimeError("Evaluation is missing. Run notebook 07 first.")
    subprocess.run(
        [sys.executable, "scripts/generate_report.py", "--drive-root", DRIVE_ROOT],
        check=True,
    )
    status = discover_model_status(DRIVE_ROOT, MODEL_ID, REPO_DIR)
    if status["report_status"] != "COMPLETE":
        raise RuntimeError("The generated report does not contain the selected final run.")
recommended = f"{MODEL_ID}__{DATASET_TRACK}__" + __import__("datetime").datetime.now(
    __import__("datetime").timezone.utc
).strftime("%Y%m%d_%H%M%S")
print("\\nRESULTS READY FOR REVIEW")
print("\\nEvaluation directory:", paths.evaluation)
print("Report directory:", paths.reports)
print("Recommended result bundle ID:", recommended)
print("Next notebook:", REPO_DIR / "notebooks" / "11_sync_results_to_github.ipynb")
"""
                ),
            ]
        elif name == "11_sync_results_to_github.ipynb":
            notebook["cells"] = [
                markdown(
                    "# Validate and publish one lightweight result bundle\n\n"
                    "Dry-run is the default. No Git branch, commit, push, or pull request is "
                    "created until you explicitly set `PUBLISH_RESULTS=True` and `DRY_RUN=False`.\n"
                ),
                code(BOOTSTRAP, ["bootstrap"]),
                markdown("## Configuration\n\nEdit this cell only.\n"),
                code(
                    """\
MODEL_ID = "rtdetrv2_l"
DATASET_TRACK = "2class"
RUN_ID = ""                 # Usually leave blank; required only if discovery is ambiguous.
RESULT_BUNDLE_ID = ""       # Usually leave blank; an existing matching bundle is reused.
CREATE_BUNDLE = True
PUBLISH_RESULTS = False
DRY_RUN = True
GIT_USER_NAME = ""
GIT_USER_EMAIL = ""
"""
                ),
                markdown("## Discover measured artifacts and choose one final run\n"),
                code(
                    """\
import json
import pandas as pd
from datetime import datetime, timezone
from src.benchmark_status import discover_model_status
from src.result_export import validate_bundle
from src.training.checkpointing import RunRegistry
from src.utils.serialization import read_yaml

status = discover_model_status(DRIVE_ROOT, MODEL_ID, REPO_DIR)
print(json.dumps(status, indent=2))
if not SMOKE_TEST and (
    status["evaluation_status"] != "COMPLETE"
    or status["report_status"] != "COMPLETE"
):
    raise RuntimeError("Run notebooks 07 and 10 before creating a publication bundle.")
registry = RunRegistry(paths)
candidates = []
for run in registry.list_available_runs(MODEL_ID, DATASET_TRACK, status="completed"):
    run_dir = Path(run.get("run_dir") or paths.final_checkpoints / MODEL_ID / run["run_id"])
    config = run_dir / "training_config.yaml"
    metrics = list(paths.evaluation.glob(f"{run['run_id']}__res*__metrics.json"))
    if config.exists() and read_yaml(config).get("run_kind") == "final_complete_official_train" and metrics:
        candidates.append({**run, "run_dir": str(run_dir), "evaluation_files": len(metrics)})
display(pd.DataFrame([
    {key: row.get(key) for key in ("run_id", "created_at", "best_validation_map", "evaluation_files")}
    for row in candidates
]))
if RUN_ID:
    selected = [row for row in candidates if row["run_id"] == RUN_ID]
elif len(candidates) == 1:
    selected = candidates
elif len(candidates) > 1:
    raise RuntimeError("Multiple compatible runs found. Copy one run_id into RUN_ID and rerun.")
elif SMOKE_TEST:
    selected = []
else:
    raise RuntimeError("No completed evaluated final run was found.")
SELECTED_RUN_ID = selected[0]["run_id"] if selected else None
matching_bundles = []
for manifest_path in paths.result_bundles.glob("*/bundle_manifest.json"):
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("model_id") == MODEL_ID
        and manifest.get("dataset_track") == DATASET_TRACK
        and manifest.get("run_id") == SELECTED_RUN_ID
        and not validate_bundle(manifest_path.parent)
    ):
        matching_bundles.append(manifest_path.parent)
existing_bundle = max(matching_bundles, key=lambda item: item.stat().st_mtime) if matching_bundles else None
RESULT_BUNDLE_ID = RESULT_BUNDLE_ID or (
    existing_bundle.name if existing_bundle
    else f"{MODEL_ID}__{DATASET_TRACK}__{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
)
print("Selected run:", SELECTED_RUN_ID or "SMOKE_TEST: none required")
print("Recommended bundle ID:", RESULT_BUNDLE_ID)
"""
                ),
                markdown("## Create or reuse the lightweight bundle\n"),
                code(
                    """\
from src.result_export import create_result_bundle
bundle_path = paths.result_bundles / RESULT_BUNDLE_ID
if SMOKE_TEST:
    print("SMOKE_TEST: bundle creation skipped.")
elif CREATE_BUNDLE and not bundle_path.exists():
    bundle_path = create_result_bundle(
        DRIVE_ROOT,
        DATASET_TRACK,
        REPO_DIR,
        RESULT_BUNDLE_ID,
        model_id=MODEL_ID,
        run_id=SELECTED_RUN_ID,
    )
elif not bundle_path.exists():
    raise FileNotFoundError(bundle_path)
print("Bundle:", bundle_path)
"""
                ),
                markdown("## Validate and preview—no Git mutation\n"),
                code(
                    """\
from src.result_export import export_bundle, validate_bundle
if SMOKE_TEST:
    print("SMOKE_TEST: validation and dry-run publishing imports passed.")
else:
    errors = validate_bundle(bundle_path)
    if errors:
        raise RuntimeError("Bundle validation failed:\\n- " + "\\n- ".join(errors))
    preview = export_bundle(
        DRIVE_ROOT,
        RESULT_BUNDLE_ID,
        REPO_DIR,
        dry_run=True,
    )
    print("Files that will be copied:")
    for item in preview["preview"]:
        if item["action"] == "copy":
            print(" +", item["destination"])
    print("Files that will be excluded:")
    for item in preview["preview"]:
        if item["action"] == "exclude":
            print(" -", item["source"])
    print("Total file count:", preview["file_count"])
    print("Total bundle size:", preview["total_size_bytes"], "bytes")
    print("Target Git branch:", preview["target_branch"])
    print("Target repository paths:", preview["target_bundle"], preview["latest_manifest"])
    print("Validation result:", preview["validation"])
    print("Git diff preview:", *preview["projected_git_diff"], sep="\\n  ")
    if DRY_RUN:
        print("DRY RUN COMPLETE: repository files, Git index, remote, and PR were not changed.")
"""
                ),
                markdown("## Explicit publishing step\n"),
                code(
                    """\
if PUBLISH_RESULTS:
    if DRY_RUN:
        raise RuntimeError("Set DRY_RUN=False only after reviewing the preview.")
    if not GIT_USER_NAME or not GIT_USER_EMAIL:
        raise RuntimeError("Set GIT_USER_NAME and GIT_USER_EMAIL in the configuration cell.")
    if subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip():
        raise RuntimeError("Repository is not clean. Preserve or remove unrelated changes before publishing.")
    subprocess.run(["gh", "auth", "status"], check=True)  # Checks auth; never prints a token.
    subprocess.run(["git", "fetch", "origin"], check=True)
    branch = "experiment-results"
    remote_exists = subprocess.run(
        ["git", "ls-remote", "--exit-code", "--heads", "origin", branch],
        capture_output=True,
    ).returncode == 0
    base_ref = f"origin/{branch}" if remote_exists else "origin/main"
    subprocess.run(["git", "checkout", "-B", branch, base_ref], check=True)
    subprocess.run(["git", "config", "user.name", GIT_USER_NAME], check=True)
    subprocess.run(["git", "config", "user.email", GIT_USER_EMAIL], check=True)
    export_bundle(DRIVE_ROOT, RESULT_BUNDLE_ID, REPO_DIR, dry_run=False)
    subprocess.run(
        [sys.executable, "-m", "scripts.validate_results", "--repo-results", "results"],
        check=True,
    )
    subprocess.run(["git", "add", "--", "results"], check=True)
    subprocess.run(["git", "diff", "--cached", "--stat"], check=True)
    subprocess.run(["git", "diff", "--cached", "--name-status"], check=True)
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    if not staged or any(not path.startswith("results/") for path in staged):
        raise RuntimeError("Staging safety check failed: only results/ may be committed.")
    commit_message = f"results({MODEL_ID}): add 2-class LR benchmark results"
    subprocess.run(["git", "commit", "-m", commit_message], check=True)
    subprocess.run(["git", "push", "-u", "origin", branch], check=True)
    title = f"Results: {MODEL_ID} VisDrone 2-class LR benchmark"
    bundle_manifest = json.loads((bundle_path / "bundle_manifest.json").read_text())
    metric_payload = json.loads((bundle_path / "metrics" / "final_metrics.json").read_text())
    metric = metric_payload["evaluations"][0]
    dataset_provenance = json.loads(
        (bundle_path / "provenance" / "dataset_hashes.json").read_text()
    )
    environment = json.loads(
        (bundle_path / "provenance" / "environment_summary.json").read_text()
    )
    statistics = dataset_provenance.get("split_statistics", {})
    train_images = statistics.get("official_full_train.json", {}).get("images", "recorded in bundle")
    validation_images = statistics.get("official_validation.json", {}).get("images", "recorded in bundle")
    body = (
        f"Model: `{MODEL_ID}`\\n\\n"
        f"Run: `{SELECTED_RUN_ID}`\\n\\n"
        f"Selected LR: `{bundle_manifest['selected_learning_rate']}`\\n\\n"
        "Search: LR-only logarithmic-grid successive halving at epochs 2/5/10/15.\\n\\n"
        "Final epoch budget: `25`\\n\\n"
        f"Final training images: `{train_images}`\\n\\n"
        f"Official validation images: `{validation_images}`\\n\\n"
        f"mAP50-95: `{metric.get('mAP')}`\\n\\n"
        f"APtiny: `{metric.get('APtiny')}`\\n\\n"
        f"Training time (seconds): `{metric.get('total_training_seconds')}`\\n\\n"
        f"GPU: `{metric.get('evaluation_hardware') or environment.get('gpu_name')}`\\n\\n"
        "Known limitations: single seed (42), one model-day controlled benchmark.\\n\\n"
        f"Bundle: `results/bundles/{RESULT_BUNDLE_ID}`\\n\\n"
        "Checkpoints, datasets, raw predictions, and credentials are excluded."
    )
    existing_pr = subprocess.run(
        ["gh", "pr", "list", "--head", branch, "--base", "main",
         "--state", "open", "--json", "url", "--jq", ".[0].url"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if existing_pr:
        print("Existing pull request:", existing_pr)
    else:
        subprocess.run(
            ["gh", "pr", "create", "--base", "main", "--head", branch,
             "--title", title, "--body", body],
            check=True,
        )
else:
    print("Publishing is OFF. Set PUBLISH_RESULTS=True and DRY_RUN=False only after review.")
"""
                ),
            ]
        if name == "00_environment_and_data_setup.ipynb":
            notebook["cells"] = [
                markdown(
                    "# Legacy environment/data setup\n\n"
                    "This notebook is retained as a compatibility entry point. Model frameworks no "
                    "longer install into the shared dataset environment. Run "
                    "`00_visdrone_colab_setup.ipynb` first.\n"
                ),
                code(BOOTSTRAP, ["bootstrap"]),
                code(
                    """\
from src.notebook_utils import preflight_dataset
for track in ("2class", "10class"):
    report = preflight_dataset(paths, track, minimum_free_gb=0)
    print(track, report)
    report.raise_for_errors()
print("Dataset environment is ready. Choose a model-specific notebook next.")
"""
                ),
            ]
        elif name == "00_colab_repository_setup.ipynb":
            notebook["cells"] = [
                markdown(
                    "# Repository and storage preflight\n\n"
                    "This lightweight compatibility notebook verifies the clone and persistent "
                    "directory tree. Dataset preparation now starts with "
                    "`00_visdrone_colab_setup.ipynb`.\n"
                ),
                code(BOOTSTRAP, ["bootstrap"]),
                code(
                    """\
from src.drive_sync import validate_drive_writable
validate_drive_writable(paths.root)
for label, output in {
    "archives": paths.archives,
    "raw": paths.raw,
    "2class": paths.coco("2class"),
    "10class": paths.coco("10class"),
    "registry": paths.registry_dir,
}.items():
    print(label, output, output.exists())
print("Next: notebooks/00_visdrone_colab_setup.ipynb")
"""
                ),
            ]
        for index, cell in enumerate(notebook["cells"]):
            cell["id"] = f"cell-{index:03d}"
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build_setup_notebook()
    build_lr_workflow_notebooks()
    patch_existing_notebooks()
