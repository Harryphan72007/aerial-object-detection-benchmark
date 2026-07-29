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
    from src.notebook_utils import require_gpu
    require_gpu(MODEL_ID)
    subprocess.run(shlex.split(cmd), check=True)
"""
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
            notebook["cells"][3] = code(
                """\
from src.training.checkpointing import RunRegistry
import pandas as pd
registry = RunRegistry(paths)
runs = registry.list_available_runs(dataset_track="2class")
display(pd.DataFrame(runs)) if runs else print("No registered 2class checkpoints yet.")
"""
            )
            notebook["cells"][4] = code(
                """\
DATASET_TRACK = "2class"
MODELS = []
MAX_IMAGES = 2 if SMOKE_TEST else None
if SMOKE_TEST:
    print("SMOKE_TEST: registry discovery verified; evaluation requires a completed checkpoint.")
else:
    command = [sys.executable, "scripts/evaluate.py", "--drive-root", DRIVE_ROOT,
               "--dataset-track", DATASET_TRACK, "--best-per-model"]
    if MODELS:
        command.extend(["--models", *MODELS])
    if MAX_IMAGES is not None:
        command.extend(["--max-images", str(MAX_IMAGES)])
    subprocess.run(command, check=True)
    subprocess.run(
        [sys.executable, "scripts/create_results_manifest.py", "--drive-root", DRIVE_ROOT,
         "--dataset-track", DATASET_TRACK],
        check=True,
    )
"""
            )
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
            notebook["cells"][3] = code(
                """\
if SMOKE_TEST:
    print("SMOKE_TEST: report inputs inspected; full report generation skipped.")
else:
    subprocess.run([sys.executable, "scripts/generate_report.py", "--drive-root", DRIVE_ROOT], check=True)
    print(paths.reports)
    for output in sorted(paths.reports.rglob("*")):
        if output.is_file():
            print(output.relative_to(paths.reports))
"""
            )
        elif name == "11_sync_results_to_github.ipynb":
            notebook["cells"][1] = code(
                BOOTSTRAP
                + """\
LOCAL_REPOSITORY = str(REPO_DIR)
RESULT_BUNDLE_ID = os.environ.get("RESULT_BUNDLE_ID", "")
GIT_USER_NAME = os.environ.get("GIT_USER_NAME", "")
GIT_USER_EMAIL = os.environ.get("GIT_USER_EMAIL", "")
""",
                ["bootstrap"],
            )
            notebook["cells"][2] = code("print('Repository and storage initialized; no Git mutation performed.')\n")
            for index in (6, 8, 10):
                notebook["cells"][index] = code(
                    "print('SMOKE_TEST: publishing step skipped.' if SMOKE_TEST else "
                    "'Run this publishing step only after selecting and reviewing a valid result bundle.')\n"
                )
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
    patch_existing_notebooks()
