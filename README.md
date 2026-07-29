# Accuracy–Efficiency Trade-offs for Dense Aerial Detection

A reproducible research repository for comparing **CNN, Transformer, Mamba, and end-to-end DETR** architectures on VisDrone2019-DET. The benchmark contains two strictly separated tracks:

- **2-class research track:** `person` and `vehicle`, with configurable bicycle/tricycle inclusion.
- **10-class benchmark track:** original VisDrone categories for comparison with published literature.

Never compare 2-class mAP directly with published 10-class mAP.

## Models

| ID | Family | Detector / backbone | Integration |
|---|---|---|---|
| `faster_rcnn_resnet50` | CNN | Faster R-CNN + ResNet-50-FPN | MMDetection |
| `faster_rcnn_swin_t` | Transformer | Faster R-CNN + Swin-T-FPN | MMDetection |
| `faster_rcnn_vmamba_t` | Mamba / SSM | Faster R-CNN + VMamba-T-FPN | official VMamba detection tree |
| `rtdetrv2_l` | end-to-end DETR | RT-DETRv2, configured large baseline | Transformers-compatible RT-DETRv2 |
| `yolox_s` | CNN control | YOLOX-S | optional official YOLOX repository |

See [LICENSES.md](LICENSES.md). VisDrone data is not included and no commercial-use claim is made.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

## GitHub → Colab → Drive → GitHub Workflow

1. Open `00_visdrone_colab_setup.ipynb`; its user cell controls storage,
   source, tracks, redownload behavior, seed, and smoke size.
2. Mount Drive when prompted. The notebook verifies/caches archives, preserves
   raw data, builds both COCO tracks, visualizes annotations, and batch-tests data.
3. Run `01_dataset_analysis.ipynb`.
4. Open one model notebook in its compatible environment. Do not combine the
   VMamba/OpenMMLab stack with the shared dataset or RT-DETR environment.
5. Resume interrupted runs with the registered `RESUME_RUN_ID`.
6. Evaluate, generate reports, then optionally publish a reviewed lightweight bundle.

See the exact runtime choices and interaction points in
[the Colab runbook](docs/colab_runbook.md). Checkpoints, datasets, raw
predictions, logs, and credentials never enter normal Git history.

## Open in Colab

| Notebook | Purpose | Open in Colab |
|---|---|---|
| `00_visdrone_colab_setup.ipynb` | **Start here:** acquire, verify, extract, convert, validate, visualize, and batch-test VisDrone | https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/00_visdrone_colab_setup.ipynb |
| `00_colab_repository_setup.ipynb` | Clone, install, mount Drive, and verify environment | https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/00_colab_repository_setup.ipynb |
| `00_environment_and_data_setup.ipynb` | Legacy compatibility preflight; does not install model stacks | https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/00_environment_and_data_setup.ipynb |
| `01_dataset_analysis.ipynb` | Analyze converted tracks | https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/01_dataset_analysis.ipynb |
| `02_train_resnet50_faster_rcnn.ipynb` | Train CNN Faster R-CNN / ResNet-50 | https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/02_train_resnet50_faster_rcnn.ipynb |
| `03_train_swin_t_faster_rcnn.ipynb` | Train Transformer Faster R-CNN / Swin-T | https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/03_train_swin_t_faster_rcnn.ipynb |
| `04_train_vmamba_t_faster_rcnn.ipynb` | Train Mamba / VMamba-T Faster R-CNN | https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/04_train_vmamba_t_faster_rcnn.ipynb |
| `05_train_rtdetrv2_l.ipynb` | Train end-to-end RT-DETRv2 | https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/05_train_rtdetrv2_l.ipynb |
| `06_train_yolox_s_optional.ipynb` | Train optional YOLOX-S CNN control | https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/06_train_yolox_s_optional.ipynb |
| `02`–`06` training notebooks | Train CNN, Transformer, Mamba, RT-DETR, and optional YOLOX | https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/02_train_resnet50_faster_rcnn.ipynb |
| `07_evaluate_all_models.ipynb` | Registry-driven evaluation and profiling | https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/07_evaluate_all_models.ipynb |
| `08_architecture_visualization.ipynb` | Inspect architecture modules and activations | https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/08_architecture_visualization.ipynb |
| `09_error_analysis.ipynb` | Analyze errors and VisDrone object-size slices | https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/09_error_analysis.ipynb |
| `10_generate_final_report.ipynb` | Generate tables, figures, and reports | https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/10_generate_final_report.ipynb |
| `08`–`10` analysis/report notebooks | Visualize, analyze errors, and report | https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/10_generate_final_report.ipynb |
| `11_sync_results_to_github.ipynb` | Preview/export results and prepare a PR | https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/11_sync_results_to_github.ipynb |

For Colab, execute notebooks in order:

```text
00_visdrone_colab_setup.ipynb
01_dataset_analysis.ipynb
02–05 one model-specific training notebook in its compatible runtime
07_evaluate_all_models.ipynb
08_architecture_visualization.ipynb
09_error_analysis.ipynb
10_generate_final_report.ipynb
11_sync_results_to_github.ipynb (optional; reviewed bundles only)
```

The notebooks use one root only:

```python
DRIVE_ROOT = "/content/drive/MyDrive/visdrone_architecture_benchmark"
```

All model-specific paths are built by `src.paths.ProjectPaths`.

## Dataset setup

Notebook 00 creates this persistent layout:

```text
$DRIVE_ROOT/datasets/VisDrone2019-DET/
  archives/
  raw/{VisDrone2019-DET-train,VisDrone2019-DET-val}/
  processed/{coco_2class,coco_10class}/
  manifests/
```

Convert and validate:

```bash
python -m scripts.prepare_data --drive-root "$DRIVE_ROOT" --tracks 2class 10class --validate
```

Exclude light vehicles from the collapsed vehicle class with `--exclude-light-vehicles`.

## Training

```bash
python scripts/train.py   --model-id faster_rcnn_resnet50   --dataset-track 2class   --image-size 1024 --batch-size 2   --gradient-accumulation-steps 8   --epochs 100 --seed 42   --drive-root "$DRIVE_ROOT"
```

Resume with `--resume-run-id RUN_ID`. A completed run writes a manifest, standardized checkpoints, histories, logs, environment metadata, and atomically updates both registries.

VMamba requires cloning the official repository and setting `VMAMBA_ROOT`. Set `VMAMBA_T_PRETRAINED` only to a locally verified VMamba-T checkpoint; when it is unset, the run is explicitly marked as training from scratch. MMDetection paths can be set with `MMDET_ROOT`; the setup notebook installs supported versions and records them. RT-DETRv2 uses the exact configured Hugging Face model ID and saves that ID in the manifest.

## Evaluation

```bash
python scripts/evaluate.py   --drive-root "$DRIVE_ROOT"   --dataset-track 2class   --split val   --best-per-model   --resolutions 640 1024 1280

python scripts/create_results_manifest.py --drive-root "$DRIVE_ROOT" --dataset-track 2class
python scripts/sync_results_to_repo.py --drive-root "$DRIVE_ROOT" --bundle-id "evaluation__2class__YYYYMMDD_HHMMSS" --repo-root . --validate --dry-run
python scripts/validate_results.py --repo-results results/

python scripts/profile_model.py --drive-root "$DRIVE_ROOT" --run-id RUN_ID
python scripts/generate_report.py --drive-root "$DRIVE_ROOT"
```

The evaluator discovers checkpoints from `experiment_registry/checkpoint_registry.json`, validates label mappings, exports common COCO JSON predictions, and writes metrics to `evaluation/`. The exporter prints its copy/exclusion preview and never stages or commits; inspect the Git diff before staging only `results/ benchmark_data/`.

## Run IDs and checkpoint registry

```text
MODEL_ID__DATASET_TRACK__RESOLUTION__TIMESTAMP__SEED
```

Example: `faster_rcnn_resnet50__2class__1024__20260725_153000__seed42`.

Registry writes use a temporary file, `fsync`, and atomic replacement. `runs.csv` is regenerated from the JSON registry, so JSON remains the source of truth.

## Expected Drive layout

```text
visdrone_architecture_benchmark/
├── datasets/VisDrone2019-DET/{archives,raw,processed,manifests}
├── checkpoints/MODEL_ID/RUN_ID/
├── experiment_registry/{checkpoint_registry.json,runs.csv}
├── predictions/
├── evaluation/
├── logs/
├── optuna/
├── profiling/
├── exports/
├── reports/
├── result_bundles/
└── cache/
```

## Reproducibility protocol

- Seeds: `17`, `42`, `3407`.
- Save exact code commit, `pip freeze`, GPU/CUDA/PyTorch versions, config copies, label mapping, and checkpoint hashes.
- Train architecture-default and controlled-recipe experiments separately.
- Compare models only on the same split, class mapping, resolution, seed policy, and hardware/software environment.
- Report mean, standard deviation, range, and confidence intervals; use paired bootstrap when prediction sets share images.

## Hardware reporting

Record GPU model, GPU count, VRAM, CPU, RAM, CUDA/driver, PyTorch, framework versions, precision mode, batch size, accumulation, workers, resolution, warm-up count, and timed iterations. Pure-forward benchmarks exclude disk and data-loader time and synchronize CUDA before/after timing.

## Sample result schema

| model_id | track | resolution | mAP | APtiny | latency_ms | peak_vram_gb | training_hours |
|---|---:|---:|---:|---:|---:|---:|---:|
| populated by evaluation | | | | | | | |

Figures are generated into `reports/figures/` as PNG and PDF. Empty placeholders are intentional until experiments are run; the project never invents benchmark values.

## Known limitations

- Heavy frameworks and CUDA extensions cannot be validated on every Colab image in CI; CI runs lightweight unit tests and import guards.
- VMamba depends on its official extension build and vendored MMDetection 3.3.0 tree; use the setup notebook's pinned environment.
- Exact RT-DETRv2 “L” naming differs across repositories. The run manifest records the concrete backbone/model ID; compare architecture fields, not aliases.
- TIDE, TensorRT, and NVML energy estimates are optional and reported only when dependencies and hardware support them.
- VisDrone test-dev labels are not public in the same manner as train/val; internal evaluation defaults to validation.

## Citation

Use [CITATION.cff](CITATION.cff) and cite VisDrone plus each model/framework used. Do not present internal 2-class results as official VisDrone leaderboard numbers.
