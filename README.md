# Aerial Object Detection Benchmark

[![CI](https://github.com/Harryphan72007/aerial-object-detection-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/Harryphan72007/aerial-object-detection-benchmark/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-experiments%20pending-F59E0B)

A reproducible benchmark for comparing CNN, Transformer, Vision Mamba, and real-time DETR detector families on VisDrone under one evaluation protocol.

> [!IMPORTANT]
> The benchmark infrastructure is implemented, but model experiments have not yet been run. Every result remains `TBD`; this repository makes no accuracy, latency, memory, or ranking claims without committed artifacts.

## Research question

Under the same data split, image size, augmentation policy, optimization budget, seeds, and evaluation code, how do detector families trade off:

- COCO-style mAP, AP50, and AP75
- small-, medium-, and large-object AP
- per-class AP
- latency, throughput, peak accelerator memory, parameters, and FLOPs

VisDrone is a useful stress test because aerial imagery combines small objects, dense scenes, occlusion, and large scale variation.

## Compared families

| Family | Planned implementation | Upstream code license | Integration status |
| --- | --- | --- | --- |
| CNN | TorchVision Faster R-CNN | BSD-3-Clause | Adapter planned |
| Transformer | Hugging Face DETR | Apache-2.0 | Adapter planned |
| Vision Mamba | HUST-VL Vim backbone with a common detector head | Apache-2.0 | Adapter planned |
| Real-time DETR | Hugging Face RT-DETR | Apache-2.0 | Adapter planned |

Pretrained weights and datasets can have terms that differ from their source-code repositories. Every exact checkpoint must be reviewed before download or redistribution. NVIDIA MambaVision is deliberately excluded because its official code and weights use noncommercial terms.

## Benchmark contract

The shared protocol is designed to prevent each model family from receiving a different experimental advantage:

- Official train/validation split; test-challenge data reserved for final submission
- Image size 640 unless an architecture constraint is documented
- Identical geometric and color augmentations
- Fixed 50-epoch and wall-clock budgets reported separately
- Seeds `17`, `42`, and `73`
- Equal per-family hyperparameter-search budgets
- Shared COCO evaluation and confidence/IoU policy
- Warm-up and device synchronization before latency measurement
- Failed, pruned, and out-of-memory runs retained in study logs

The complete publication checklist is in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Repository structure

```text
configs/       shared protocol and one configuration per detector family
notebooks/     thin, reproducible notebook entry points
scripts/       dataset conversion, evaluation, dry runs, and Optuna studies
src/           parsing, metrics, provenance, results, and study utilities
tests/         fast tests for the benchmark contract
results/       empty templates; generated artifacts remain ignored
```

## Quick start

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,eval,optimize]"
```

Install a model framework only when running its adapter, after checking the exact code and checkpoint licenses.

## Prepare VisDrone

Download VisDrone from its official project and accept the dataset terms. This repository never redistributes source images or annotations.

Expected detection layout:

```text
data/raw/VisDrone2019-DET-train/
  images/
  annotations/
data/raw/VisDrone2019-DET-val/
  images/
  annotations/
```

Convert a split to COCO format and record a source manifest:

```bash
python scripts/prepare_visdrone.py \
  --split-dir data/raw/VisDrone2019-DET-val \
  --output data/processed/visdrone-val.json
```

## Run and evaluate

Inspect a resolved protocol without training:

```bash
python scripts/run_experiment.py --config configs/cnn.yaml --dry-run
```

Adapters emit standard COCO detection JSON. All families are evaluated through the same command:

```bash
python scripts/evaluate.py \
  --ground-truth data/processed/visdrone-val.json \
  --predictions artifacts/cnn-seed17/predictions.json \
  --model cnn-faster-rcnn \
  --seed 17 \
  --output results/runs.jsonl
```

Each result records the Git revision, environment, hardware, timestamp, and prediction-file hash. Missing measurements are rejected rather than inferred.

## Resumable optimization

```bash
python scripts/run_study.py \
  --study cnn-visdrone \
  --storage sqlite:///artifacts/optuna.db \
  --objective my_adapter.objectives:train_and_evaluate \
  --trials 30
```

Optuna studies use persistent storage and `load_if_exists=True`, so interrupted searches resume without losing completed trials.

## Results

No experiments have been run.

| Model | Seed | mAP | AP50 | AP75 | AP-small | Latency | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN / Faster R-CNN | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Transformer / DETR | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Vision Mamba / Vim | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Real-time / RT-DETR | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Roadmap

- [x] Shared dataset conversion, evaluation, provenance, and result schema
- [x] Fairness protocol and resumable Optuna contract
- [x] Four family configuration files and notebook entry points
- [ ] Implement and smoke-test each detector adapter
- [ ] Run one-seed pilot experiments
- [ ] Freeze search spaces and run equal-budget studies
- [ ] Run three-seed final experiments
- [ ] Publish plots, error analysis, model cards, and a tagged release

## Verification

```bash
ruff check .
pytest
```

The same checks run in GitHub Actions for pushes to `main` and pull requests.

## Citation and reuse

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Cite the VisDrone dataset and every exact model implementation and checkpoint used in a run.

This repository currently has no license. Reuse permission has not been granted.
