# Aerial Object Detection Benchmark

[![CI](https://github.com/Harryphan72007/aerial-object-detection-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/Harryphan72007/aerial-object-detection-benchmark/actions/workflows/ci.yml)

Reproducible comparison of CNN, Transformer, Vision Mamba, and real-time DETR detector families on VisDrone.

> **Status: scaffold complete, experiments not yet run.** Every result cell is `TBD`. This repository does not claim accuracy, latency, memory, or ranking results before timestamped artifacts are committed.

## Research question

Under the same data split, image size, augmentation policy, optimization budget, seeds, and evaluation code, how do detector families trade off:

- COCO-style mAP, AP50, and AP75
- small-, medium-, and large-object AP
- per-class AP
- latency, throughput, peak accelerator memory, parameters, and FLOPs

VisDrone is a useful stress test because aerial imagery combines small objects, dense scenes, occlusion, and large scale variation.

## Compared families

| Family | Planned implementation | Upstream code license | Integration |
| --- | --- | --- | --- |
| CNN | TorchVision Faster R-CNN | BSD-3-Clause | adapter planned |
| Transformer | Hugging Face DETR | Apache-2.0 | adapter planned |
| Vision Mamba | HUST-VL Vim backbone + common detector head | Apache-2.0 | adapter planned |
| Real-time DETR | Hugging Face RT-DETR | Apache-2.0 | adapter planned |

Licenses were checked against upstream repositories, but pretrained weights and datasets can have separate terms. Re-verify every exact checkpoint before downloading or redistributing it. NVIDIA MambaVision is deliberately excluded because its official code and weights use noncommercial terms.

This repository itself has no license yet; reuse permission has not been granted.

## Repository map

```text
configs/       shared protocol and one config per detector family
notebooks/     one thin, reproducible notebook per family
scripts/       dataset conversion, evaluation, dry runs, and Optuna studies
src/           shared parsing, metrics, provenance, results, and study code
tests/         fast unit tests for the benchmark contract
results/       empty templates; generated artifacts are ignored
```

## Set up

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev,eval,optimize]"
```

Install the model framework for the adapter you are running only after checking its exact code and weight licenses.

## Prepare VisDrone

Download VisDrone from its official project and accept the dataset terms. The benchmark never redistributes the images or annotations.

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

Inspect a family’s resolved protocol without training:

```bash
python scripts/run_experiment.py --config configs/cnn.yaml --dry-run
```

Adapters must emit standard COCO detection JSON. Evaluate all families through the same code:

```bash
python scripts/evaluate.py \
  --ground-truth data/processed/visdrone-val.json \
  --predictions artifacts/cnn-seed17/predictions.json \
  --model cnn-faster-rcnn \
  --seed 17 \
  --output results/runs.jsonl
```

The command records metrics with Git revision, environment, hardware, timestamp, and prediction-file hash. It refuses to invent missing measurements.

## Resumable Optuna studies

```bash
python scripts/run_study.py \
  --study cnn-visdrone \
  --storage sqlite:///artifacts/optuna.db \
  --objective my_adapter.objectives:train_and_evaluate \
  --trials 30
```

The study uses persistent storage and `load_if_exists=True`; interrupted jobs resume without losing completed trials. The objective is an adapter-owned function receiving one Optuna `Trial`.

## Fair-comparison protocol

- official train/validation split; test-challenge only for a final submission
- image size 640 unless a documented architecture constraint requires otherwise
- same geometric/color augmentations
- fixed 50-epoch and wall-clock budgets reported separately
- seeds `17`, `42`, and `73`
- no test-set tuning
- identical COCO evaluation and confidence/IoU policy
- warm-up before latency measurement; batch size and precision reported
- per-family hyperparameter search budget kept equal
- failed and out-of-memory runs retained in the study log

See [the reproducibility checklist](docs/REPRODUCIBILITY.md).

## Results

No experiments have been run.

| Model | Seed | mAP | AP50 | AP75 | AP-small | Latency | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN / Faster R-CNN | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Transformer / DETR | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Vision Mamba / Vim | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Real-time / RT-DETR | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Milestones

- [x] Shared dataset conversion, evaluation, provenance, and result schema
- [x] Fairness protocol and resumable Optuna contract
- [x] Four configuration files and notebooks
- [ ] Implement and smoke-test each detector adapter
- [ ] Run one-seed pilot experiments
- [ ] Freeze search spaces and run equal-budget studies
- [ ] Run three-seed final experiments
- [ ] Publish plots, error analysis, model cards, and a tagged release

## Citation

Use [CITATION.cff](CITATION.cff) for this scaffold and cite the VisDrone dataset and each model implementation used in a run.
