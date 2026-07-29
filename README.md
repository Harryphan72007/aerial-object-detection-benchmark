# Aerial Object Detection Benchmark

## Project purpose

This repository compares four detector families on VisDrone with one controlled,
learning-rate-only protocol. It provides deterministic dataset preparation,
resumable successive halving, full-official-train fine-tuning, common evaluation,
profiling, reporting, and safe lightweight result publishing.

No benchmark metrics are placeholders. The repository stays empty of results until
measured, compatible runs are published.

## Models

| `MODEL_ID` | Architecture |
|---|---|
| `faster_rcnn_resnet50` | Faster R-CNN + ResNet-50-FPN |
| `faster_rcnn_swin_t` | Faster R-CNN + Swin-T-FPN |
| `faster_rcnn_vmamba_t` | Faster R-CNN + VMamba-T-FPN |
| `rtdetrv2_l` | RT-DETRv2 R101 |

YOLOX is not part of the controlled benchmark.

## Run the benchmark

1. Run dataset setup once.
2. Choose one `MODEL_ID` in notebook 01.
3. Rerun notebook 01 after interruptions; it resumes the next compatible stage.
4. Dry-run and publish that model with notebook 02.
5. After at least two models finish, compare them with notebook 03.

See [docs/RUN.md](docs/RUN.md) for the complete student workflow.

## Required notebooks

| Step | Notebook | Colab |
|---|---|---|
| 0 | [`00_prepare_visdrone.ipynb`](notebooks/00_prepare_visdrone.ipynb) | [Open in Colab](https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/00_prepare_visdrone.ipynb) |
| 1 | [`01_run_model_day.ipynb`](notebooks/01_run_model_day.ipynb) | [Open in Colab](https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/01_run_model_day.ipynb) |
| 2 | [`02_publish_results.ipynb`](notebooks/02_publish_results.ipynb) | [Open in Colab](https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/02_publish_results.ipynb) |
| 3 | [`03_compare_all_models.ipynb`](notebooks/03_compare_all_models.ipynb) | [Open in Colab](https://colab.research.google.com/github/Harryphan72007/aerial-object-detection-benchmark/blob/main/notebooks/03_compare_all_models.ipynb) |

Optional analysis notebooks live under [`notebooks/optional/`](notebooks/optional/)
and are not required to complete the benchmark.

## Current status

The code and CPU synthetic journey are testable in CI. Full model construction,
CUDA training, VMamba selective-scan compilation, and benchmark metrics require
the documented GPU runtimes and have not been fabricated by repository tests.

The same discovery layer is available from one CLI:

```bash
python -m scripts.benchmark status
python -m scripts.benchmark next --model-id rtdetrv2_l
python -m scripts.benchmark run-model-day --model-id rtdetrv2_l
python -m scripts.benchmark publish --model-id rtdetrv2_l --dry-run
python -m scripts.benchmark compare
```

## Results

Only validated lightweight bundles belong under `results/bundles/`. Datasets,
checkpoints, raw predictions, training logs, and credentials remain outside Git.
See [docs/RESULTS.md](docs/RESULTS.md).

## Licenses and citation

Project code is MIT licensed. VisDrone and upstream model/framework assets retain
their own terms; review [LICENSES.md](LICENSES.md). Cite this repository using
[CITATION.cff](CITATION.cff).
