# Run the benchmark

The supported workflow is one path:

```text
00_prepare_visdrone
        ↓
10–13 HPO notebooks
        ↓
20–23 final-training notebooks
        ↓
30 evaluate → 31 publish
```

Before either expensive stage starts, the model's GPU adapter smoke gate must
have passed on that GPU. See
[`docs/release/GPU_VALIDATION_CHECKLIST.md`](release/GPU_VALIDATION_CHECKLIST.md)
for the command and
[`docs/release/HOSTED_GPU_RUNBOOK.md`](release/HOSTED_GPU_RUNBOOK.md) for the
per-model Colab/Kaggle order.

> **Retired:** `lr_controlled_v1` (notebooks 01 → 02 → 03) is **not** runnable.
> `run_model_day` raises `RetiredProtocolError` if a new run is started. Notebooks
> 01–03 remain in the repository only so historical artifacts stay readable. Any
> older instruction to run notebook 01 with `RUN_MODE="environment"` as an
> integration gate is obsolete; that gate is now
> `python -m scripts.gpu_adapter_smoke`.

## Prepare once

Open [`00_prepare_visdrone.ipynb`](../notebooks/00_prepare_visdrone.ipynb),
mount Drive, and run all cells. Continue only after `DATA CONTRACT VERIFIED:
YES`. Reruns validate and reuse good archives, extraction inventories, and COCO
conversions; they do not redownload or delete valid data.

The 2-class track is prepared by default. The 10-class track is opt-in: set
`PREPARE_10CLASS_TRACK = True` before running. Selecting `DATASET_TRACK =
"10class"` later without having prepared it stops immediately with that
instruction rather than failing deep inside training.

## Tune (notebooks 10–13)

Choose notebook 10, 11, 12, or 13 for the model. Set `DATASET_TRACK` to `2class`
or `10class`. Inspect with `START_HPO = False`, then set it to `True`. Preview
mode does not provision the model-family runtime and does not require the smoke
gate.

Optuna uses `RandomSampler(seed=42)`: five broad trials, followed by five trials
from refined ranges. Both search subsets come only from official train; the
official validation split is excluded. The persistent study directory contains
`study.db`, both trial CSVs, both best-config formats, search-space/summary JSON,
and a parameter-application report.

A trial whose training produced no usable objective pair — a missing, null,
non-numeric, NaN, infinite, or all-zero `(mAP50-95, APtiny)` — fails. It is never
recorded as a completed trial with a score of zero, for any model family.

## Train finally (notebooks 20–23)

Open the matching notebook 20–23. It discovers `best_config.yaml` automatically.
Preview with `START_FINETUNING = False` does not install model dependencies; set
it to `True` to provision and run.

Two matrices exist, and the notebook parameter cell selects between them:

| Parameter | Runs | Content |
|---|---:|---|
| `FULL_MATRIX = False` (**default**) | 1 | tuned recipe, seed **42** |
| `FULL_MATRIX = True` (opt-in) | 6 | `baseline` + `tuned` × seeds **17, 42, 3407** |

The default is the headline matrix that fits roughly one model per GPU-day. The
full matrix adds a default-LR baseline diagnostic and a multi-seed variance
estimate; it is a multi-session job and must be **reported separately**. A
single-seed table cannot claim differences below seed noise (~0.3–0.8 mAP on
VisDrone). Both matrices are defined in
[`configs/controlled/benchmark.yaml`](../configs/controlled/benchmark.yaml).

### What "final training" trains on

Final runs do **not** use the complete official train split. The splits are:

| Split | File | Size | Used for |
|---|---|---|---|
| Official source train | `instances_train.json` | 6,471 images | the source all others are derived from |
| Search subset (train/val) | `search_train_seed42.json`, `search_validation_seed42.json` | 20% of official train | HPO trials only |
| Model-selection holdout | `model_selection_seed42.json` | 5% of official train | choosing the canonical `best.pth` |
| **Final training set** | `final_train_seed42.json` | official train **minus** the 5% holdout (6,147 images) | final runs |
| Official validation | `instances_val.json` | untouched | one final evaluation, never selection |

The 5% holdout exists so that checkpoint selection never touches official
validation: selecting on the reported split would leak it into the result. Any
claim that final training uses "the complete official train set" is therefore
wrong — it uses official train minus that holdout.

Resume requires matching model, track, protocol, seed, resolution, effective
batch, configuration hash, and scheduler contract. HPO checkpoints are never used
as final starting weights.

## Evaluate and publish

Notebook 30 discovers compatible final checkpoints and writes baseline/tuned mean
and standard deviation. Missing runs remain missing. Track A and Track B reports
are separate.

Notebook 31 defaults to a non-mutating dry-run. To publish, add `GH_TOKEN` as a
Colab secret, review the preview, then set:

```python
PUBLISH_RESULTS = True
DRY_RUN = False
```

Publication uses a temporary clone and leaves the training checkout on clean
`main`. It creates `experiment-results` from `origin/main` if absent and stages
only the selected bundle plus the required latest manifest.

## Runtime estimates

GPU-hour figures require a measured seconds-per-iteration. Run

```bash
python -m scripts.measure_throughput --drive-root <artifact_root>
```

on the target GPU before relying on
[`docs/reference/runtime_budget.md`](reference/runtime_budget.md). It times every
controlled-track model in one pass; there is no per-model flag. Until it has run,
that document's per-model hour columns are `null` and its example rows are
explicitly illustrative, not measurements.

## After a disconnect

Rerun all cells. Environments, gates, studies, compatible checkpoints, and
completed evaluations are discovered automatically.
