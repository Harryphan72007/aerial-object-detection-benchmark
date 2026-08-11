# Run the benchmark

The supported workflow is five runs:

```text
10_resnet50 ─┐
11_swin_t   ─┤  each: dataset → environment → smoke gate
12_vmamba_t ─┤         → HPO → final training → evaluation
13_rtdetrv2 ─┘
        ↓
30_report   evaluate anything missing → aggregate → tables/figures → publish
```

> **Retired and deleted:** `lr_controlled_v1` (notebooks 01 → 02 → 03) no longer
> ships. Its notebooks, `src/workflows/model_day.py`, and
> `src/training/lr_workflow.py` were removed once the protocol stopped being
> runnable; artifacts it produced remain readable through the versioned
> evaluator. Any older instruction to run notebook 01 with
> `RUN_MODE="environment"` as an integration gate is obsolete; that gate is now
> a stage of the pipeline.

## Run one model

Open the notebook for the model and run all cells. With the shipped
`START = False` you get a preview of every stage's contract: nothing is
downloaded, no environment is provisioned, no GPU is touched. Review it, set
`START = True`, and run all cells again.

Four parameters, and only `START` normally changes:

| Parameter | Default | Meaning |
|---|---|---|
| `DATASET_TRACK` | `"2class"` | `2class` or `10class` |
| `START` | `False` | `False` previews; `True` runs |
| `FULL_MATRIX` | `False` | opt-in variance matrix (below) |
| `USE_GOOGLE_DRIVE` | `True` | `False` uses session storage — smoke runs only |

### Dataset

The pipeline prepares and verifies the data contract before anything reads it,
reusing good archives, extraction inventories, and COCO conversions; it does not
redownload or delete valid data. The 2-class track is prepared by default.

The 10-class track is opt-in and is not prepared automatically. Build it first
with:

```bash
python -m scripts.prepare_dataset --drive-root <artifact_root> --prepare-10class-track
```

That script is also the escape hatch for a forced redownload over a corrupt
archive, or for preparing the dataset in one session and training in another.

### Adapter gate

Before any expensive stage the model's GPU adapter smoke gate must pass on that
GPU. The pipeline runs it when no valid record exists and refuses to continue
unless the resulting record is `READY` for this commit, environment, dataset
track, and resolution. To run it on its own:

```bash
python -m scripts.gpu_adapter_smoke --drive-root <artifact_root> --dataset-track 2class
```

See [`docs/release/GPU_VALIDATION_CHECKLIST.md`](release/GPU_VALIDATION_CHECKLIST.md)
and [`docs/release/HOSTED_GPU_RUNBOOK.md`](release/HOSTED_GPU_RUNBOOK.md).

### Tuning

Optuna uses `RandomSampler(seed=42)`: five broad trials, followed by five trials
from refined ranges. Both search subsets come only from official train; the
official validation split is excluded. The persistent study directory contains
`study.db`, both trial CSVs, both best-config formats, search-space/summary JSON,
and a parameter-application report.

A trial whose training produced no usable objective pair — a missing, null,
non-numeric, NaN, infinite, or all-zero `(mAP50-95, APtiny)` — fails. It is never
recorded as a completed trial with a score of zero, for any model family.

### Final training

Final training discovers the search's `best_config.yaml` automatically. Two
matrices exist, and the notebook parameter cell selects between them:

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

#### What "final training" trains on

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

Each model notebook evaluates its own completed runs as its last stage, so by
the time all four have finished the metrics usually exist already.

[`30_report.ipynb`](../notebooks/30_report.ipynb) then evaluates anything still
missing, aggregates baseline/tuned mean and standard deviation across seeds,
writes the comparison tables and figures, and optionally publishes. Missing runs
remain missing and are reported as such. The cross-model comparison needs at
least two completed models and reports itself `UNAVAILABLE` below that rather
than discarding the per-model tables. Track A and Track B reports are separate.

Publishing defaults to a non-mutating dry run. To publish, add `GH_TOKEN` as a
Colab secret, review the preview, then set **both**:

```python
PUBLISH_RESULTS = True
DRY_RUN = False
```

The same publication is available as
`python -m scripts.publish_results --model-id <MODEL_ID>`, which is a dry run
unless `--publish` is passed.

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
