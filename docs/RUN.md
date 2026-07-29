# Run the benchmark

The required journey is:

```text
00 prepare VisDrone once
        ↓
01 select one MODEL_ID and run/resume its model day
        ↓
02 validate, dry-run, and explicitly publish
        ↓
03 compare compatible completed models
```

## 1. Prepare the dataset

Open
[`00_prepare_visdrone.ipynb`](../notebooks/00_prepare_visdrone.ipynb) and run all
cells. Its default configuration prepares only the controlled 2-class track.
Verified archives, extraction, conversions, and search manifests are reused on
reruns. Raw data remains unchanged.

The final cell prints the exact official-train, official-validation, COCO, and
search-manifest paths, plus the next notebook link.

## 2. Run one model day

Open [`01_run_model_day.ipynb`](../notebooks/01_run_model_day.ipynb), change only
`MODEL_ID`, and run all cells. Review the derived state, then set
`START_EXPENSIVE_STAGE=True`.

`RUN_MODE="auto"` selects the first incomplete stage:

```text
ENVIRONMENT → DATA → LR_SEARCH → FINAL_TRAINING
            → EVALUATION → PROFILING → REPORT → COMPLETE
```

Persistent search rungs are skipped. Each candidate resumes from its own
checkpoint. Final training resumes only when model, track, seed, input size,
selected LR, scheduler horizon, effective batch, and run kind match. It never
resumes from search weights.

If a session disconnects, reopen the same notebook with the same `MODEL_ID` and
run all cells again.

## 3. Publish safely

Open [`02_publish_results.ipynb`](../notebooks/02_publish_results.ipynb). Defaults
create/reuse and validate a lightweight bundle, scan it, and preview Git without
modifying Git or GitHub.

Publishing occurs only with:

```python
PUBLISH_RESULTS = True
DRY_RUN = False
```

Only the approved `results/bundles/<bundle-id>/` path is staged.

## 4. Compare completed models

After at least two models complete, open
[`03_compare_all_models.ipynb`](../notebooks/03_compare_all_models.ipynb). It
accepts only the controlled 2-class, seed-42, 640-pixel, 25-epoch protocol and
marks missing or incompatible models without inventing values.

## CLI

The notebooks use the same reusable workflow exposed by:

```bash
python -m scripts.benchmark status
python -m scripts.benchmark next --model-id MODEL_ID
python -m scripts.benchmark run-model-day --model-id MODEL_ID
python -m scripts.benchmark publish --model-id MODEL_ID --dry-run
python -m scripts.benchmark compare
```

The run command is a preview unless `--start-expensive-stage` is passed.
