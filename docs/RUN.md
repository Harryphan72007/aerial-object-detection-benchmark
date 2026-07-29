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
reruns only after their hashes, inventories, counts, and provenance match. A
damaged extracted split is rebuilt atomically from its verified ZIP without
unnecessarily changing that ZIP.

`DATASET_SOURCE="auto"` tries a verified Drive archive, then the resumable public
download, then prints manual placement instructions. Use `"drive"` to require
existing Drive ZIPs, `"download"` to use the public source, or `"manual"` to
upload/copy the exact ZIP filenames into the printed Drive destination. Every
mode applies the same size, ZIP-layout, CRC, SHA-256, and manifest checks.

The final cell must print `DATA CONTRACT VERIFIED: YES` followed by the exact
archive, raw-image, COCO-annotation, and LR-manifest paths.

## 2. Run one model day

Open [`01_run_model_day.ipynb`](../notebooks/01_run_model_day.ipynb), change only
`MODEL_ID`, and run all cells. Review the derived state, then set
`START_EXPENSIVE_STAGE=True`.

Notebook 01 independently installs or validates its lightweight dependencies and
reruns the authoritative data preflight. It refuses an expensive stage unless
the result is `DATA CONTRACT VERIFIED: YES`; it does not inherit imports or
variables from notebook 00.

In Colab, `DATA_ACCESS_MODE="local_cache"` is the default. Each new session
synchronizes verified images to `/content/visdrone_cache`, copies the COCO and
LR-search JSON files, and prints size, copy time, and verification status.
Outside Colab the notebook resolves to `"drive_direct"`. To avoid a local copy in
Colab, set `DATA_ACCESS_MODE="drive_direct"`. Insufficient local disk stops with
a clear warning; switching to Drive reads requires that explicit setting.

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
run all cells again. The local cache is rebuilt as needed and is never persistent
state. Checkpoints, search state, selected configurations, metrics, predictions,
and reports stay on Drive in both access modes.

## Canonical data paths

```text
$DRIVE_ROOT/
└── datasets/VisDrone2019-DET/
    ├── archives/
    ├── raw/
    │   ├── VisDrone2019-DET-train/
    │   │   ├── images/
    │   │   └── annotations/
    │   └── VisDrone2019-DET-val/
    │       ├── images/
    │       └── annotations/
    ├── processed/
    │   └── coco_2class/
    │       └── annotations/
    └── manifests/
        └── lr_search/
```

The processed tree contains JSON annotations, audits, statistics, and conversion
manifests only. Training and evaluation obtain images from the two persistent
raw directories through the centralized path contract.

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
