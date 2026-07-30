# Run the benchmark

## Prepare once

Open [`00_prepare_visdrone.ipynb`](../notebooks/00_prepare_visdrone.ipynb),
mount Drive, and run all cells. Continue only after `DATA CONTRACT VERIFIED:
YES`. Reruns validate and reuse good archives, extraction inventories, and COCO
conversions; they do not redownload or delete valid data.

## Choose a protocol

The preserved `lr_controlled_v1` path is:

```text
00 → 01_run_model_day → 02_publish_results → 03_compare_all_models
```

For `two_stage_random_hpo_v1`, choose notebook 10, 11, 12, or 13 for the model.
Set `DATASET_TRACK` to `2class` or `10class`. Inspect with `START_HPO=False`,
then set it to `True`.

Optuna uses `RandomSampler(seed=42)`: five broad trials, followed by five trials
from refined ranges. Both search subsets come only from official train. The
official validation split is excluded. The persistent study directory contains
`study.db`, both trial CSVs, both best-config formats, search-space/summary JSON,
and a parameter-application report.

Next open the matching notebook 20–23. It discovers `best_config.yaml`
automatically. Set `START_FINETUNING=True`. The final workflow restarts from the
original pretrained model and runs baseline and tuned recipes at seeds 17, 42,
and 3407 using complete official train.

Resume requires matching model, track, protocol, seed, resolution, effective
batch, configuration hash, and scheduler contract. HPO checkpoints are never
used as final starting weights.

## Evaluate and publish

Notebook 30 discovers compatible final checkpoints and writes baseline/tuned
mean and standard deviation. Missing runs remain missing. Track A and Track B
reports are separate.

Notebook 31 defaults to a non-mutating dry-run. To publish, add `GH_TOKEN` as a
Colab secret, review the preview, then set:

```python
PUBLISH_RESULTS = True
DRY_RUN = False
```

Publication uses a temporary clone and leaves the training checkout on clean
`main`. It creates `experiment-results` from `origin/main` if absent and stages
only the selected bundle plus the required latest manifest.

After a disconnect, rerun all cells. Environments, gates, studies, compatible
checkpoints, and completed evaluations are discovered automatically.
