# Aerial Object Detection Benchmark

Student-friendly, resumable VisDrone experiments for four detector families:

| `MODEL_ID` | Family |
|---|---|
| `faster_rcnn_resnet50` | Faster R-CNN + ResNet-50-FPN |
| `faster_rcnn_swin_t` | Faster R-CNN + Swin-T-FPN |
| `faster_rcnn_vmamba_t` | Faster R-CNN + VMamba-T-FPN |
| `rtdetrv2_l` | RT-DETRv2 R101 |

YOLOX is excluded. No benchmark metric in this repository is a placeholder.

## Start in Colab

1. Run [`00_prepare_visdrone.ipynb`](notebooks/00_prepare_visdrone.ipynb).
2. Choose either the preserved LR workflow (01 → 02 → 03) or one model’s HPO
   and final pair below.
3. Change only the small parameter cell and run all cells.
4. After interruption, reopen the same notebook and run all cells again.

All artifacts are discovered automatically under
`/content/drive/MyDrive/visdrone_architecture_benchmark`; users never copy a
checkpoint path, run ID, study name, or configuration between notebooks.

| Model | HPO | Baseline + tuned final runs |
|---|---|---|
| ResNet-50 | [`10_hpo_resnet50.ipynb`](notebooks/10_hpo_resnet50.ipynb) | [`20_finetune_resnet50.ipynb`](notebooks/20_finetune_resnet50.ipynb) |
| Swin-T | [`11_hpo_swin_t.ipynb`](notebooks/11_hpo_swin_t.ipynb) | [`21_finetune_swin_t.ipynb`](notebooks/21_finetune_swin_t.ipynb) |
| VMamba-T | [`12_hpo_vmamba_t.ipynb`](notebooks/12_hpo_vmamba_t.ipynb) | [`22_finetune_vmamba_t.ipynb`](notebooks/22_finetune_vmamba_t.ipynb) |
| RT-DETRv2 | [`13_hpo_rtdetrv2.ipynb`](notebooks/13_hpo_rtdetrv2.ipynb) | [`23_finetune_rtdetrv2.ipynb`](notebooks/23_finetune_rtdetrv2.ipynb) |

Then use [`30_evaluate_all_models.ipynb`](notebooks/30_evaluate_all_models.ipynb)
and [`31_publish_results.ipynb`](notebooks/31_publish_results.ipynb).

## Protocols and tracks

- `lr_controlled_v1` preserves the seed-42, LR-only workflow in notebooks 01–03.
- `two_stage_random_hpo_v1` runs five broad and five refined random trials, then
  baseline and tuned final recipes at seeds 17, 42, and 3407.
- Track A (`2class`) preserves the PERSON/VEHICLE collapse. Track B (`10class`)
  preserves the original ten classes. Their mAP values are never compared.

## Hosted Colab environments

Each model family receives a pinned, content-addressed environment below
`/content/visdrone_model_envs`; one family cannot overwrite another. Exact
versions, revisions, and licenses are in
[`configs/runtime_environments.yaml`](configs/runtime_environments.yaml).

VMamba requires `$DRIVE_ROOT/pretrained/vmamba_tiny_e292.pth`; training it from
scratch is disabled. CPU repository tests do not claim GPU compatibility. The
real model must pass construction, one-small-batch train/predict, checkpoint
save, and checkpoint reload before search.

## Persistent artifacts

Drive stores datasets, environment manifests, Optuna studies, selected configs,
checkpoints, registry files, predictions, evaluation, profiling, reports, and
result bundles. Datasets, checkpoints, predictions, logs, credentials, and
executed notebook outputs are never committed.

See [`docs/RUN.md`](docs/RUN.md), [`docs/ENVIRONMENTS.md`](docs/ENVIRONMENTS.md),
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md), and
[`docs/RESULTS.md`](docs/RESULTS.md).

## Verification

```bash
ruff check src scripts tests
pytest -q
python -m compileall -q src scripts tests
python scripts/validate_notebooks.py
python scripts/clean_notebooks.py --check notebooks
python scripts/run_notebook_smoke.py --timeout 180
python scripts/validate_doc_links.py
python scripts/scan_repository_secrets.py
python -m scripts.validate_results --repo-results results/
```

Project code is MIT licensed. Upstream assets retain their own terms; see
[`LICENSES.md`](LICENSES.md). Treat VisDrone use and derived results as
research-only unless separate permission is confirmed.
