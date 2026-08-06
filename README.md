# Aerial Object Detection Benchmark

Student-friendly, resumable VisDrone experiments for four detector families:

| `MODEL_ID` | Family |
|---|---|
| `faster_rcnn_resnet50` | Faster R-CNN + ResNet-50-FPN |
| `faster_rcnn_swin_t` | Faster R-CNN + Swin-T-FPN |
| `faster_rcnn_vmamba_t` | Faster R-CNN + VMamba-T-FPN |
| `rtdetrv2_l` | RT-DETRv2-L (R50) |

YOLOX is excluded. No benchmark metric in this repository is a placeholder.

## Current progress

- Swin-T uses the official MMDetection backbone and the runtime data pipeline is
  rewritten to the selected experiment resolution. The retired timm adapter's
  fixed 224-pixel/NHWC path is not used; smoke and full workflows remain 320- and
  640-pixel capable where selected by the workflow.
- VMamba-T uses the pinned official VMamba package registration path. The retired
  single-file `vmamba_official` dynamic loader (and its partial-import failure
  mode) is not used by the current notebooks.
- Smoke artifacts and Optuna storage are isolated from full experiments. Studies
  are persistent, use `load_if_exists=True`, and retain completed trials across
  interruptions.
- Learning rate is the only Optuna-suggested value. Every trial launches a fresh
  model, optimizer, scheduler, and scaler with resume disabled.
- RT-DETRv2 keeps the full `1e-6` to `5e-4` LR interval in both search phases.
  Expected numerical divergence and CUDA OOM candidates are recorded as
  `PRUNED`; unexpected implementation failures still stop the workflow.
- ResNet-50 and Swin-T use their base configs' ImageNet-pretrained backbones;
  VMamba-T uses its required classification checkpoint. Detector heads are
  newly initialized for the selected VisDrone track. Track A has two classes
  (`person`, `vehicle`); Track B has ten classes.

High LR candidates can be pruned because numerical divergence is a valid search
outcome, not evidence that annotations are invalid. Full GPU execution has not
been claimed by the repository's CPU smoke tests and must still be completed by
the researcher.

For a data smoke validation, set `SMOKE_TEST = True` in notebook 00 and keep
resume disabled. For the full HPO path, use `SMOKE_TEST = False`, run notebook 00,
then set `START_HPO = True` in the matching 10-13 notebook and
`START_FINETUNING = True` in its matching 20-23 notebook. New final runs begin
from pretrained weights; an interrupted final run resumes only when its complete
configuration contract matches.

## Start in Colab, Kaggle, or local Jupyter

1. Run [`00_prepare_visdrone.ipynb`](notebooks/00_prepare_visdrone.ipynb).
2. Choose either the preserved LR workflow (01 → 02 → 03) or one model’s HPO
   and final pair below.
3. Change only the small parameter cell and run all cells.
4. After interruption, reopen the same notebook and run all cells again.

These are thin, package-backed notebooks and are the only supported entry points. The
same files detect Colab, Kaggle, or local Jupyter automatically. The dataset setup
workflow downloads and verifies archives into the selected artifact root, so no
separate platform notebooks are required.
Historical notebook-local model, training, checkpoint, resume, artifact, and evaluator
implementations are deprecated; see
[`docs/migration/NOTEBOOK_DEPRECATION.md`](docs/migration/NOTEBOOK_DEPRECATION.md).

Artifacts are discovered automatically under Google Drive in Colab,
`/kaggle/working/visdrone_architecture_benchmark` in Kaggle, or
`local_artifacts` in a local checkout. `VISDRONE_DRIVE_ROOT` overrides all three;
users never copy a checkpoint path, run ID, study name, or configuration between
notebooks.

| Model | HPO | Baseline + tuned final runs |
|---|---|---|
| ResNet-50 | [`10_hpo_resnet50.ipynb`](notebooks/10_hpo_resnet50.ipynb) | [`20_finetune_resnet50.ipynb`](notebooks/20_finetune_resnet50.ipynb) |
| Swin-T | [`11_hpo_swin_t.ipynb`](notebooks/11_hpo_swin_t.ipynb) | [`21_finetune_swin_t.ipynb`](notebooks/21_finetune_swin_t.ipynb) |
| VMamba-T | [`12_hpo_vmamba_t.ipynb`](notebooks/12_hpo_vmamba_t.ipynb) | [`22_finetune_vmamba_t.ipynb`](notebooks/22_finetune_vmamba_t.ipynb) |
| RT-DETRv2-L | [`13_hpo_rtdetrv2.ipynb`](notebooks/13_hpo_rtdetrv2.ipynb) | [`23_finetune_rtdetrv2.ipynb`](notebooks/23_finetune_rtdetrv2.ipynb) |

RT-DETRv2-L is a full member of the four-family controlled benchmark. Its earlier
quarantine is lifted now that (a) the variant is the intended L/`r50vd` model, and
(b) it shares the identical controlled epoch budget as every other model. Before
trusting any RT-DETRv2 run, the
[GPU adapter smoke gate](docs/release/GPU_VALIDATION_CHECKLIST.md) must pass on
the target hardware, exactly as for the other three families — no full run should
start without a stored passing smoke record. The historical over-fine-tuning
observation (see
[`docs/reference/rtdetr_training_results_diagnosis_2026-08-03.md`](docs/reference/rtdetr_training_results_diagnosis_2026-08-03.md))
described a 25–40 epoch run selected on official validation; the controlled track
now trains only 8 epochs with early stopping.

Then use [`30_evaluate_all_models.ipynb`](notebooks/30_evaluate_all_models.ipynb)
and [`31_publish_results.ipynb`](notebooks/31_publish_results.ipynb).

## Protocols and tracks

- `lr_controlled_v1` preserves the seed-42, LR-only workflow in notebooks 01–03.
- `two_stage_random_hpo_v1` runs five Phase A and five Phase B LR-only random
  trials, then a single **tuned** final run at seed **42** on the `2class` track
  — the headline matrix that fits roughly one model per GPU-day. The full
  `baseline`+`tuned` × seed `17/42/3407` matrix (a default-LR diagnostic plus a
  multi-seed variance estimate) is an explicit opt-in (`full_matrix=True`),
  reported separately before any conclusions are drawn. A single-seed table
  cannot claim differences below seed noise (~0.3–0.8 mAP on VisDrone).
  RT-DETRv2 retains `1e-6` to `5e-4` in both phases; other models refine Phase B
  around the strongest finite Phase A candidates.
- Track A (`2class`) preserves the PERSON/VEHICLE collapse. Track B (`10class`)
  preserves the original ten classes. Their mAP values are never compared.

## Hosted notebook environments

Each model family receives a pinned, content-addressed environment below
`/content/visdrone_model_envs` in Colab or `/kaggle/working/visdrone_model_envs`
in Kaggle; one family cannot overwrite another. Exact versions, revisions, and licenses are in
[`configs/runtime_environments.yaml`](configs/runtime_environments.yaml).

VMamba requires `$DRIVE_ROOT/pretrained/vmamba_tiny_e292.pth`; training it from
scratch is disabled. CPU repository tests do not claim GPU compatibility. The
real model must pass construction, one-small-batch train/predict, checkpoint
save, and checkpoint reload before search.

## Persistent artifacts

The selected artifact root stores datasets, environment manifests, Optuna studies,
selected configs, checkpoints, registry files, predictions, evaluation, profiling,
reports, and result bundles. Datasets, checkpoints, predictions, logs, credentials,
and executed notebook outputs are never committed.

See [`docs/RUN.md`](docs/RUN.md), [`docs/ENVIRONMENTS.md`](docs/ENVIRONMENTS.md),
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md), and
[`docs/RESULTS.md`](docs/RESULTS.md). Current repair and GPU follow-up status is
tracked in [`docs/CURRENT_PROGRESS.md`](docs/CURRENT_PROGRESS.md).

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
