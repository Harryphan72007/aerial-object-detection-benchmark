# Current benchmark progress

## Completed fixes

- The current Swin-T integration uses pinned MMDetection source rather than the
  retired timm feature wrapper. Runtime dataset resize transforms follow the
  requested experiment size, avoiding the old 224-versus-320/640 mismatch and
  NHWC-to-NCHW handoff.
- The current VMamba-T integration imports the pinned official package through
  its registration module. It no longer dynamically executes a single file as
  `vmamba_official`, so the old `sys.modules` partial-import defect is absent.
- All canonical Drive roots and run directories are created before manifests,
  checkpoints, metrics, plots, or SQLite files are written.
- Smoke HPO state uses a `smoke_test/study_smoke.db` namespace; full HPO retains
  the existing `study.db` path consumed by final and comparison workflows.
- Learning rate is the only Optuna suggestion. Trials have independent run
  directories and explicitly disable resume, which gives each candidate a new
  model, optimizer, scheduler, and gradient scaler.
- Optuna studies are persistent and loaded with `load_if_exists=True`. The runner
  counts finite `COMPLETE` trials, retries after `PRUNED` candidates up to a
  bounded attempt limit, and never deletes prior trials.
- RT-DETRv2 numerical divergence and CUDA OOM are recorded and pruned. Other
  exceptions remain fatal so implementation bugs are not hidden.
- Final tuned runs reload the selected LR, start from the original pretrained
  model, and select the single canonical `best.pth` by mAP@[0.50:0.95] on a
  held-out model-selection split carved from official train (disjoint from the
  search subsets); official validation is evaluated once, at the end, and never
  drives selection. New runs do not write duplicate compatibility aliases.

The older seven-notebook request referenced custom timm Swin and single-file
VMamba constructors. The maintained repository now centralizes these behaviors
in `scripts/run_mmdetection.py`, `scripts/run_rtdetr_training.py`, and
`src/hpo/`; obsolete notebook-local implementations were not reintroduced.

## Current RT-DETR LR-search status

The researcher reported one short trial near `1.02535e-5` as `COMPLETE`, with
validation mAP near `0.1727`. This is a short LR-search measurement and **not** a
final model result. A candidate near `3.68086e-4` previously produced NaN model
predictions inside the Hungarian matcher before divergence handling was added.
An equivalent future failure is recorded as `PRUNED` with its LR, failure reason,
failure type, and elapsed training time.

The controlled RT-DETRv2 range remains `1e-6` through `5e-4` in both phases. No
local database or GPU artifact was available in this checkout to independently
verify the reported trial values, and no full GPU completion is claimed.

## External RT-DETR training observation

A researcher-provided legacy notebook snapshot contains 40 completed epochs of
a two-class RT-DETRv2 R50/L run at seed 42. Validation mAP peaked at 0.211 after
the first epoch and declined to 0.127 by epoch 39; execution was manually
interrupted during epoch 40. The persistent 39.7% decline is consistent with
over-fine-tuning, although the absent training-loss output prevents a conclusive
classic-overfitting diagnosis.

This observation is **not a project benchmark result**. It uses the same R50vd
(RT-DETRv2-L) architecture the repository now trains as `rtdetrv2_l`, so it is
architecturally comparable, but it remains methodologically incompatible: batch
size 2 rather than the controlled effective batch 8, one final seed rather than
three, and official validation for repeated checkpoint selection. It must not be published
under `results/` or compared with canonical runs. See the full
[compatibility, evaluation, and disposition note](reference/rtdetrv2_legacy_training_status.md).

## Optuna states

- `COMPLETE`: training returned finite validation mAP and APtiny objectives.
- `PRUNED`: an expected numerical-divergence or CUDA-memory candidate was safely
  stopped and recorded. It does not invalidate completed trials.
- `FAIL`: an unexpected error occurred. The workflow re-raises it and stops so
  the implementation can be inspected.

## Known numerical behavior

At high learning rates, RT-DETRv2 can emit NaN or infinite predicted boxes before
the Hungarian matcher. Messages containing `nan`, `non-finite`, `not finite`,
`infinite`, `boxes1 must be`, or `boxes2 must be` are treated as divergence for
RT-DETRv2. Ground-truth annotation validation remains a separate data-contract
gate.

## Remaining experiments

1. Run notebook 00, then the GPU adapter smoke gate
   (`python -m scripts.gpu_adapter_smoke`) for each model on the target GPU. HPO
   and final training refuse to start without its READY record; see
   [the hosted GPU runbook](release/HOSTED_GPU_RUNBOOK.md).
2. Resume/complete five finite Phase A and five finite Phase B LR trials for each
   model and selected dataset track.
3. Run the default final matrix: the **tuned** recipe at seed **42**
   (`FULL_MATRIX = False`). The `baseline`+`tuned` × seeds 17/42/3407 matrix is an
   explicit opt-in (`FULL_MATRIX = True`) and is reported separately as the
   variance estimate the single-seed headline lacks.
4. Evaluate compatible final checkpoints, profile them, and generate the
   track-specific comparison report.
5. Measure `t_iter` with `scripts/measure_throughput.py` before quoting any
   GPU-hour figure.

## Expected artifact locations

All paths are below
`/content/drive/MyDrive/visdrone_architecture_benchmark`:

- Full studies: `hpo/two_stage_random_hpo_v1/<model>/<track>/study.db`
- Smoke studies:
  `hpo/two_stage_random_hpo_v1/<model>/<track>/smoke_test/study_smoke.db`
- Trial runs: `hpo/two_stage_random_hpo_v1/<model>/<track>/trials/`
- Final checkpoints: `checkpoints/final/<model>/<run_id>/`
- Evaluation: `evaluation/`
- Reports: `reports/`

The original full-study and final-checkpoint paths remain unchanged for notebooks
30 and 31 and for the comparison workflows.

## Safe resume settings

Run notebook 00 first and continue only after `DATA CONTRACT VERIFIED: YES`.
For HPO, open the matching notebook 10-13, keep the correct `DATASET_TRACK`, and
set `START_HPO = True`. Re-running the notebook loads its SQLite study and only
runs missing finite trials. Do not delete or recreate the database.

For final training, open the matching notebook 20-23 and set
`START_FINETUNING = True`. A new configuration starts from pretrained weights.
Only an interrupted run with the same model, track, seed, image size, effective
batch size, LR/configuration hash, and scheduler contract may resume from
`last.pth`.

Final training reads `final_train_seed42.json` — official train minus the 5%
model-selection holdout — not the complete official train split. The holdout
exists so the canonical `best.pth` is never selected on official validation.

Model environments now use a `READY`-only transactional reuse contract with exact
RT-DETR/OpenMMLab/VMamba probes. A failed child verifier records its stage,
stdout/stderr, Python, command and probe path; preview mode does not install the
family runtime. See `docs/reference/environment_provisioning_audit.md`.

A missing isolated environment is a hard failure: `model_python_executable()` no
longer falls back to `sys.executable`, so model training can never start in the
notebook kernel and produce misleading import errors.
