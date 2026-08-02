# Single-checkpoint lifecycle audit

Audit baseline: `main` at `c0fb08bdb115c7453733af8138fcb2540a8e5afd`.

No dataset, pretrained source weight, Optuna database, prediction, completed
experiment metadata, or existing checkpoint was modified during this audit. The
cleanup command described below is dry-run by default and was not applied to Drive.

## 1. Current checkpoint creation map

| Entry point | Backend | Baseline behavior on Drive | Why it accumulated |
|---|---|---|---|
| `10_hpo_resnet50.ipynb` | MMDetection | Up to five `epoch_*.pth`, `last.pth`, mAP best, APtiny best | Built-in hook used `max_keep_ckpts=5` and two `save_best` metrics; HPO run directory was persistent |
| `11_hpo_swin_t.ipynb` | MMDetection | Same as notebook 10 | Shared HPO and backend path |
| `12_hpo_vmamba_t.ipynb` | MMDetection/VMamba | Same as notebook 10 | Shared HPO and backend path |
| `13_hpo_rtdetrv2.ipynb` | RT-DETR | `last.pth`, `best_raw.pth`, `best_map.pth`, `best_aptiny.pth`, `best.pt` | Best events copied the rolling checkpoint to several names; trial directory was persistent |
| `20_finetune_resnet50.ipynb` | MMDetection final | Epoch/best files plus `last.pth`, `best_map.pth`, `best_aptiny.pth`, `latest.pt`, `best.pt` | Backend normalization and final-workflow aliases both copied weights |
| `21_finetune_swin_t.ipynb` | MMDetection final | Same as notebook 20 | Shared final workflow |
| `22_finetune_vmamba_t.ipynb` | MMDetection/VMamba final | Same as notebook 20 | Shared final workflow |
| `23_finetune_rtdetrv2.ipynb` | RT-DETR final | `last.pth`, `best_raw.pth`, `best_map.pth`, `best_aptiny.pth`, `best.pt`, `latest.pt` | RT-DETR selection aliases plus final-workflow aliases |
| `30_evaluate_all_models.ipynb` | Evaluation | Created no weights; read registry `checkpoint_best_map` | Filename-specific registry consumer |
| `31_publish_results.ipynb` | Publication | Did not publish weights, but validated `checkpoint_best_map` hashes | Filename-specific registry consumer |

Baseline and tuned final experiments already use separate run/seed directories.
Smoke runs use isolated directories but previously retained their backend files.

## 2. Proposed and implemented lifecycle

```text
ACTIVE FINAL RUN
  last.pth
    current model + optimizer + scheduler + AMP scaler + EMA (when enabled)
    epoch + optimizer updates + RNG + sampler + early stopping
    best-metric state + selected raw model state + configuration identity

COMPLETED FINAL RUN
  best.pth
    load-verified selected raw weights + v2 checkpoint identity

INTERRUPTED OR FAILED FINAL RUN
  last.pth
    cleanup is never entered

HPO TRIAL
  /content/visdrone_hpo_trials/... while running
  no Drive checkpoint after objective finalization
  Drive retains study DB/snapshot, trial_record.json, sanitized final_metrics.json,
  applied overrides, runtime metadata, and failure/warning information
```

The authoritative selection metric is validation mAP. APtiny remains in metrics and
Optuna's secondary objective but never creates another checkpoint. RT-DETR declares
`weight_variant: raw`; MMDetection also promotes raw runner weights.

## 3. Files changed

- `src/training/checkpointing.py`: v2 manifest contract, legacy resolver, load and
  identity verification, canonical best extraction, checksum, bounded one-file
  enforcement, and registry compatibility.
- `src/training/checkpoint_selection.py`: canonical-first resolution and no new
  alias materialization.
- `scripts/run_rtdetr_training.py`: one atomic rolling resume file, embedded
  best-mAP raw state, canonical `best.pth`, identity/checksum, metric-only APtiny.
- `scripts/run_mmdetection.py`: disables MMEngine's epoch/best history hook;
  installs one atomic rolling full-state hook and embeds one best-mAP raw state.
- `src/training/trainer.py`: v2 manifests, backend scientific contract checks,
  interrupted status, completed-manifest boundary, and final-run cleanup.
- `src/hpo/workflow.py` and `src/hpo/rtdetr_v2.py`: local scratch trials with
  `resume=False`, persistent non-weight records, mandatory scratch removal, and
  documented storage policy.
- `src/hpo/final_workflow.py`: removes `latest.pt` and `best.pt` creation; selected
  HPO output remains hyperparameters only and final training starts pretrained.
- `scripts/evaluate.py`, `src/hpo/result_bundle.py`, `src/result_export.py`, and
  `src/benchmark_status.py`: manifest-selected canonical resolution with legacy
  fallback.
- `src/workflows/model_day.py`: deletes successful adapter-smoke model files after
  reload validation.
- `scripts/cleanup_checkpoints.py`: separate dry-run-first migration command with
  completion, readability, checksum, boundary, and incomplete-run guards.
- `schemas/run_manifest_v2.schema.json`: canonical checkpoint manifest schema.
- `schemas/legacy/notebook_artifact_inventory_v1.json`: regenerated frozen
  notebook/source artifact inventory for the changed checkpoint references.
- `docs/CURRENT_PROGRESS.md`, `docs/models/checkpoint_selection.md`,
  `docs/reference/checkpoint_format.md`, and
  `docs/compatibility/legacy_artifact_contract.md`: document the new policy,
  manifest version, canonical resolver, and frozen v1 read compatibility.
- `tests/test_checkpoint_lifecycle_v2.py`: adds lifecycle, failure-injection,
  cleanup-boundary, legacy-resolution, HPO, baseline/tuned, smoke, evaluation,
  and static backend policy coverage.
- `tests/test_checkpoint_selection_early_stopping.py`,
  `tests/test_hpo_workflow.py`, `tests/test_runtime_hardening.py`, and
  `tests/test_training_backend_launch.py`: update existing contracts for the
  canonical file and local-scratch behavior.

## 4. Validation classification

CPU/static validation covers atomic replacement logic, exact file counts, identity
metadata, checksum enforcement, failure retention, cleanup safety, HPO scratch
deletion/persistence, manifest resolution, lint, schemas, and notebook contracts.
The completed local validation run reported 249 passed and 2 dependency-gated
skips. Ruff, notebook validation/cleanliness, all 12 notebook smoke executions,
documentation links, repository secret scanning, result validation, JSON schema
loading, and `git diff --check` also passed.

CPU tests cannot establish CUDA/MMCV operator correctness or prove MMEngine behavior
under a real Colab disconnect. One RT-DETR and one MMDetection Colab GPU finalization
smoke remain required before launching the complete search/final-training matrix.

## 5. Migration compatibility

Existing manifests and directories are not rewritten. Readers try, in order:

1. `best.pth`;
2. `best_map.pth`;
3. `best_raw.pth`;
4. `best.pt` and `best_aptiny.pth` only when legacy aliases are explicitly enabled;
5. `last.pth`/`latest.pt` only for an explicitly requested resume read.

The frozen v1 manifest fields remain accepted. New v2 manifests write
`checkpoint_best` and do not require duplicate paths.

## 6. Cleanup dry-run contract

```bash
python -m scripts.cleanup_checkpoints \
  --drive-root /content/drive/MyDrive/visdrone_architecture_benchmark \
  --dry-run
```

The JSON report defaults to
`reports/checkpoint_cleanup_latest.json`. Each run entry contains its run ID,
status, selected source, canonical destination, checksum, planned removals, actual
removals, skipped files, warnings, and errors. Dry-run leaves `removed_files` empty.
`--apply` is required for mutation. Incomplete, failed, interrupted, invalid,
unreadable, or out-of-boundary runs are never modified. Completed HPO trial weights
are reported as disposable; Optuna databases are outside the deletion set.

No real Drive cleanup was executed as part of the code migration.

## 7. Remaining risks

- A Colab disconnect between epochs preserves the last fully replaced `last.pth`;
  work after that atomic boundary may repeat.
- Drive latency makes a full-state atomic replacement slower and temporarily needs
  space for both old and temporary files.
- A completely full Drive can prevent the temporary checkpoint from being written;
  the previous valid `last.pth` remains.
- Atomic replacement depends on source and destination being on the same filesystem;
  all rolling writes deliberately use a temporary file in the run directory.
- The embedded selected state increases the single active checkpoint size because it
  carries current resume weights and earlier best weights in one file.
- MMEngine 0.10.7 and the pinned MMDetection/CUDA stack still require a real GPU smoke
  to validate hook order, AMP optimizer-wrapper state, and framework checkpoint load.
- Cleanup readability validation requires the matching PyTorch/framework environment;
  dry-run should be inspected before `--apply`.
