# Notebook-local logic deprecation

The package-backed notebooks are now the only supported entry points. They expose a small
parameter cell, bootstrap the repository, and call versioned code under `src/`. Model
construction, training, checkpoint/resume, artifact handling, evaluation, and comparison
logic must not be implemented in notebook cells.

## Canonical replacements

| Historical responsibility | Canonical entry point |
|---|---|
| Environment and repository setup | the bootstrap cell every notebook shares |
| Dataset preparation and verification | the model notebook, or `scripts/prepare_dataset.py` |
| GPU adapter smoke gate | the model notebook, or `scripts/gpu_adapter_smoke.py` |
| Model-specific HPO | `notebooks/10_resnet50.ipynb` through `13_rtdetrv2.ipynb` |
| Final training | the same four notebooks |
| Versioned evaluation and publication | `notebooks/30_report.ipynb` |

Two rounds of consolidation produced this list.

The `lr_controlled_v1` notebooks (`00_bootstrap_colab`, `01_run_model_day`,
`02_publish_results`, `03_compare_all_models`) were deleted with that protocol. They had
already stopped being runnable; keeping unrunnable entry points on disk only made it
harder to tell which workflow was live.

The twelve-run flow (`00_prepare_visdrone`, `10`–`13` HPO, `20`–`23` finetune,
`30` evaluate, `31` publish) then collapsed into one notebook per model plus one
report. Every stage was already idempotent or resumable, so running them as one
ordered sequence removed eleven manual launches and the ordering mistakes they
invited, without changing any stage's contract.

## Compatibility and operation

Users open notebooks directly from GitHub in Colab. Dataset archives are downloaded and
verified into the artifact root by the pipeline, so no browser upload or manual notebook
transfer is required. Rerunning a canonical notebook discovers versioned state
automatically.

Legacy checkpoints, manifests, predictions, and metrics remain readable through the
compatibility modules documented in `docs/compatibility/legacy_artifact_contract.md`.
Archived notebook source is retained through Git history and is not executable policy.

Rollback is limited to reverting the canonical notebook files. Package implementations
and legacy artifact readers remain intact.
