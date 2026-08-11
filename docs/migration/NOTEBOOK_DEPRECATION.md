# Notebook-local logic deprecation

The package-backed notebooks are now the only supported entry points. They expose a small
parameter cell, bootstrap the repository, and call versioned code under `src/`. Model
construction, training, checkpoint/resume, artifact handling, evaluation, and comparison
logic must not be implemented in notebook cells.

## Canonical replacements

| Historical responsibility | Canonical entry point |
|---|---|
| Environment and repository setup | the bootstrap cell every notebook shares |
| Dataset preparation and verification | `notebooks/00_prepare_visdrone.ipynb` |
| Model-specific HPO | `notebooks/10_hpo_resnet50.ipynb` through `13_hpo_rtdetrv2.ipynb` |
| Final training | `notebooks/20_finetune_resnet50.ipynb` through `23_finetune_rtdetrv2.ipynb` |
| Versioned evaluation and publication | `notebooks/30_evaluate_all_models.ipynb` and `31_publish_results.ipynb` |

The planned 04–06 responsibilities were absorbed by model-specific 10–23 notebooks and
the versioned 30–31 evaluation flow; creating placeholder notebooks would produce a
second, ambiguous workflow.

The `lr_controlled_v1` notebooks (`00_bootstrap_colab`, `01_run_model_day`,
`02_publish_results`, `03_compare_all_models`) were deleted with that protocol. They had
already stopped being runnable; keeping unrunnable entry points on disk only made it
harder to tell which workflow was live.

## Compatibility and operation

Users open notebooks directly from GitHub in Colab. Dataset archives are downloaded and
verified into Drive by notebook 00, so no browser upload or manual notebook transfer is
required. Rerunning a canonical notebook discovers versioned state automatically.

Legacy checkpoints, manifests, predictions, and metrics remain readable through the
compatibility modules documented in `docs/compatibility/legacy_artifact_contract.md`.
Archived notebook source is retained through Git history and is not executable policy.

Rollback is limited to reverting the canonical notebook files. Package implementations
and legacy artifact readers remain intact.
