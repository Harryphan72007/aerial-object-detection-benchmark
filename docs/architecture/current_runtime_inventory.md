# Current runtime and dependency inventory

Status: migration PR 2 inventory. This document records the repository as it
exists and does not declare that any GPU family was executed successfully.

## Runtime layers

The repository does not have one safe universal environment. Its current
dependency contract has three layers:

| Layer | Consumers | Existing requirement source | Purpose |
|---|---|---|---|
| Shared notebook/data kernel | 00-03, 10-13, 20-23, 30-31, optional notebooks | `requirements-dataset-colab.txt`; frozen view in `requirements/legacy-colab.txt` | data, COCO conversion/evaluation, plots, notebook control flow |
| Shared HPO extension | 10-13 | `requirements-hpo-colab.txt` | Optuna orchestration; no detector binaries |
| Model-family subprocess | training/HPO workflow selected by model | `requirements-rtdetr-colab.txt` or `requirements-openmmlab-py310-cu118.txt` | mutually pinned framework and CUDA stacks |

`requirements-colab.txt` is a backward-compatible alias for the dataset stack.
`requirements-notebook-test.txt` adds static notebook execution dependencies for
development and CI, not model training.

## Notebook import inventory

PR 1 statically parsed all 17 notebooks. Direct third-party imports are:

```text
IPython.display
PIL
google.colab
matplotlib
numpy
pandas
sklearn.decomposition
torch
```

Standard-library imports are omitted here. `torch` is intentionally not pinned
in the shared legacy file because model families require different versions.
`google.colab` is supplied by hosted Colab. `scikit-learn` and IPython/tabulate
are recorded in `requirements/legacy-colab.txt` because optional analysis and
Markdown rendering assume them.

The machine-readable notebook and delegated Python-source inventory is
`schemas/legacy/notebook_artifact_inventory_v1.json`.

## Pinned model-family environments

The authoritative source is `configs/runtime_environments.yaml`.

| Family | Python | PyTorch / CUDA | Framework packages | External source |
|---|---|---|---|---|
| RT-DETRv2 | 3.11.13 | 2.7.1+cu128 | Transformers 4.52.4, Accelerate 1.7.0 | RT-DETR `a21d516aca15da57e65f35c47659c7535ad2b6b3`; PekingU weights `a558a4798734af61997652ec97d9b82961c92450` |
| ResNet-50 / Swin-T | 3.10.16 | 2.1.0+cu118 | MMDetection 3.3.0, MMEngine 0.10.7, MMCV 2.1.0 | MMDetection `44ebd17b145c2372c4b700bfb9cb20dbd28ab64a` |
| VMamba-T | 3.10.16 | 2.1.0+cu118 | OpenMMLab stack plus selective scan | VMamba `2ed52ead062a51a64521ed3871d52914bf532876` |

The environment provisioner is uv 0.8.15. Hosted Colab model environments are
content-addressed under `/content/visdrone_model_envs`. Runtime manifests are
written to `$DRIVE_ROOT/environment_manifests` by existing workflow code.

VMamba additionally assumes a non-empty
`$DRIVE_ROOT/pretrained/vmamba_tiny_e292.pth`, and its selective-scan CUDA
extension must import. Training from scratch is rejected.

## Dependency-to-capability map

| Dependency group | Capabilities currently using it |
|---|---|
| Pillow, NumPy | image loading, box conversion, fixtures |
| PyYAML | project, dataset, model, and runtime configuration |
| pandas, tabulate | comparisons, CSVs, Markdown tables |
| matplotlib | training curves, comparison plots, optional visualization |
| pycocotools | COCO mAP/recall evaluation |
| psutil | CPU RAM and process telemetry |
| filelock | atomic registry updates |
| Optuna | two-stage random HPO workflow |
| scikit-learn | optional PCA feature visualization |
| Transformers/Accelerate | RT-DETRv2 adapter and trainer |
| MMDetection/MMEngine/MMCV | Faster R-CNN ResNet/Swin/VMamba adapters |
| timm/einops/selective scan | Swin/VMamba backbones |

## Third-party repository and network assumptions

The current setup may access GitHub, PyPI, the PyTorch wheel index, the
OpenMMLab wheel index, and Hugging Face. Git, pip, and uv must be executable.
Pinned upstream clones are stored under `$DRIVE_ROOT/frameworks`; their Git
revision is checked out detached. No credentials are read by the diagnostic.

The repository URL is currently
`https://github.com/Harryphan72007/aerial-object-detection-benchmark.git`, and
most canonical notebooks hard-code branch `main`. Optional notebooks allow
`BENCHMARK_REPOSITORY_URL` and `BENCHMARK_REPOSITORY_BRANCH` overrides.

## Known inventory risks

- Import availability is not a model compatibility gate.
- The model families cannot safely be installed into one shared Python process.
- Several notebooks assume Colab-preinstalled `torch`, IPython, and
  `google.colab` rather than installing them directly.
- OpenMMLab requires its exact Python/PyTorch/CUDA/MMCV combination; a source
  build is intentionally refused.
- VMamba CUDA and selective-scan performance remain GPU-unverified in this PR.
- Existing version comments mention a designed Colab release; they are an
  inventory statement, not proof that the current hosted image is unchanged.

## Read-only diagnostic

Run without installing packages or constructing a model:

```bash
python -m scripts.diagnostics.report_current_environment
```

It reports repository identity, dirty state, configured paths, whitelisted path
environment variables, installed package metadata, pinned sources, notebook
imports, and `nvidia-smi` availability. It sets
`model_construction_performed=false` and never imports detector frameworks.
