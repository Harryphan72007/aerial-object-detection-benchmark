# Environment provisioning audit and remediation

Audit baseline: remote `main` at `c0fb08bdb115c7453733af8138fcb2540a8e5afd`.

No dataset, checkpoint, prediction, Optuna database, pretrained checkpoint, or
completed experiment manifest was modified. CPU tests used temporary directories.

## 1. Root cause

The exact package or compiled symbol that failed in the original Colab session is
not recoverable from the screenshot. The parent used
`subprocess.run(..., check=True)` without capturing child output, and the failed
child's `environment_probe.json` was not available. The original verifier could
have failed in its package-prefix check, CUDA check, or environment collection.
Rerun notebook 12 after this patch to capture the exact stage, stdout, stderr, and
structured probe.

Three code defects are confirmed:

1. `benchmark_runtime.json` was written as `PACKAGES_INSTALLED` before the probe.
   A matching hash and an existing Python executable then made `packages_changed`
   false, so a failed or interrupted runtime could be reused.
2. The shared `_run` helpers discarded diagnostic context. Colab therefore showed
   only `CalledProcessError: returned non-zero exit status 1`.
3. VMamba invoked `--environment openmmlab`. That checked only package-name
   prefixes and GPU visibility, not the VMamba revision, detector configuration,
   registration module, `selective_scan_cuda`, pretrained checkpoint, or model
   construction.

The eight HPO/final notebooks also called provisioning before checking their
`START_HPO`/`START_FINETUNING` flag, so preview mode installed the full framework.

## 2. Shared-impact analysis

Notebooks 10, 11, 12, 13, 20, 21, 22, and 23 shared the eager provisioning and
generic `CalledProcessError` paths. Notebook 01 reaches the same provisioning API
through `run_model_day`; its preview guard was already before provisioning.
Notebook 30 reaches it through `scripts.evaluate_all_models` only when missing
evaluations are requested. Notebook 31 publishes metadata and does not construct a
model runtime.

The confirmed reuse defect affected all three runtime families. The missing
compiled MMCV check affected ResNet, Swin, and VMamba. The missing
`selective_scan_cuda`, source, pretrained, and construction gates affected only
VMamba. A stale `VISDRONE_MODEL_PYTHON` could affect the next model after a failed
setup; selection variables are now cleared before setup and set only after `READY`.

## 3. Findings table

| Severity | Confidence | File/function | Trigger | Previous behavior | Intended behavior / smallest fix | Impact | Regression test | Rollback |
|---|---|---|---|---|---|---|---|---|
| Critical | Confirmed | `src/workflows/isolated_environment.py::provision_isolated_environment` | Probe fails after installation | Matching hash/executable reused attempted install | Atomic `CREATING`/`INSTALLING`/`VERIFYING`/`READY`/`FAILED`; reuse only `READY` plus quick probe | Broken CUDA runtime repeatedly selected | Failed, interrupted, ready, stale-ready tests | Revert transactional provisioner commit; local hash creates an independent runtime |
| High | Confirmed | former `_run` helpers | Any child exits nonzero | Generic `CalledProcessError` | Shared `run_checked` includes redacted command, cwd, code, stdout, stderr, family, Python, stage, probe and log | Root cause hidden in Colab | Fake exit-code-1 diagnostic test | Restore direct subprocess calls |
| Critical | Confirmed | `scripts/verify_model_environments.py` | VMamba setup | Generic OpenMMLab prefix probe marked package stack verified | Explicit `vmamba` mode after exact OpenMMLab and compiled-op checks; verify revision, config, registration, selective scan, checkpoint and construction | HPO could begin with unusable VMamba stack | VMamba base+specific, missing extension/checkpoint tests | Disable `--construct-model` only; retain fail-closed package/source gates |
| High | Confirmed | same verifier | ABI-incompatible MMCV wheel | Version could look correct while `_ext` failed | Import `mmcv.ops.nms` | Training failed later | Compiled-op failure injection | Revert compiled-op import only |
| High | Confirmed | same verifier | Wrong exact Python/Torch/CUDA build | Prefix checks accepted some incompatible builds | Enforce runtime spec exactly | Binary incompatibility and missing GPU | Family command and contract tests | Change pins in one runtime YAML revision, producing a new hash |
| Medium | Confirmed | eight HPO/final notebooks | Start flag is false | Multi-GB installation in preview | Return `SKIPPED_PREVIEW`; provision only when start flag is true | Cost, latency and avoidable failures | Eight-notebook source-contract test | Restore one expression per notebook |
| High | Confirmed | `src/workflows/adapter_gate.py` | Same fingerprint after `FAILED_ENVIRONMENT` | Persistent gate returned `blocked` | Retry through transactional provisioner | Rerun could not repair runtime | Adapter-gate focused test | Restore blocked decision |
| Medium | Confirmed | process environment selection | Prior model setup failed or next family started | Stale model Python/source variables remained | Clear selection before setup; publish variables only after `READY` | Wrong child Python or framework root | Cross-family variable test | Restore previous environment selection |
| High | Confirmed | local runtime deletion | Partial runtime repair | No deterministic repair boundary | Delete only the direct hashed child of configured local runtime base | Prevent accidental Drive deletion | Boundary-escape refusal test | Disable repair deletion; a new runtime hash remains safe |

## 4. Patch summary

- `src/subprocess_utils.py`: adds redacted checked-process diagnostics and optional
  atomic logs while preserving the headless child-environment policy.
- `src/workflows/isolated_environment.py`: implements transactional state,
  READY-only quick-probe reuse, bounded local rebuild, exact source preparation,
  family verification, VMamba extension installation, success-only persistent
  manifest creation, and post-READY environment selection.
- `scripts/verify_model_environments.py`: adds structured `rtdetr`, `openmmlab`,
  and `vmamba` probes with failure-stage JSON output.
- `src/workflows/environment.py`: delegates the complete Colab lifecycle to the
  transactional provisioner instead of marking a generic base probe sufficient.
- `src/notebook_utils.py`: makes the legacy direct preflight exact and adds VMamba
  compiled-extension validation.
- `src/workflows/adapter_gate.py`: permits deterministic retries after an
  environment failure.
- `src/workflows/model_day.py`, `scripts/evaluate_all_models.py`, and notebook 30:
  use shared checked-process diagnostics for model-family subprocesses.
- HPO notebooks 10-13 and final notebooks 20-23: skip environment installation in
  preview mode and retain the start flag as the explicit authorization boundary.
- `scripts/validate_notebooks.py`: rejects inline package/venv setup in canonical
  model notebooks and requires the shared environment API in direct model notebooks.
- `schemas/benchmark_runtime_v2.schema.json`: documents the transactional marker.
- `tests/test_environment_provisioning.py` and existing environment/adapter tests:
  cover diagnostics, states, repair, family probes, preview, Drive safety and
  environment isolation.

## 5. Notebook matrix

| Notebook | Model | Family | Setup API | Child Python | Verification | GPU | Compiled extensions | Required pretrained source | Same failure possible now? | Recommended behavior |
|---|---|---|---|---|---|---|---|---|---|---|
| 01 | User-selected | Derived | `run_model_day` -> shared API | Hashed family runtime | Family full probe + adapter gate | Yes | MMCV; selective scan for VMamba | Model contract | Diagnosed and repairable | Preview with start false; adapter smoke when true |
| 10 | ResNet-50 | openmmlab | `ensure_model_environment` | `openmmlab-<hash>/.../python` | Exact packages, CUDA, MMDet revision, `mmcv.ops.nms` | Yes | MMCV | MMDetection COCO initialization | Diagnosed and repairable | Provision only when `START_HPO=True` |
| 11 | Swin-T | openmmlab | Same | Same family runtime | Same plus later adapter construction | Yes | MMCV | Official Swin initialization | Diagnosed and repairable | Provision only when `START_HPO=True` |
| 12 | VMamba-T | vmamba | Same | `vmamba-<hash>/.../python` | OpenMMLab base + complete VMamba probe | Yes | MMCV + `selective_scan_cuda` | Non-empty `vmamba_tiny_e292.pth` | Diagnosed and repairable | HPO blocked until complete probe passes |
| 13 | RT-DETRv2-L | rtdetr | Same | `rtdetr-<hash>/.../python` | Exact packages/CUDA/classes + pinned HF revision | Yes | PyTorch CUDA | Pinned `PekingU/rtdetr_v2_r101vd` | Diagnosed and repairable | Provision only when `START_HPO=True` |
| 20 | ResNet-50 | openmmlab | Same | OpenMMLab runtime | Same as 10 | Yes | MMCV | Same as 10 | Diagnosed and repairable | Provision only when `START_FINETUNING=True` |
| 21 | Swin-T | openmmlab | Same | OpenMMLab runtime | Same as 11 | Yes | MMCV | Same as 11 | Diagnosed and repairable | Provision only when `START_FINETUNING=True` |
| 22 | VMamba-T | vmamba | Same | VMamba runtime | Same as 12 | Yes | MMCV + selective scan | Same as 12 | Diagnosed and repairable | Provision only when `START_FINETUNING=True` |
| 23 | RT-DETRv2-L | rtdetr | Same | RT-DETR runtime | Same as 13 | Yes | PyTorch CUDA | Same as 13 | Diagnosed and repairable | Provision only when `START_FINETUNING=True` |
| 30 | Each missing run | Per model | `scripts.evaluate_all_models` -> shared API | Re-selected per run | Family probe before evaluation | Yes | Per family | Per model | Diagnosed and repairable | No provisioning unless `EVALUATE_MISSING=True` |
| 31 | None | None | Publication only | Notebook Python | No model probe | No | None | None | No | Never provision a model runtime |

## 6. Tests actually executed

The completed local CPU/static validation was:

```text
267 passed, 2 skipped because the local environment does not include PyTorch
Ruff: passed
Notebook validation: passed
Notebook cleanliness: passed
14 guarded notebook smoke executions: passed
Documentation links: passed
Repository secret scan: passed
Result validation: passed
JSON schema loading and git diff check: passed
```

No Colab or GPU test has been executed locally. CPU mocks do not establish CUDA,
MMCV ABI, selective-scan ABI, model-memory, or real forward/backward correctness.

## 7. Manual Colab validation

In a fresh GPU Colab session, mount Drive, clone the reviewed commit, install the
shared notebook dependencies, and prepare the dataset. Then run this once per model:

```python
from pathlib import Path
from src.workflows.environment import ensure_model_environment

REPO_PATH = Path("/content/aerial-object-detection-benchmark")
DRIVE_ROOT = Path("/content/drive/MyDrive/visdrone_architecture_benchmark")

for model_id in (
    "faster_rcnn_resnet50",
    "faster_rcnn_swin_t",
    "faster_rcnn_vmamba_t",
    "rtdetrv2_l",
):
    runtime = ensure_model_environment(model_id, REPO_PATH, DRIVE_ROOT)
    assert runtime["state"] == "READY", runtime
    print(model_id, runtime["python_executable"], runtime["probe_path"])
```

For the adapter forward/backward gates, run notebook 01 separately for each model
with `RUN_MODE="environment"`, `START_EXPENSIVE_STAGE=True`, and `SMOKE_TEST=True`.
That gate constructs the selected model, trains/backpropagates/steps one small batch,
validates, atomically saves, reloads, and records its fingerprint. Specifically:

- ResNet: confirm `mmcv.ops.nms`, construct the detector, and complete the gate.
- Swin: confirm the pinned config/image size and complete one forward/backward gate.
- VMamba: confirm pinned revision, `selective_scan_cuda_oflex`, checkpoint size,
  VMamba-T Faster R-CNN construction, and one forward/backward gate.
- RT-DETR: resolve the pinned Hugging Face revision, construct the two-class model,
  and complete one forward/backward gate.

Archive `benchmark_runtime.json`, `environment_probe.json`, the adapter gate, and
`runtime_environment.json`. Together they record commit, GPU/driver, Python, Torch,
torchvision, CUDA build, framework versions, runtime hash, result, and log/probe
paths.

To reproduce the formerly hidden VMamba failure directly after provisioning, use
the exact command printed in `logs/vmamba_complete_probe.log`. The new exception and
`environment_probe.json` will identify the failing stage and child stderr.

## 8. Remaining risks

- **CUDA/driver:** exact wheel pins cannot guarantee compatibility with a future
  Colab host driver. The full GPU probe must be rerun on each new runtime hash.
- **Compiled extensions:** MMCV and selective scan can import successfully but fail
  on a particular kernel input. Only the real adapter forward/backward gate covers
  that behavior.
- **Checkpoint:** non-empty VMamba pretrained validation does not prove semantic
  compatibility; factory construction and the adapter smoke are the next gates.
- **Colab lifecycle:** a disconnect can leave `INSTALLING` or `VERIFYING`; the next
  run deletes only that hashed local runtime and rebuilds it. Framework source
  checkouts and all Drive experiment artifacts remain intact.
- **Network/cache:** pinned Git/Hugging Face revisions still require availability on
  the first setup. A READY quick RT-DETR probe uses the local Hugging Face cache.
- **Resource cost:** VMamba model construction is intentionally part of its full
  gate and can consume noticeable CPU RAM/time before HPO begins.
