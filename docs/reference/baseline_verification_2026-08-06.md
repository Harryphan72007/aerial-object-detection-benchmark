# Baseline verification — 2026-08-06

Ground-truth verification run **before** any remediation-plan code changes, so every
later PR can be measured against it (a PR must not turn a passing check red).

- **Base commit:** `4cec87f` (branch `codex/single-checkpoint-lifecycle`; remediation
  work branched from here as `remediation/full-plan`).
- **Environment:** Windows 11, Python 3.11, `torch 2.10.0+cpu`, `pytest` present.
  No CUDA/GPU. Optional deps `optuna` and `mmengine` are **not** installed.

## Results

| Command | Exit | Notes |
|---|---|---|
| `python -m pytest -q` | **fail (1)** | 301 passed, 6 skipped, 1 failed in ~259 s |

### The 6 skips — all missing optional dependencies (not repo faults)

| Test | Cause |
|---|---|
| `tests/test_hpo_workflow.py` (×3) | `optuna` not installed |
| `tests/test_rtdetr_hpo_v2.py` | `optuna` not installed |
| `tests/test_runtime_hardening.py` | `optuna` not installed |
| `tests/test_mmdetection_pinned_integration.py` | `mmengine` not installed |

### The 1 failure — environment-caused, not repo logic

`tests/test_workflow_utils.py::test_atomic_checkpoint_replacement`

```
OSError: [Errno 9] Bad file descriptor
  src/training/checkpointing.py:258 -> os.fsync(handle.fileno())
```

The atomic-checkpoint helper opens the temp file and calls `os.fsync(handle.fileno())`.
On this Windows/Python build that `fsync` raises `EBADF`. This is a **platform**
behaviour of the test environment, not a defect introduced by the repository, and it
is unrelated to any remediation change. Treated as a known-red baseline: remediation
PRs must not introduce *new* failures beyond this one.

## Commands not run and why

- `ruff`, `scripts/validate_notebooks.py`, `scripts/clean_notebooks.py`,
  `scripts/run_notebook_smoke.py`, `scripts/validate_doc_links.py`,
  `scripts/scan_repository_secrets.py`, `scripts/verify_model_environments.py`,
  `python -m scripts.validate_results` — not exercised in this baseline pass; the
  pytest suite is the primary regression signal used to gate the no-GPU PRs. Any PR
  that touches those surfaces re-runs the relevant checker.
- All GPU paths — no CUDA in this environment. PR-07/08/09/13 GPU portions are
  implemented but must be validated on Colab/L4 hardware by the owner.
