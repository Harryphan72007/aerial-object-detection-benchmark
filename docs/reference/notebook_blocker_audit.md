# Notebook blocker audit

Audit date: 2026-07-30
Repair branch: `codex/fix-backend-imports`

| Confirmed blocker | Root cause | Repair |
|---|---|---|
| `No module named 'src'` | Child Python files launched by path | Repository children use `python -m`; CLI help tests cover training/evaluation/profile |
| PR notebook CI failure | Outputs, counts, widget/Colab metadata, debug output, expensive guard enabled | All notebooks cleaned and CI-enforced |
| Old failed gate blocked repaired code | No source/environment identity | Versioned fingerprints explain reuse/invalidation/retry and preserve checkpoints |
| Evaluation/profile could appear complete after failure | Failure evidence was treated as success/file presence | Selected evaluation fails nonzero; batch-one profile must complete |
| Hosted Colab claim was false | OpenMMLab pins conflicted with the kernel | Per-family isolated pinned environments |
| Publisher modified training checkout and assumed result branch | In-place switch; latest manifest not staged | Disposable clone, missing/existing branch handling, exact two-path allowlist |

CPU tests use synthetic data and local bare Git repositories. They do not run
full training, download VisDrone, compile VMamba CUDA code, or claim GPU
verification.
