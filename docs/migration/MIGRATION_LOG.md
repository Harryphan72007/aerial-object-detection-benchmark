# VisDrone2019 migration log

This append-only log tracks the numbered migration sequence. GitHub and GPU
validation are recorded only when actually executed. A `SELF` commit value means
the commit containing that entry; the next migration entry replaces it with the
resolved SHA.

## PR 1 — Freeze legacy artifact contracts

- Status: completed
- Dependencies: none
- Files changed: compatibility contract; legacy JSON schemas; notebook/source
  inventory and generator; representative JSON/CSV fixtures; compatibility tests
- Conceptual change: froze current artifact names, paths, fields, metric keys,
  registry layout, and notebook consumers without changing runtime behavior
- Preserved behavior: all notebooks, checkpoint aliases, evaluator inputs,
  metric names, class IDs, Drive paths, and publishing layouts
- Tests executed: focused artifact tests; full CPU suite; Ruff; Python compile;
  notebook validator; secret scanner; inventory drift check
- Observed result: 109 passed, 2 skipped because PyTorch was unavailable; all
  focused/static checks passed
- Validation level: CPU/static (below G0; no Colab session was executed)
- Unverified: Colab, Drive, CUDA, GPU, training, resume, latency, and mAP
- Compatibility effect: additive schemas, fixtures, inventory, and documentation
- Deviation: the planned notebooks 01–06 do not exist; all 17 actual notebooks
  and delegated Python producers/consumers were inventoried instead
- Rollback: remove only PR 1 docs, schemas, fixture, inventory tool, and tests;
  preserve all Drive artifacts
- Commit SHA: SELF
- PR URL: pending
- Remaining risks: legacy binary checkpoint internals are documented but not
  executable without model-family runtimes
