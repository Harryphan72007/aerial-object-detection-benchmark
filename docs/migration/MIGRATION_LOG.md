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
- Commit SHA: `1a850dec57cb8e4e3ff9e8974f59c5c787dcdf38`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/8
- Remaining risks: legacy binary checkpoint internals are documented but not
  executable without model-family runtimes

## PR 2 — Inventory dependencies and path assumptions

- Status: completed
- Dependencies: PR 1
- Files changed: runtime inventory; Colab path inventory; frozen shared-kernel
  requirements; read-only environment reporter; diagnostics tests; regenerated
  PR 1 source inventory
- Conceptual change: recorded notebook imports, mutually isolated model-family
  environments, upstream revisions, implicit clone/Drive paths, and local/Colab
  divergences without installing packages or constructing models
- Preserved behavior: all setup cells, requirement entry points, family-specific
  binary environments, Drive paths, and model selection behavior
- Tests executed: focused PR 1–2 tests; full CPU suite; Ruff; Python compile;
  notebook validator; secret scanner; before/after worktree check
- Observed result: 113 passed, 2 skipped because PyTorch was unavailable; the
  diagnostic left the worktree unchanged and reported 17 notebooks
- Validation level: CPU/static (below G0; no Colab session was executed)
- Unverified: live Colab package availability, Drive mount/write reliability,
  CUDA/GPU compatibility, and model construction
- Compatibility effect: additive documentation, requirements inventory, and
  read-only diagnostics only
- Deviation: actual notebooks use several inconsistent local fallbacks; PR 2
  documents them and defers centralization to PR 6
- Rollback: remove the PR 2 docs, requirements file, diagnostic, and test, then
  regenerate the PR 1 inventory snapshot
- Commit SHA: `17af755f408146f243e58242451818333c2ee439`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/9
- Remaining risks: hosted Colab images can drift; import metadata is not a model
  compatibility gate

## PR 3 — Repository hygiene

- Status: completed
- Dependencies: PR 1–2
- Files changed: ignore and attribute policy; security and contribution policy;
  GitHub/Drive boundary; prohibited-file validator and tests; migration log;
  regenerated source inventory
- Conceptual change: enforce the source/artifact boundary for tracked or staged
  files while allowing small reviewed fixtures and validated lightweight results
- Preserved behavior: existing source package `src.data`, notebooks, result
  publication layout, and all Drive artifact paths
- Tests executed: prohibited-file unit/CLI checks; full CPU suite; Ruff; Python
  compile; notebook validator; secret scanner; inventory drift check
- Observed result: 117 passed, 2 skipped because PyTorch was unavailable; all
  focused, hygiene, notebook, inventory, and security checks passed
- Validation level: CPU/static
- Unverified: GitHub branch protection and Drive-side enforcement
- Compatibility effect: additive validation; existing tracked repository passes
- Deviation: existing hygiene rules were consolidated rather than replaced with
  a second competing policy
- Rollback: revert PR 3; restore the previous ignore/attribute and contributor
  files; no runtime artifacts are deleted
- Commit SHA: `cfddf38fc839676bc1bff49d9fa9338d02da380c`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/10
- Remaining risks: pattern and size checks cannot detect every sensitive or
  copyrighted artifact, so review remains mandatory

## PR 4 — Read-only Git and environment diagnostics

- Status: completed
- Dependencies: PR 1–3
- Files changed: reusable repository/environment diagnostic modules; canonical
  runner; clean/dirty/detached/non-Git/CPU/no-GPU tests; migration log;
  regenerated source inventory
- Conceptual change: expose structured read-only diagnostics without blocking
  runs or importing detector frameworks
- Preserved behavior: PR 2 diagnostic entry point, all notebooks, training, and
  artifact paths
- Tests executed: focused diagnostic cases; full CPU suite; Ruff; Python
  compile; notebook, prohibited-file, secret, and inventory checks
- Observed result: 121 passed, 2 skipped because PyTorch was unavailable;
  focused diagnostics covered every required repository and CPU/no-GPU state
- Validation level: CPU/static; no G0 Colab execution
- Unverified: live CUDA device discovery and hosted Colab hardware reporting
- Compatibility effect: additive package APIs and CLI only
- Deviation: repository uses flat `src` package, so modules live under
  `src/diagnostics` rather than `src/visdrone_benchmark/diagnostics`
- Rollback: revert PR 4 and regenerate the PR 1 inventory; PR 2 reporter remains
- Commit SHA: SELF
- PR URL: pending
- Remaining risks: OS access and `nvidia-smi` metadata do not prove framework
  compatibility
