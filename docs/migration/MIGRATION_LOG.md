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
- Commit SHA: `726ed529bb1fe18239aa9127bd0dae63ef3f7926`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/11
- Remaining risks: OS access and `nvidia-smi` metadata do not prove framework
  compatibility

## PR 5 — Canonical Colab bootstrap

- Status: completed with live G0 pending
- Dependencies: PR 1–4
- Files changed: canonical bootstrap notebook; safe branch/tag/commit checkout
  helper; first/later-session docs; notebook and local-Git tests; migration log;
  regenerated notebook/source inventory
- Conceptual change: provide one GitHub-openable entry point for a disposable
  Colab clone, Drive mount, shared install, exact ref selection, and diagnostics
- Preserved behavior: all existing canonical notebooks and Drive layout; no model
  training is started by bootstrap
- Tests executed: notebook JSON/output/syntax checks; local fresh clone and
  branch/tag/commit selection; dirty-tree refusal; full CPU/static suite;
  hygiene, secret, and inventory checks
- Observed result: 123 passed, 2 skipped because PyTorch was unavailable;
  local Git fixtures verified fresh clone, branch/tag/commit selection, and
  dirty-tree refusal; notebook/static checks passed
- Validation level: CPU/static; G0 pending because Colab was not available
- Unverified: live GitHub-to-Colab open, Google Drive mount/write, hosted pip
  install, and Colab clean-tree completion
- Compatibility effect: additive notebook and backward-compatible helper API
- Deviation: actual repository already had `src.colab_setup`; it was extended
  instead of introducing duplicate bootstrap logic in a new package
- Rollback: revert PR 5; existing notebooks and setup cells remain usable
- Commit SHA: `9befcc93e2b4bbc46c65ec88481b9b988d17a2a0`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/12
- Remaining risks: a stale pre-PR-5 clone lacks the new helper and should be
  recloned; hosted dependency availability can drift

## PR 6 — Central path resolver in legacy-compatible mode

- Status: completed
- Dependencies: PR 1–5
- Files changed: versioned path layout config; new pathing package; one minimal
  canonical notebook path-cell adapter; path/evaluator tests; documentation;
  migration log and regenerated inventory
- Conceptual change: preserve every legacy path while exposing isolated
  smoke/controlled/performance/full/sliced/ensemble namespaces for later PRs
- Preserved behavior: `src.paths.ProjectPaths`, all existing Drive paths,
  registry discovery, checkpoint names, and evaluator behavior
- Tests executed: old/new path equivalence; namespace isolation and invalid
  combinations; legacy evaluator fixture; full CPU/static suite; notebook,
  hygiene, secret, Ruff, compile, and inventory checks
- Observed result: 127 passed, 2 skipped because PyTorch was unavailable; all
  legacy path equivalence, isolation, evaluator, notebook, and static checks passed
- Validation level: CPU/static
- Unverified: existing Drive artifact lookup in a live Colab session
- Compatibility effect: additive resolver; legacy notebook resolves the same type
  and values
- Deviation: `src.paths` is already a module, so the new package is `src.pathing`
  to avoid a destructive module-to-package rename
- Rollback: revert PR 6; the notebook import returns to `src.paths.ProjectPaths`;
  new namespaces were unused and require no Drive cleanup
- Commit SHA: `9256d1477f945a38e7938a3448c7c7936993a919`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/13
- Remaining risks: later producers must opt into the new resolver consistently
  before isolation is enforceable

## PR 7 — Config schemas and deterministic hashes

- Status: completed
- Dependencies: PR 1–6
- Files changed: portable experiment-config schema; strict dependency-free runtime
  validator and resolver; per-model legacy/smoke configs; equivalence/hash tests;
  migration log and regenerated source inventory
- Conceptual change: express existing per-model constants as versioned configs and
  provide semantic SHA-256 identities independent of YAML formatting
- Preserved behavior: existing model-track YAML files, notebook orchestration,
  training defaults, and runtime config loading remain unchanged
- Tests executed: all checked-in config validation; legacy-value equivalence;
  deterministic hash and invalid-field cases; full CPU/static suite; hygiene,
  secret, Ruff, compile, notebook, and inventory checks
- Observed result: 140 passed, 2 skipped because PyTorch was unavailable; all
  config equivalence, hash, rejection, notebook, and static checks passed
- Validation level: CPU/static
- Unverified: live framework consumption of the new configs, which is deliberately
  deferred until producer migrations
- Compatibility effect: additive schema/config API only
- Deviation: repository uses a flat `src` package, so config code lives under
  `src/config`; smoke configs reduce runtime while preserving model identity and
  optimization constants
- Rollback: revert PR 7; existing config consumers continue using their original
  files and literal workflow values
- Commit SHA: `e394b68cee9cbdd45a5a0b00b6786ae8b9558d6e`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/14
- Remaining risks: producer code must opt into validation before the schema can
  prevent invalid live runs

## PR 8 — Experiment manifests

- Status: completed
- Dependencies: PR 1–7
- Files changed: portable v1 experiment-manifest schema; strict manifest
  lifecycle package; completed/failed examples; success/failure/compatibility
  tests; migration log and regenerated source inventory
- Conceptual change: record code, config, dataset, environment, hardware, seed,
  output namespace, lifecycle status, results, and failures in one atomic record
- Preserved behavior: legacy `run_manifest.json`, registry, evaluator globbing,
  training loop, and checkpoint discovery are unchanged
- Tests executed: create/finalize/reload round trips for success and failure;
  lifecycle rejection cases; example validation; legacy evaluator filename
  isolation; full CPU/static suite and repository checks
- Observed result: 146 passed, 2 skipped because PyTorch was unavailable; both
  terminal round trips, lifecycle guards, examples, compatibility, and static
  checks passed
- Validation level: CPU/static
- Unverified: integration into GPU-backed producer lifecycles, deferred until
  each training loop is migrated
- Compatibility effect: additive `experiment_manifest.json` contract that legacy
  readers ignore
- Deviation: repository uses flat `src`, so implementation lives in
  `src/manifests`
- Rollback: revert PR 8; any generated experiment manifests are harmless metadata
  and may be retained or moved to quarantine
- Commit SHA: `71ff516665517fb224981dbfef9643f69e730941`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/15
- Remaining risks: until producer integration, callers must explicitly finalize
  a running manifest after interruption handling

## PR 9 — Namespace guards

- Status: completed
- Dependencies: PR 1–8
- Files changed: pre-write namespace guard API; public pathing exports;
  namespace policy documentation; collision matrix tests; migration log and
  regenerated source inventory
- Conceptual change: reject cross-track, cross-mode, and cross-artifact writes by
  comparing proposed destinations with the exact resolved run namespace
- Preserved behavior: no directories are created, and legacy producers remain
  unchanged until their model-specific migrations
- Tests executed: smoke/full, controlled/performance, full/sliced, artifact-kind,
  duplicate-root, descendant, and read-only creation cases; full CPU/static suite
  and repository checks
- Observed result: 153 passed, 2 skipped because PyTorch was unavailable; every
  invalid namespace pairing was rejected before directory creation and all
  static checks passed
- Validation level: CPU/static
- Unverified: enforcement inside not-yet-migrated GPU producers
- Compatibility effect: additive guards for new namespaced producers
- Deviation: implementation lives in the existing `src/pathing` package
- Rollback: revert PR 9; no test directories were created, so no cleanup is
  required
- Commit SHA: `664a20df8fa11909f304ff90551cfd2919b47e93`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/16
- Remaining risks: legacy writers remain outside guard coverage until explicitly
  migrated

## PR 10 — Legacy evaluator compatibility suite

- Status: completed
- Dependencies: PR 1–9
- Files changed: two-model controlled comparison fixture plus smoke candidate;
  frozen evaluator/comparison tests; evaluator contract documentation; migration
  log and regenerated source inventory
- Conceptual change: none; freeze consumer behavior before producer migrations
- Preserved behavior: detailed metrics, controlled-config filtering, legacy file
  discovery, output columns, missing-model rows, and rejected-run reporting
- Tests executed: exact detailed evaluator fixture; deterministic two-model
  comparison twice; smoke exclusion; full CPU/static suite and repository checks
- Observed result: 155 passed, 2 skipped because PyTorch was unavailable; exact
  evaluator metrics, deterministic comparison rows, smoke exclusion, and all
  static checks passed
- Validation level: CPU/static
- Unverified: pycocotools/GPU evaluator parity on the full validation dataset
- Compatibility effect: contract tests only; no production evaluator changes
- Deviation: reused the PR 1 COCO fixture and added a comparison-case bundle to
  avoid committing generated images or checkpoints
- Rollback: revert PR 10; evaluator and notebooks require no restoration because
  runtime code was unchanged
- Commit SHA: `55975f7059e30bf6a6447a05146e80582504a672`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/17
- Remaining risks: synthetic fixtures cannot reveal scale- or ordering-dependent
  behavior in a full VisDrone evaluation

## PR 11 — Shared dataset and class mapping

- Status: completed
- Dependencies: PR 1–10
- Files changed: canonical class mapping, COCO dataset, and dataset manifest
  modules; legacy import adapters; tiny COCO fixture; notebook 00, evaluator, and
  RT-DETR trainer imports; parity and one-batch tests; migration log/inventory
- Conceptual change: one implementation now owns official IDs, two-class collapse,
  ignored-region IDs, stable record loading, collating, and dataset summaries
- Preserved behavior: old `collapse_classes` and `dataloaders` imports remain
  valid; converted annotations and RT-DETR zero-based processor labels are unchanged
- Tests executed: one two-image batch; class IDs, boxes, areas, image IDs, ignored
  counts, deterministic manifests, legacy/new parity; full CPU/static suite and
  repository checks
- Observed result: 158 passed, 2 skipped because PyTorch was unavailable; shared
  mapping, manifest, one-batch, legacy/new parity, notebook, trainer syntax, and
  all static checks passed
- Validation level: CPU/static; one-batch loader without PyTorch
- Unverified: live RT-DETR processor/DataLoader execution because PyTorch and
  Transformers model assets are unavailable
- Compatibility effect: compatibility re-exports preserve old imports while new
  consumers use canonical modules
- Deviation: the repository's trainer is an external-framework launcher; notebook
  00, the evaluator consumer, and the native RT-DETR trainer were minimally migrated
- Rollback: revert PR 11; source datasets are never modified, and legacy modules
  return to their prior implementations
- Commit SHA: `ab45cfbe47e28ad8cd880c5e51e0ca7f46d21fd1`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/18
- Remaining risks: full-dataset file corruption and framework worker behavior
  require later G1/G2 and Colab validation

## PR 12 — Versioned artifact writers with legacy dual-write

- Status: completed
- Dependencies: PR 1–11
- Files changed: strict checkpoint/metric/prediction schemas; artifact identity,
  readers, and writers; legacy view adapters; synthetic dual-write compatibility
  tests; migration log and regenerated source inventory
- Conceptual change: new producers can write schema-v1 envelopes and atomically
  materialize the frozen flat metrics, COCO prediction arrays, and checkpoint names
- Preserved behavior: `best_map.pth`, `best_aptiny.pth`, `last.pth`, legacy metric
  keys, and COCO prediction records remain byte/value compatible
- Tests executed: new/legacy metric equality; old/new evaluator prediction parity;
  checkpoint byte/hash/alias equality; schema strictness; full CPU/static suite and
  repository checks
- Observed result: 162 passed, 2 skipped because PyTorch was unavailable;
  new/legacy metrics, predictions, evaluator results, checkpoint bytes/hashes,
  schemas, and all static checks passed
- Validation level: CPU/static with synthetic checkpoint bytes
- Unverified: framework-native tensor checkpoint serialization on GPU
- Compatibility effect: additive writers/readers; no existing producer is switched
  until its model migration
- Deviation: prediction envelopes were included with the requested checkpoint and
  metric schemas because evaluator parity requires the frozen COCO view
- Rollback: revert PR 12; direct legacy writers remain active, and test artifacts
  are temporary only
- Commit SHA: `bfcbebee3ef9430238c7757ff0d2f673fb0b39f3`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/19
- Remaining risks: native framework state dictionaries require model-specific
  compatibility tests in later PRs

## PR 13A — ResNet model factory

- Status: completed with live G1 pending
- Dependencies: PR 1–12
- Files changed: ResNet Faster R-CNN factory package; dependency-injected factory
  contract tests; minimal notebook 01 availability hook; migration log/inventory
- Conceptual change: centralize pinned config resolution and public MMDetection
  construction/forward boundaries without changing training or optimization
- Preserved behavior: model ID, pinned config, public `init_detector` and
  `inference_detector` APIs, checkpoint argument, device, parameter counting, and
  notebook orchestration
- Tests executed: fake-backend legacy/new argument, parameter-count, and output
  parity; config resolution and missing-runtime failure; full CPU/static suite and
  repository checks
- Observed result: 166 passed, 2 skipped because PyTorch was unavailable;
  injected legacy/new construction arguments, parameter counts, forward outputs,
  notebook/static checks, and expected missing-runtime failure all passed
- Validation level: CPU/static; dependency-injected construction and forward only
- Unverified: real MMDetection construction, parameter count, weights, CUDA, and
  tensor forward because PyTorch/MMDetection are unavailable
- Compatibility effect: additive factory; legacy adapter and trainer remain active
- Deviation: the actual repository has one generic model-day notebook rather than
  a ResNet-specific notebook 01, so it receives only a non-constructing hook
- Rollback: revert PR 13A; notebook orchestration continues through the existing
  adapter/launcher
- Commit SHA: SELF
- PR URL: pending
- Remaining risks: G1 must confirm exact real parameter count and detector output
  structure at the pinned MMDetection revision
