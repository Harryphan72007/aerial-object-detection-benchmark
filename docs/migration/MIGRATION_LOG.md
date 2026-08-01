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
- Commit SHA: `f2053ab0957ce09953328aeaccbb0bc61d87afaf`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/20
- Remaining risks: G1 must confirm exact real parameter count and detector output
  structure at the pinned MMDetection revision

## PR 13B — Swin factory and feature adapter

- Status: completed with live G1 pending
- Dependencies: PR 1–13A
- Files changed: dynamic-resolution Swin Faster R-CNN factory; NHWC-to-NCHW FPN
  feature adapter; smoke/controlled contract tests; generic notebook hook;
  migration log and inventory
- Conceptual change: centralize construction-time image size, mask-branch removal,
  class count, and validation of every FPN feature level
- Preserved behavior: pinned MMDetection config and public builder, Swin channel
  widths, two-class head, legacy launcher, and notebook orchestration
- Tests executed: 128/640 config and four-level feature contracts; NHWC conversion;
  NCHW pass-through; invalid channel rejection; injected public builder; full
  CPU/static suite and repository checks
- Observed result: 170 passed, 2 skipped because PyTorch was unavailable; dynamic
  smoke/controlled sizes, every FPN level, layout conversion, config mutation,
  injected builder, notebook, and static checks passed
- Validation level: CPU/static with tensor-shaped test doubles
- Unverified: real Swin tensors, parameter count, MMDetection construction, CUDA,
  and forward at smoke/controlled sizes
- Compatibility effect: additive factory/adapter; existing trainer remains active
- Deviation: the actual notebook 02 is result publishing, so the generic model-day
  notebook receives the non-constructing Swin hook
- Rollback: revert PR 13B and continue through the existing MMDetection launcher
- Commit SHA: `ea22681cf3a6320df1b2d3ec5109bc222b2796fb`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/21
- Remaining risks: upstream Swin outputs must be checked against the adapter on a
  pinned real runtime before enabling it in training

## PR 13C — VMamba factory and importer

- Status: completed with live G1 pending
- Dependencies: PR 1–13B
- Files changed: clean VMamba detection importer; revision and selective-scan
  gates; VMamba factory/build report; NCHW stage validator; clean-process and
  construction contract tests; generic notebook hook; migration log/inventory
- Conceptual change: centralize fragile module registration and refuse construction
  unless revision, optimized scan, config, and pretrained-weight gates pass
- Preserved behavior: pinned VMamba revision, `model` registration name, official
  tiny detection config, Faster R-CNN mask removal, two-class head, and required
  pretrained checkpoint
- Tests executed: clean Python process import; revision mismatch; optimized backend
  selection; NCHW feature stages; injected build/report; full CPU/static suite and
  repository checks
- Observed result: 175 passed, 2 skipped because PyTorch was unavailable; clean
  process registration, revision/backend/pretraining gates, injected construction,
  feature validation, notebook, and static checks passed
- Validation level: CPU/static with synthetic upstream tree and tensor shapes
- Unverified: compiled selective-scan import/execution, real model construction,
  parameter count, CUDA, and tensor forward
- Compatibility effect: additive factory/importer; legacy launcher remains active
- Deviation: the actual notebook 03 is environment setup, so the generic model-day
  notebook receives the non-constructing VMamba hook
- Rollback: revert PR 13C and continue through the existing registered-import
  launcher path
- Commit SHA: `121523246f6ebec27be70af8ece030d2d8451486`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/22
- Remaining risks: G1 must prove the pinned extension is the implementation used by
  real SS2D blocks rather than merely importable

## PR 13D — RT-DETR legacy model factory

- Status: completed with live G1 pending
- Dependencies: PR 1–13C
- Files changed: RT-DETRv2 construction/result factory; adapter delegation; legacy
  evaluation-weight and forward contract tests; generic notebook hook; migration
  log and inventory
- Conceptual change: extract construction, pinned processor/model provenance, strict
  legacy weight loading, device selection, and inference context from the adapter
- Preserved behavior: base model/revision, input size, label metadata, strict state
  loading, device/eval transition, adapter API, optimizer, and scheduler
- Tests executed: injected pretrained construction; one synthetic legacy checkpoint;
  strict incompatible-key failure; one injected forward; source guard against
  optimizer/scheduler logic; full CPU/static suite and repository checks
- Observed result: 179 passed, 2 skipped because PyTorch was unavailable; pinned
  construction calls, strict legacy-weight loading, incompatibility rejection,
  injected forward, adapter imports, notebook, and static checks passed
- Validation level: CPU/static with model/processor test doubles
- Unverified: real Transformers/PyTorch construction, legacy tensor checkpoint,
  parameter count, CUDA, and forward output parity
- Compatibility effect: existing RT-DETR adapter now delegates construction while
  keeping its public loading/prediction contract
- Deviation: the actual notebook 04 does not exist, so the generic model-day
  notebook receives the non-constructing RT-DETR hook
- Rollback: revert PR 13D to restore construction inside `RTDetrV2Adapter`; optimizer
  and scheduler need no rollback because they were untouched
- Commit SHA: `4afb278c660a4bab166bd996424c5cb4ff54d43f`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/23
- Remaining risks: G1 must confirm the legacy checkpoint's serialized tensor keys
  against the pinned Transformers implementation

## PR 14 — Shared training state and engine

- Status: completed with live G2 pending
- Dependencies: PR 1–13D
- Files changed: shared training state, accumulation engine, atomic state checkpoint,
  ResNet engine boundary, CPU resume/evaluator integration tests, migration log and
  inventory
- Conceptual change: centralize epoch/microstep/optimizer-step transitions and
  serializable component restore through a ResNet-first wrapper
- Preserved behavior: legacy MMDetection launcher remains available; legacy
  prediction view and evaluator remain readable
- Tests executed: accumulated optimizer/scheduler steps; epoch-one save and
  epoch-two resume; component mismatch rejection; legacy evaluator read; full
  CPU/static suite and repository checks
- Observed result: 181 passed, 2 skipped because PyTorch was unavailable;
  accumulation, optimizer/scheduler ordering, save/resume, mismatch rejection,
  legacy evaluator, and static checks passed
- Validation level: CPU/static integration with injected components
- Unverified: real ResNet forward/backward, scaler, tensor optimizer state, CUDA,
  and framework-native checkpointing
- Compatibility effect: additive engine; model-specific runtime switch awaits G2
- Deviation: MMDetection owns its loop internally, so the ResNet migration boundary
  is implemented and tested but the live launcher is retained until G2
- Rollback: revert PR 14 and continue using the MMDetection runner; temporary smoke
  state files are test-only
- Commit SHA: `4a46f623312fbde4be8ed44792b0e1f6633fd79c`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/24
- Remaining risks: framework callbacks must preserve the tested step ordering when
  wired into a real MMEngine runner

## PR 15 — Checkpoint schema and loading modes

- Status: completed
- Dependencies: PR 1–14
- Files changed: strict checkpoint v2 schema; loading-mode classifier/guard;
  compatibility policy; mode tests; migration log and inventory
- Conceptual change: classify checkpoint intent before tensor loading as full resume,
  weights only, evaluation only, or incompatible
- Preserved behavior: legacy checkpoints remain evaluation inputs and are never
  rewritten; current checkpoint aliases remain unchanged
- Tests executed: exact resume, config/accumulation drift, legacy metadata absence,
  model mismatch, schema parity; full CPU/static suite and repository checks
- Observed result: 186 passed, 2 skipped because PyTorch was unavailable; all
  loading classifications, refusal guards, schema parity, and static checks passed
- Validation level: CPU/static metadata classification
- Unverified: real tensor deserialization for each framework
- Compatibility effect: additive policy; legacy artifacts default safely to
  evaluation-only
- Deviation: incompatible is an explicit fourth refusal state rather than forcing
  unsafe artifacts into one of the three allowed loading modes
- Rollback: revert PR 15; retain v2 metadata as harmless files or quarantine it
- Commit SHA: `15982ebba6f787e9d0d4f0bf5e4577e826807f32`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/25
- Remaining risks: model-signature generation must be wired consistently by each
  migrated producer

## PR 16 — RT-DETR parameter-group discovery

- Status: completed with live model report pending
- Dependencies: PR 1–15
- Files changed: deterministic RT-DETR group discovery/report; smoke-manifest
  attachment helper; diagnostic docs/notebook line; tests; migration log/inventory
- Conceptual change: audit backbone/detector ownership without enabling differential LR
- Preserved behavior: global-LR optimizer remains unchanged
- Tests executed: zero duplicate/unassigned, exact tensor/value count sums, frozen
  parameter exclusion, report round trip and manifest attachment; full CPU/static suite
- Observed result: 189 passed, 2 skipped because PyTorch was unavailable; all
  parameter ownership, count, report, and static checks passed
- Validation level: CPU/static parameter test doubles
- Unverified: reviewed report from the pinned real RT-DETRv2-L model
- Compatibility effect: additive diagnostics only
- Deviation: report attachment is exposed for the smoke producer; live manifest
  generation awaits the GPU smoke run
- Rollback: revert PR 16 and keep the global-LR optimizer
- Commit SHA: `805d70c66880ef3d44625303fd80ab363a19a7b6`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/26
- Remaining risks: upstream parameter names may require an explicitly reviewed marker
  update before PR 17 enables group-specific rates

## PR 17 — RT-DETR optimizer and scheduler repair

- Status: completed with GPU smoke pending
- Dependencies: PR 1–16
- Files changed: recipe-v2 optimizer and scheduler modules; smoke/performance
  policies; HPO/final workflow wiring; RT-DETR trainer/notebook; tests/docs/log
- Conceptual change: enable reviewed differential LR, stronger clipping, and
  update-based warm-up/cosine decay behind `rtdetr_recipe_v2`
- Preserved behavior: legacy global-LR construction remains selectable
- Tests executed: exact group LRs; exact search/full early trace; scheduler step and
  state round trip; HPO/final/notebook contracts; full CPU/static suite
- Observed result: 192 passed, 2 skipped because PyTorch was unavailable; exact
  differential rates and shared-horizon trace contracts passed
- Validation level: CPU/static optimizer and scheduler test doubles
- Unverified: one real optimizer step and short CUDA smoke
- Compatibility effect: additive versioned recipe; old checkpoints remain loadable
  under PR 15 policies
- Deviation: the performance horizon is 100 epochs and remains fixed during short
  HPO trials, rather than truncating cosine decay to the trial length
- Rollback: revert PR 17 and select the legacy global-LR recipe
- Commit SHA: `df534e46bf28f35bae8a0196c1fdcc6150907769`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/27
- Remaining risks: upstream optimizer construction must preserve custom group metadata

## PR 18 — EMA and effective batch

- Status: completed with live tensor smoke pending
- Dependencies: PR 1–17
- Files changed: framework-light EMA and accumulation modules; shared engine callback;
  RT-DETR checkpoint/trainer integration; performance policy; tests/docs/log
- Conceptual change: optional EMA state and explicit effective-batch/update accounting
- Preserved behavior: raw model state remains canonical and checkpoints without EMA load
- Tests executed: optimizer-step count under partial accumulation; EMA update count;
  save/load and raw-state restoration; effective-batch validation; full CPU/static suite
- Observed result: 195 passed, 2 skipped because PyTorch was unavailable; EMA,
  accumulation, effective-batch, checkpoint, and shared-engine contracts passed
- Validation level: CPU/static state test doubles
- Unverified: tensor EMA round trip and accumulated CUDA update
- Compatibility effect: additive optional `ema_state_dict` checkpoint field
- Deviation: raw weights remain the default evaluation target; EMA evaluation is an
  explicit separate context to prevent silent metric mixing
- Rollback: revert PR 18, disable EMA/accumulation, and load raw model state only
- Commit SHA: `cd260f14d4d3bde9149744c4983c27207d6052c9`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/28
- Remaining risks: GradScaler overflow detection should be confirmed in the CUDA smoke

## PR 19 — Persistent RT-DETR Optuna search v2

- Status: completed with real training trials pending
- Dependencies: PR 1–18
- Files changed: RT-DETR v2 objective/storage policy; study metadata schema;
  five-parameter search config; final-workflow selection; notebook; tests/docs/log
- Conceptual change: move RT-DETR search to isolated `rtdetr_optuna_v2` persistence
- Preserved behavior: the legacy study path and database are never modified
- Tests executed: one trial plus reconnect/second trial; independent v2 namespace;
  valid SQLite snapshot; failure classification; metadata/search schema; full suite
- Observed result: 200 passed, 2 skipped because PyTorch was unavailable; Optuna
  reconnect, SQLite snapshot, isolation, schema, and failure-policy checks passed
- Validation level: CPU Optuna/SQLite persistence with synthetic objectives
- Unverified: real RT-DETR objective values and Drive reconnect latency
- Compatibility effect: final workflow prefers v2 output and retains legacy fallback
- Deviation: SQLite backup is atomic at the file target and replaces a rolling latest
  snapshot rather than retaining an unbounded snapshot per trial
- Rollback: revert PR 19 and quarantine the v2 DB; the legacy DB remains unchanged
- Commit SHA: `b906488586af7292ca392cd0e3b8ed24fc4ce109`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/29
- Remaining risks: Drive-mounted SQLite locking still requires the planned Colab trial

## PR 20 — Best checkpoint and early stopping

- Status: completed with live training resume pending
- Dependencies: PR 1–19
- Files changed: checkpoint-selection and early-stopping modules; metric schema v2;
  RT-DETR trainer/performance config; tests/docs/log
- Conceptual change: separate resumable last, best raw, optional best EMA, and alias
- Preserved behavior: `best_map.pth` and `best.pt` resolve to the raw best checkpoint
- Tests executed: synthetic early peak; patience save/resume; raw/EMA independence;
  legacy alias resolution; last preservation; full CPU/static suite
- Observed result: 203 passed, 2 skipped because PyTorch was unavailable; early
  peak, persisted patience, raw/EMA selection, and legacy alias checks passed
- Validation level: CPU filesystem/state tests
- Unverified: interrupt/resume during a real RT-DETR run
- Compatibility effect: additive checkpoint names/state and metric schema v2
- Deviation: early stopping keys off raw mAP until EMA evaluation is explicitly enabled
- Rollback: revert PR 20 and recreate `best.pt` from the last known-good raw checkpoint
- Commit SHA: `a76ad959957c3b21ea1a2e27a40b4b704586c16f`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/30
- Remaining risks: older checkpoint producers continue using legacy best-map names until PR 22

## PR 21 — Controlled/performance config split

- Status: completed
- Dependencies: PR 1–20
- Files changed: controlled/performance configs and validator; comparison guard;
  canonical notebook selectors; tests/protocol docs/log
- Conceptual change: formalize disjoint option sets, output roots, and summary tables
- Preserved behavior: untagged legacy artifacts remain controlled-compatible
- Tests executed: performance-only rejection; root/table separation; explicit cross-track
  rejection; notebook parameter/format checks; full CPU/static suite
- Observed result: 206 passed, 2 skipped because PyTorch was unavailable; track
  option, namespace, comparison, legacy fallback, and notebook contracts passed
- Validation level: CPU/static config, artifact identity, and notebook contracts
- Unverified: live performance-run output production in Colab
- Compatibility effect: additive track tags; controlled comparison explicitly guards them
- Deviation: actual canonical notebooks 01–03 and 30–31 receive selectors because the
  plan's notebook 01–06 numbering does not exist in this repository
- Rollback: revert PR 21 and exclude tagged performance runs from legacy summaries
- Commit SHA: `227a3a154aa24630a9b820d68dab228ecddf2a55`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/31
- Remaining risks: performance producers added in PRs 23–25 must use the new root helper

## PR 22A — Migrate Swin training onto the shared engine

- Status: completed with live model smoke pending
- Dependencies: PR 1–21
- Files changed: Swin shared trainer boundary; fine-tuning notebook; smoke/resume/
  compatibility test; migration log/inventory
- Conceptual change: route Swin lifecycle callbacks through the shared engine contract
- Preserved behavior: existing Swin model factory/backend and legacy evaluator artifacts
- Tests executed: one batch/forward/backward/optimizer/scheduler step; save/resume;
  legacy evaluator read; full CPU/static suite
- Observed result: 207 passed, 2 skipped because PyTorch was unavailable; Swin
  batch, step, resume, and legacy evaluation contracts passed
- Validation level: CPU/static callbacks and legacy fixture
- Unverified: real Swin tensor forward and CUDA optimizer step
- Compatibility effect: additive typed trainer boundary
- Deviation: framework callbacks stay dependency-injected until the model runtime is present
- Rollback: revert PR 22A and restore the existing Swin backend entry point
- Commit SHA: `185da749875cd68ca74a65b76dca250be32a2e5f`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/32
- Remaining risks: MMDetection hook ordering requires live smoke confirmation

## PR 22B — Migrate VMamba training onto the shared engine

- Status: completed with live model smoke pending
- Dependencies: PR 1–22A
- Files changed: VMamba shared trainer boundary; fine-tuning notebook; smoke/resume/
  compatibility test; migration log/inventory
- Conceptual change: route VMamba lifecycle callbacks through the shared engine contract
- Preserved behavior: existing VMamba importer/factory/backend and legacy artifacts
- Tests executed: one accumulated batch pair/forward/backward/optimizer/scheduler step;
  save/resume; legacy evaluator read; full CPU/static suite
- Observed result: 208 passed, 2 skipped because PyTorch was unavailable; VMamba
  accumulation, step, resume, and legacy evaluation contracts passed
- Validation level: CPU/static callbacks and legacy fixture
- Unverified: optimized selective-scan tensor forward and CUDA optimizer step
- Compatibility effect: additive typed trainer boundary
- Deviation: framework callbacks stay dependency-injected until the VMamba runtime is present
- Rollback: revert PR 22B and restore the existing VMamba backend entry point
- Commit SHA: `b1dcf60a9d11af2801707d4f9718fbc6c78b80b6`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/33
- Remaining risks: upstream selective-scan autograd must be confirmed live

## PR 22C — Migrate RT-DETR training onto the shared engine

- Status: completed with live model smoke pending
- Dependencies: PR 1–22B
- Files changed: RT-DETR shared trainer boundary; fine-tuning notebook; smoke/resume/
  EMA-hook/compatibility test; migration log/inventory
- Conceptual change: route RT-DETR lifecycle callbacks through the shared engine contract
- Preserved behavior: recipe-v2 optimizer/scheduler/EMA and legacy evaluator artifacts
- Tests executed: accumulated forward/backward; one optimizer/scheduler/EMA update;
  save/resume; legacy evaluator read; full CPU/static suite
- Observed result: 209 passed, 2 skipped because PyTorch was unavailable; RT-DETR
  accumulation, optimizer/scheduler/EMA, resume, and legacy evaluation contracts passed
- Validation level: CPU/static callbacks and legacy fixture
- Unverified: real RT-DETR tensor forward and CUDA optimizer step
- Compatibility effect: additive typed trainer boundary
- Deviation: the optimized script retains tensor-specific timing/evaluation callbacks while
  sharing lifecycle state and update-boundary semantics
- Rollback: revert PR 22C and restore the previous RT-DETR script entry point
- Commit SHA: `a53a5a6dac1b7b7dde9d2f87b94af83d5b3fa504`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/34
- Remaining risks: GradScaler skipped-update behavior requires live confirmation

## PR 23 — Higher resolution and tiled training data

- Status: completed with image materialization smoke pending
- Dependencies: PR 1–22C
- Files changed: deterministic tiling/clipping module; tile manifest schema; performance
  config; boundary fixture/tests; notebook 07; migration log/inventory
- Conceptual change: derive versioned tiled COCO data without mutating source annotations
- Preserved behavior: source image and annotation identifiers remain recorded in manifests
- Tests executed: boundary clipping, visible fraction, ignore regions, empty tiles,
  deterministic IDs/hashes, source immutability; full CPU/static suite
- Observed result: 212 passed, 2 skipped because PyTorch was unavailable; all
  boundary, ignore, empty-tile, determinism, hash, and immutability checks passed
- Validation level: CPU annotation transformation and manifest generation
- Unverified: real image crop materialization and high-resolution GPU training
- Compatibility effect: additive performance-only dataset variant
- Deviation: this PR writes annotation/manifest artifacts; image pixels are materialized by
  the runtime producer using the manifest's deterministic crop coordinates
- Rollback: revert PR 23 and delete only `datasets/tiles/v1` generated artifacts
- Commit SHA: `6a442c216c15792e11d75e5673f075af8306de14`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/35
- Remaining risks: image decoder orientation metadata must be normalized during live crop

## PR 24 — Sliced inference

- Status: completed with live model inference pending
- Dependencies: PR 1–23
- Files changed: sliced inference/coordinate restoration/class-aware merge; prediction
  schema/config; isolated output paths; tests; notebook 07; migration log/inventory
- Conceptual change: restore slice-local boxes globally and merge duplicates without
  sharing full-image artifact namespaces
- Preserved behavior: category IDs/scores and full-image predictions remain untouched
- Tests executed: one-slice equivalence; multi-slice offsets; class preservation;
  duplicate merge; isolated metrics/predictions; separate latency; full suite
- Observed result: 215 passed, 2 skipped because PyTorch was unavailable; coordinate,
  equivalence, merge, class, namespace, and latency contracts passed
- Validation level: CPU synthetic geometry and callback inference
- Unverified: real model slice batching and GPU latency
- Compatibility effect: additive performance-only prediction artifact
- Deviation: merge v1 uses deterministic class-aware NMS; weighted box fusion can be
  added as a later schema version without changing v1 results
- Rollback: revert PR 24 and remove only sliced prediction/metric namespaces
- Commit SHA: `d954e5398ef540c9a7f8512a8ab8e04d6e8a8025`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/36
- Remaining risks: optimal overlap and merge threshold remain performance experiments

## PR 25 — Label-granularity ablation

- Status: completed
- Dependencies: PR 1–24
- Files changed: direct/merged label-granularity configs; mapping module; label-space
  schema; class-merge/isolation tests; documentation/log/inventory
- Conceptual change: evaluate original-10-class training in merged person/vehicle space
- Preserved behavior: ignored categories 0/11 and direct two-class evaluation semantics
- Tests executed: all ten category mappings; ignored/unknown IDs; prediction provenance;
  manifest hashes; direct-versus-merged comparison rejection; full suite
- Observed result: 218 passed, 2 skipped because PyTorch was unavailable; all ten
  mappings, ignore, provenance, manifest, and comparison-isolation checks passed
- Validation level: CPU synthetic prediction/annotation mappings
- Unverified: end-to-end original-class trained checkpoint evaluation
- Compatibility effect: additive performance ablation with an isolated namespace
- Deviation: records preserve source category provenance rather than destructively replacing it
- Rollback: revert PR 25 and remove only label-granularity ablation artifacts
- Commit SHA: `d8b65c6014fc3685347e035a4bcc975764782aad`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/37
- Remaining risks: overlapping original-class predictions may require class-merge NMS policy

## PR 26 — Versioned evaluator and comparison pipeline

- Status: completed
- Dependencies: PR 1–25
- Files changed: legacy/versioned evaluation adapters; v2 metric envelopes; five isolated
  comparison tables; canonical evaluation/comparison notebooks; parity tests/log/inventory
- Conceptual change: normalize old/new predictions before one metric implementation and
  separate controlled, performance, full, sliced, and ensemble result tables
- Preserved behavior: frozen legacy fixture metrics and untagged controlled artifacts
- Tests executed: same fixture through old, legacy-adapter, and versioned paths within
  tolerance; table membership/output isolation; notebook/full CPU suite
- Observed result: 220 passed, 2 skipped because PyTorch was unavailable; exact
  old/adapter/new fixture parity and all five table-isolation checks passed
- Validation level: CPU deterministic legacy/versioned fixture parity
- Unverified: pycocotools parity on a full validation artifact
- Compatibility effect: additive evaluator v2; legacy files remain readable through adapter
- Deviation: canonical notebooks 30 and 03 replace nonexistent planned notebooks 05–06
- Rollback: revert PR 26 and restore original evaluation/comparison notebook calls
- Commit SHA: `09ec7fed7e08226b0c5b6a38ac9a4d7164692936`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/38
- Remaining risks: external metrics with unsupported fields require explicit adapters

## PR 27 — CI and release policy

- Status: completed with manual GPU/Colab validation deferred by policy
- Dependencies: PR 1–26
- Files changed: four required-check workflows; configuration/schema validator;
  notebook validation API; PR template; GPU checklist; release policy; tests/cross-platform inventory
- Conceptual change: make static, CPU, notebook, schema, artifact, and security
  validation explicit, independently visible pull-request checks
- Preserved behavior: existing umbrella CI and result validation remain available
- Tests executed: malformed config, prohibited artifact, malformed notebook;
  configuration/schema, notebook, Ruff, compile, and full CPU suite
- Observed result: 225 passed, 2 skipped because PyTorch was unavailable; all
  static, config/schema, artifact, secret, and notebook validation passed
- Validation level: local CPU/static parity with GitHub Actions commands
- Unverified: protected-branch administrator settings and manual GPU/Colab checklist
- Compatibility effect: additive workflows with no GPU secrets or private data
- Deviation: existing `ci.yml` remains as an umbrella compatibility check while the
  four granular workflows provide stable names for branch protection
- Rollback: revert PR 27 or disable the four granular required checks in branch protection
- Commit SHA: `1d10a22fdb27af1fedc6d6d42c930e3c16b8c3a6`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/39
- Remaining risks: repository administrators must select the documented job names as required checks

## PR 28 — Deprecate notebook-local legacy logic

- Status: completed with manual GPU/Colab validation pending
- Dependencies: PR 1–27
- Files changed: canonical dataset/model-day notebooks; shared notebook entrypoint helper;
  thin-notebook validator/tests; archive policy; README; deprecation guide; cross-platform inventory/log
- Conceptual change: canonical notebooks now contain parameters, bootstrap, presentation,
  and package calls only; notebook-local workflow definitions are rejected by CI
- Preserved behavior: dataset recovery instructions, model-day orchestration, model-specific
  HPO/final smoke entrypoints, and legacy artifact reads
- Tests executed: thin/package-backed structure; one HPO/final smoke entrypoint per model;
  no browser uploads; legacy prediction read; notebook validation/cleaning; full CPU suite
- Observed result: 229 passed, 2 skipped because PyTorch was unavailable; all 14
  guarded canonical workflow notebooks passed CPU smoke execution
- Validation level: CPU synthetic and static notebook contract validation
- Unverified: real Drive download and GPU execution in Colab
- Compatibility effect: old artifacts remain readable; old notebooks remain in Git history
- Deviation: planned notebooks 04–06 are represented by model-specific 10–23 and versioned
  30–31 flows, avoiding duplicate canonical entry points
- Rollback: revert PR 28 notebook/docs changes; keep package and compatibility code intact
- Commit SHA: `9cfa488dc04e49485a8b7cdffb8051d5ca754247`
- PR URL: https://github.com/Harryphan72007/aerial-object-detection-benchmark/pull/40
- Remaining risks: framework/GPU behavior still requires the documented manual checklist
