# GPU validation checklist

No full run (HPO or final training) may start until the model has a stored
**READY** adapter-smoke record at the current commit, environment, dataset track,
and resolution. `src.workflows.adapter_gate.require_ready_adapter_gate` enforces
this from `TwoStageRandomHPO.run` and `FinalExperimentWorkflow.run`; it is not
advisory.

## Two smoke levels, and only one of them proves GPU readiness

| | Level 1 — CI notebook smoke | Level 2 — GPU adapter smoke |
|---|---|---|
| Command | `python scripts/run_notebook_smoke.py` | `python -m scripts.gpu_adapter_smoke …` |
| Where | CPU, CI | the target GPU |
| Runtime | minutes | minutes per model, after provisioning |
| Validates | notebook execution order, imports safe on CI, path and config resolution, syntax, notebook structure | environment provisioning, framework import, real model construction, pretrained loading, forward/backward, head class count, feature contract, checkpoint roundtrip |
| Proves the models work? | **No** | Yes, for the checked contract |

Level 1 runs every notebook with `SMOKE_TEST=1`. The notebook still calls its
real workflow entry point — that is what proves it is wired to one — but with
the expensive stage forced off, so the pipeline reports its stage contracts and
stops. It is a structural test. **A green Level 1 is compatible with every
adapter being broken** — it never constructs a model, never loads a checkpoint,
and never touches CUDA. Only Level 2 answers that question, and it is
deliberately not part of normal CI because it needs a GPU.

## Run the gate

```bash
python -m scripts.gpu_adapter_smoke --repo-root . --drive-root <artifact_root> --dataset-track 2class
```

Add `--model-id <model>` to limit it to one model. For each model the driver
provisions (or reuses) the isolated model runtime, then re-executes itself
**inside that runtime's interpreter** — the notebook kernel does not have
MMDetection, VMamba, or the pinned Transformers stack, so the checks cannot run
there.

It writes one signed JSON record per model to `<artifact_root>/adapter_smoke/`
(never to `results/` — a smoke pass is not a benchmark result) and exits non-zero
if any model is not READY.

## What each model must pass, in order

1. **constructs** — the configured model builds from its pinned revision, through
   the same config transformations the training backend applies.
2. **pretrained_weights_complete** — the checkpoint is present and its weights
   actually reached the constructed model. The adapter reports one of four
   explicit states — `loaded`, `incomplete`, `missing`, `not_applicable` — and
   anything but `loaded`/`not_applicable` fails. There is no way to express "no
   checkpoint information", because a swallowed load reading as success is
   precisely the historical Swin and VMamba failure mode. Upstream VMamba's
   `Backbone_VSSM.load_pretrained` catches every exception and only *prints* its
   incompatible keys, so this check verifies the load independently.
3. **feature_map_contract** — feature levels have the expected channels and
   strides with correct NCHW spatial dims at the **configured** resolution, never
   224. Faster R-CNN variants are checked at their five FPN levels
   (256ch, strides 4/8/16/32/64). RT-DETRv2 has no FPN, so the
   architecture-equivalent levels are the hybrid encoder's `encoder_input_proj`
   outputs (256ch, strides 8/16/32), captured during a real forward pass.
4. **detection_head_class_count** — read from the *constructed* head
   (`roi_head.bbox_head.num_classes`, or `class_embed[i].out_features` for
   RT-DETRv2), not from a copied config value, and equal to the track class count
   (2 or 10) with no COCO-80 residue.
5. **forward_backward_finite_loss** — one real forward + backward on a two-image
   batch yields a finite loss, gradients on trainable parameters, and a non-zero
   maximum gradient. No optimizer step is taken; CUDA memory is released
   afterwards.
6. **predict_wellformed** — one `predict` returns ordered box corners, scores in
   [0, 1], and one-based COCO category ids within `[1, num_classes]`.
7. **checkpoint_roundtrip** — save to a temporary directory, reload into a freshly
   constructed compatible model with `strict=True`, and compare per-parameter
   digests. The temporary file is always removed.

## Failure behaviour

Every check produces a structured result. A failure stops the sequence and the
record is written anyway, with status `FAILED_ADAPTER` and:

```text
model family · dataset track · image size · failed check · exception type
human-readable error · traceback · timestamp · GPU and environment fingerprint
adapter fingerprint · signature
```

A run that aborts before its checks (provisioning failure, import error, crash)
still produces a complete `FAILED_ADAPTER` record. Missing checks are recorded as
`smoke run stopped before completing`, never as a pass. `READY` requires every
check in the list, in order, all passed.

## Before starting a benchmark

- [ ] A READY record exists for `faster_rcnn_resnet50` at the current commit/GPU.
- [ ] A READY record exists for `faster_rcnn_swin_t`.
- [ ] A READY record exists for `faster_rcnn_vmamba_t`.
- [ ] A READY record exists for `rtdetrv2_l`.
- [ ] Each record's `dataset_track` and `image_size` match the run you will start.
- [ ] The runtime budget (`docs/reference/runtime_budget.md`) has been regenerated
      from a real `t_iter` on this hardware.

The gate binds a record to the model family, adapter fingerprint (commit,
framework, Python/PyTorch/CUDA/GPU, dependency-lock hash, schema version),
environment, dataset track, image size, checkpoint identity, and smoke-contract
version, and verifies the record's own signature. A stale record from a different
adapter, commit, environment, or configuration cannot authorize a run.

The assertion logic behind these checks is unit-tested on CPU
(`tests/test_adapter_gate.py`, `tests/test_adapter_gate_enforcement.py`); the
orchestration around real adapters is validated only by running this script on
hardware.
