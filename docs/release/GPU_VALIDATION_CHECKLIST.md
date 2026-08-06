# GPU validation checklist

No full run (HPO or final training) may start until every controlled-track model
has a stored **READY** adapter-smoke record at the current commit and GPU. A
green CPU CI is compatible with all adapters being broken; this gate is what
proves they are not.

## Run the gate

On the target GPU, with the model environments provisioned:

```bash
python -m scripts.gpu_adapter_smoke --repo-root . --drive-root <artifact_root> --dataset-track 2class
```

It writes one signed JSON record per model to `<artifact_root>/adapter_smoke/`
(never to `results/` — a smoke pass is not a benchmark result) and exits non-zero
if any model is not READY.

## What each model must pass, in order

1. **constructs** — builds from the pinned upstream revision.
2. **pretrained_weights_complete** — weights load with **zero** missing/unexpected
   keys. A partial load raises; a warn-only load is not accepted. This explicitly
   covers the historical Swin and VMamba partial-load failure modes.
3. **feature_map_contract** — backbone/FPN feature maps have the expected channels
   and strides, with correct NCHW spatial dims at the **configured** resolution
   (640 and 1024), never 224.
4. **detection_head_class_count** — the head is reset to the track class count
   (2 or 10) with no COCO-80 residue.
5. **forward_backward_finite_loss** — one forward + backward on a 2-image batch
   yields a finite loss.
6. **predict_wellformed** — one `predict` returns well-formed boxes (ordered
   corners, scores in [0, 1], labels within the class count).
7. **checkpoint_roundtrip** — `save` → `load` yields identical parameters.

## Before starting a benchmark

- [ ] A READY record exists for `faster_rcnn_resnet50` at the current commit/GPU.
- [ ] A READY record exists for `faster_rcnn_swin_t`.
- [ ] A READY record exists for `faster_rcnn_vmamba_t`.
- [ ] A READY record exists for `rtdetrv2_l`.
- [ ] Each record's `commit` matches the commit you will run, and `gpu` matches
      the hardware.
- [ ] The runtime budget (`docs/reference/runtime_budget.md`) has been regenerated
      from a real `t_iter` on this hardware.

The assertion logic behind these checks is unit-tested on CPU
(`tests/test_adapter_gate.py`); the orchestration around real adapters is
validated only by running this script on GPU.
