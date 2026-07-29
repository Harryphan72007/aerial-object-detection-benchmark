# Model environments

Notebook 01 determines the model family before installing or validating packages.
Both notebook 00 and the fresh-session bootstrap in notebook 01 install only the
CPU/GPU-neutral `requirements-dataset-colab.txt` before initial data/status
inspection. Notebook 01 does not depend on notebook 00's Python process.

## Drive persistence and local Colab cache

The two ZIP files, extracted raw images/annotations, COCO JSON, extraction and
conversion manifests, LR manifests, checkpoints, search state, metrics, and
reports are persistent under `$DRIVE_ROOT`. Processed COCO directories do not
contain image copies.

In Colab, notebook 01 defaults to `DATA_ACCESS_MODE="local_cache"` and
synchronizes only the required image and JSON read view to:

```text
/content/visdrone_cache/
├── train/images/
├── val/images/
└── annotations/
    ├── coco_2class/
    └── lr_search/
```

The cache is verified against the persistent Drive image count and filename
inventory, is rebuilt after runtime loss, and is never used for checkpoints or
results. `DATA_ACCESS_MODE="drive_direct"` reads the same verified dataset
directly. If local capacity is insufficient, the cache request stops and asks
the user to switch explicitly; it never silently changes modes.

## RT-DETRv2

Use a supported hosted Colab NVIDIA GPU runtime. Notebook 01 installs
`requirements-rtdetr-colab.txt`, verifies PyTorch/CUDA/Transformers, and resolves
the canonical `PekingU/rtdetr_v2_r101vd` checkpoint. If pip replaces imported
packages, restart the session once and rerun from the top.

## ResNet-50 and Swin-T Faster R-CNN

Use a Colab local/custom or local GPU runtime with:

```text
Python 3.10
PyTorch 2.1.0+cu118
torchvision 0.16.0+cu118
NumPy 1.26.4
MMCV 2.1.0
MMEngine 0.10.7
MMDetection 3.3.0
```

Notebook 01 fails before an unsafe MMCV source build when Python, Torch, or CUDA
is incompatible. In a compatible runtime it installs missing pinned packages,
checks out MMDetection `v3.3.0`, sets `MMDET_ROOT`, and runs the adapter gate.

## VMamba-T

VMamba uses the same OpenMMLab stack plus:

- VMamba commit `2ed52ead062a51a64521ed3871d52914bf532876`;
- a working `selective_scan_cuda` build;
- the canonical pretrained checkpoint at
  `<DRIVE_ROOT>/pretrained/vmamba_tiny_e292.pth`.

Notebook 01 sets `VMAMBA_ROOT` and `VMAMBA_T_PRETRAINED` only after verification.
It refuses to silently train VMamba from scratch.

## Adapter gate

Before search, each selected adapter must construct, train one synthetic-small
batch, run forward/backward/optimizer operations, predict validation data, save a
checkpoint, and reload it. Persistent status is one of `READY`,
`FAILED_ENVIRONMENT`, `FAILED_ADAPTER`, or `FAILED_OOM`.

Only effective-batch-8 policies are attempted: batch 2/accumulation 4, then batch
1/accumulation 8.
