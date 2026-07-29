# Model environments

Notebook 01 determines the model family before installing or validating packages.
The dataset notebook installs only `requirements-dataset-colab.txt`.

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
