# Colab compatibility matrix

Research and repository verification date: 2026-07-30. CPU smoke verification
does not imply GPU verification.

| Family | Hosted Colab design | Status |
|---|---|---|
| Shared kernel | Current hosted kernel, dataset stack, Optuna 4.5.0 | CPU synthetic smoke tested |
| RT-DETRv2 | Isolated Python 3.11.13, PyTorch 2.7.1+cu128 | Pinned; real GPU adapter gate required |
| ResNet-50 / Swin-T | Isolated Python 3.10.16, PyTorch 2.1.0+cu118, MMCV 2.1.0 | Pinned; real GPU adapter gate required |
| VMamba-T | OpenMMLab environment plus exact VMamba commit/selective scan | GPU/extension unverified; pretrained checkpoint required |

No custom/local Colab runtime is required. Import probes do not establish
compatibility; the construction, one-batch train/predict, save/reload gate does.
