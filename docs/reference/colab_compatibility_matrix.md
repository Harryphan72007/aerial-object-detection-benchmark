# Colab compatibility matrix

Verified/researched on 2026-07-29. “Smoke verified” means the notebook's
CPU-compatible path actually ran with `nbclient`; it does not mean GPU training
ran.

| Environment | Python / PyTorch / CUDA | Pinned files | Notebooks | Status | Exact action |
|---|---|---|---|---|---|
| Shared dataset/evaluation | Colab runtime 2026.04: Python 3.12.13, PyTorch 2.10.0; GPU optional for setup | `requirements-dataset-colab.txt` | 00 setup, 01, 07, 09, 10, 11 | CPU smoke verified | Choose current hosted runtime. GPU is not needed until profiling/evaluation of a real model. |
| RT-DETRv2 large | Current hosted Colab GPU plus supplied PyTorch | `requirements-rtdetr-colab.txt`; checkpoint `PekingU/rtdetr_v2_r101vd` | 05 | Dataset/preflight smoke verified; GPU unverified | Choose an NVIDIA GPU runtime, install only this requirement file, rerun after any pip restart notice, and run notebook 05. |
| Faster R-CNN ResNet-50/Swin-T | Python 3.10, PyTorch 2.1.0+cu118, torchvision 0.16.0+cu118, MMCV 2.1.0, MMEngine 0.10.7, MMDetection 3.3.0 | `requirements-openmmlab-py310-cu118.txt` | 02, 03 | Dataset/preflight smoke verified; GPU unverified | Use a Colab local/custom runtime with this exact stack. Standard hosted 2026.04 is not compatible with the pinned MMCV binary stack. |
| Faster R-CNN VMamba-T | Same OpenMMLab stack plus VMamba commit `2ed52ead062a51a64521ed3871d52914bf532876` and its selective-scan extension | `requirements-openmmlab-py310-cu118.txt` | 04 | Dataset/preflight smoke verified; extension/GPU unverified | Use a custom/local Colab runtime with CUDA 11.8 and a compiler. Build selective scan with `--no-build-isolation`; verify import before training. |
| Optional YOLOX-S | Not pinned as a runnable adapter | none | 06 | Honest early stop only | Do not use for benchmark results until the official predictor adapter is implemented and tested. |

## Why the environments are separate

Google's [runtime version FAQ](https://research.google.com/colaboratory/runtime-version-faq.html)
lists the 2026.04 image as Python 3.12.13/PyTorch 2.10.0. MMDetection 3.3.0
requires MMCV `>=2.0,<2.2` and MMEngine `>=0.7.1,<1` according to its
[compatibility table](https://mmdetection.readthedocs.io/en/v3.3.0/notes/faq.html).
The upstream [VMamba setup](https://github.com/MzeroMiko/VMamba) documents the
older Python/PyTorch/MMCV stack and a compiled selective-scan extension. MMCV's
[installation guide](https://mmcv.readthedocs.io/en/2.x/get_started/build.html)
explains that an unmatched PyTorch/CUDA combination falls back to a source
build. That fallback is not treated as a verified Colab installation.

The shared notebook therefore installs no model framework. This prevents a
successful dataset setup from being invalidated by one model family's binary
dependencies.

## Verified dataset source

The official [VisDrone dataset repository](https://github.com/VisDrone/VisDrone-Dataset)
publishes the train/validation distributions. The automatic token-free path
uses the public GitHub release assets used by the Ultralytics VisDrone recipe:

- `https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip`
- `https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-val.zip`

The repository does not claim a publisher-provided checksum. It calculates
SHA-256 after download, records URL/size/date/hash, verifies ZIP CRC and layout,
and requires future cache restores to match that manifest.

