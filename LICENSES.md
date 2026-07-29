# Third-party license review

Verified on **2026-07-25**. Re-check every source repository and every weight artifact before redistribution or commercial deployment. This project is a research benchmark; it does **not** claim that VisDrone2019-DET is commercially usable.

| Model | Implementation repository | Source-code license | Pretrained-weight source | Weight notes | Verified |
|---|---|---|---|---|---|
| Faster R-CNN ResNet-50-FPN | `open-mmlab/mmdetection` | Apache-2.0 | MMDetection model zoo / TorchVision ImageNet initialization | Preserve notices; verify the exact weight-host terms and training-data constraints before redistribution. | 2026-07-25 |
| Faster R-CNN Swin-T-FPN | `open-mmlab/mmdetection`, backbone derived from `microsoft/Swin-Transformer` | Apache-2.0 / MIT | MMDetection or Microsoft Swin release assets | The code license and a weight artifact's permitted uses are separate questions. Record the exact URL and checksum in each run manifest. | 2026-07-25 |
| Faster R-CNN VMamba-T-FPN | `MzeroMiko/VMamba` detection tree | MIT | User-supplied, locally verified VMamba classification checkpoint | Set `VMAMBA_T_PRETRAINED` to the local file. When unset, the project clears the stale upstream path, trains from scratch, and writes `PRETRAINING_WARNING.txt`. VMamba's detection directory is based on MMDetection 3.3.0; check transitive CUDA-extension licenses and weight provenance. | 2026-07-25 |
| RT-DETRv2-L | `lyuwenyu/RT-DETR`; compatible `huggingface/transformers` integration | Apache-2.0 | `PekingU/rtdetr_v2_r50vd` or another explicitly configured official conversion | The public naming uses backbone variants rather than a literal “L” checkpoint in every integration. This benchmark records the exact model ID and architecture fields; do not silently substitute variants. | 2026-07-25 |
| YOLOX-S (optional control) | `Megvii-BaseDetection/YOLOX` | Apache-2.0 | Official YOLOX-S release | Verify the specific checkpoint artifact and any deployment runtime separately. | 2026-07-25 |

## Dataset warning

VisDrone2019-DET is downloaded separately. The repository does not redistribute images or annotations. Review the dataset's official terms, citation requirements, and restrictions yourself. Treat this project and all produced results as **research-only** unless you obtain independent permission for another use.
