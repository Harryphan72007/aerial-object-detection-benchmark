# Third-party license review

Verified 2026-07-30. Source-code licenses and pretrained-artifact terms are
separate; re-check both before redistribution or commercial deployment.

| Family | Exact source | Source license | Pretrained artifact |
|---|---|---|---|
| ResNet-50 Faster R-CNN | MMDetection `44ebd17b145c2372c4b700bfb9cb20dbd28ab64a` | Apache-2.0 | MMDetection model-zoo initialization recorded by each run |
| Swin-T Faster R-CNN | Same MMDetection revision; Swin-derived backbone | Apache-2.0 / MIT | Exact resolved artifact recorded by each run |
| VMamba-T Faster R-CNN | VMamba `2ed52ead062a51a64521ed3871d52914bf532876`; same MMDetection revision | MIT / Apache-2.0 | Required local `vmamba_tiny_e292.pth`; scratch training disabled |
| RT-DETRv2 L (R50) | RT-DETR `a21d516aca15da57e65f35c47659c7535ad2b6b3` | Apache-2.0 | `PekingU/rtdetr_v2_r50vd` revision `282494075698cab9faa1096ae26856890030c817`, Apache-2.0 |

The environment provisioner is uv 0.8.15 (MIT OR Apache-2.0). Runtime manifests
record the selected implementation, revision, artifact, and package/GPU facts.

## Dataset warning

VisDrone2019-DET is downloaded separately and is not redistributed here. Treat
the dataset and all derived results as research-only unless separate permission
is confirmed. Follow the official citation and usage terms.
