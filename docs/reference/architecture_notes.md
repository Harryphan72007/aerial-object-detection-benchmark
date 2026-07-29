# Architecture notes

## ResNet-50 Faster R-CNN

Convolutions build local-to-global C2–C5 features. FPN produces P2–P6, preserving high-resolution features for small objects. RPN proposes candidate regions, and the RoI head classifies/refines them. Inspect C2–C5, P2–P6, proposals, proposal recall, and final RoIs.

## Swin-T Faster R-CNN

Patch embedding creates tokens; window attention limits local cost; shifted windows exchange information; hierarchical stages downsample and increase channels. FPN reuses the four stage outputs. Inspect patch embedding, windowed stages, FPN outputs, and attention rollout only when the installed implementation exposes compatible tensors.

## VMamba-T Faster R-CNN

VMamba uses hierarchical VSS blocks and SS2D selective scan over multiple directions. The official detection tree registers the real `MM_VSSM` backbone. Do not hardcode undocumented child names: print `named_modules()`, identify actual blocks, and attach hooks to those names. Inspect stage outputs, scan-sensitive activations, FPN, and effective receptive field.

## RT-DETRv2

Backbone features enter a hybrid encoder; selected encoder positions initialize object queries/reference boxes; decoder layers refine boxes without NMS. Inspect encoder features, query selection, reference boxes, layer-wise refinement, final predictions, and query utilization. Query-count saturation is especially important in dense aerial scenes.

## Interpretation

Architectural family is not the only causal factor. Resolution, P2, max detections, query count, tiling, pretraining, and recipe can dominate small-object results. The benchmark therefore reports both architecture-default and controlled ablations.
