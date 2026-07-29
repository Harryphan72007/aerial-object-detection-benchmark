# Experiment protocol

## Comparison units

A comparison row is identified by model ID, concrete upstream config/checkpoint, dataset track, class mapping, train/validation split, seed, resolution, recipe, precision, tiling/TTA flags, and hardware/software environment. Rows that differ on any of these fields are not silently pooled.

## Primary experiment grid

- Models: ResNet-50 Faster R-CNN, Swin-T Faster R-CNN, VMamba-T Faster R-CNN, RT-DETRv2 configured large baseline; optional YOLOX-S.
- Tracks: 2-class and original 10-class.
- Seeds: 17, 42, 3407.
- Resolutions: 640, 1024, 1280.
- Recipes: architecture-default and shared-controlled.
- Precision: AMP/FP16 and FP32 ablation.

## Controlled recipe

Use the same train/validation images, augmentation family, epoch budget, effective batch size, optimizer family where architecturally reasonable, validation frequency, max detections, and early-stop interpretation. Architecture-specific optimizers remain allowed but must be marked as default-recipe results.

## Statistical reporting

Report mean, standard deviation, minimum, maximum, and 95% confidence interval across seeds. Do not declare a winner from a single run or from a difference smaller than uncertainty. When predictions exist on the same images, use paired image bootstrap for mAP/APtiny differences.

## Ablations

P2 feature level, resolution, tiled inference, RT-DETR query count, max detections, pretraining source, frozen stages, augmentation strength, single/multi-scale training, AMP/FP32, and default/shared recipe. Change one controlled factor at a time.
