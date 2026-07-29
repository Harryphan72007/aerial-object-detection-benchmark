# Evaluation protocol

## Accuracy

COCO AP uses IoU 0.50:0.95. The custom aerial bins are: tiny `<16²`, small `[16²,32²)`, medium `[32²,96²)`, and large `>=96²` pixels by box area. Standard COCO small/medium/large should be labeled separately if also reported.

Calibration uses class-aware greedy one-to-one matching at a documented IoU threshold. A matched prediction is a positive outcome; unmatched predictions are negative outcomes. This avoids treating every detection score as an independent image-class probability.

Occlusion/truncation slices use the attributes retained during VisDrone conversion. Density bins are quartiles derived from the training split and saved in dataset statistics.

## Efficiency

Use 100 warm-ups and at least 500 timed iterations, CUDA synchronization around timing, and no disk/data loading in pure-forward timing. Report preprocessing, transfer, forward, postprocessing, and end-to-end separately. Report mean, median, p90, p95, p99, batch-1 FPS, batch throughput, and peak VRAM.

Record PyTorch FP32, PyTorch AMP/FP16, ONNX Runtime, and TensorRT FP16 independently. An unsupported operator or failed export is a valid result with the error recorded.

## Published results

Only ten-class rows may be compared with published VisDrone ten-class results. Keep single-model, ensemble, tiled, and TTA results in separate groups. Missing values remain empty or `not reported`.
