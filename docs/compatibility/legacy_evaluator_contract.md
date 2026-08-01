# Legacy evaluator and comparison contract

Migration PR 10 freezes the existing CPU evaluator and controlled-comparison
reader before any model producer is migrated.

The detailed evaluator continues to accept COCO ground truth and a flat COCO
prediction array. Its deterministic fixture locks per-class matching, confidence
selection, false-positive rate, and tiny-object miss rate.

The controlled comparison reader continues to accept only completed 2-class runs
whose training configuration is seed 42, resolution 640, 25 epochs, effective
batch size 8, AMP enabled, and `final_complete_official_train`. It reads legacy
metric and batch-one profile filenames. A newer smoke run is deliberately present
in the fixture and must never become a comparison row.

The two representative controlled models are Faster R-CNN ResNet-50 and
RT-DETRv2-L. Missing primary models remain `MISSING`; an incompatible candidate is
recorded under `rejected`. These are compatibility expectations, not benchmark
quality claims.
