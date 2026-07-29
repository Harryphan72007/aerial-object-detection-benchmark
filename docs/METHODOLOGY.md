# Controlled benchmark methodology

This cleanup does not change the benchmark design.

```yaml
dataset_track: 2class
search_seed: 42
image_size: 640
effective_batch_size: 8
amp: true

learning_rate_candidates: 9
search_rungs:
  - {epoch: 2, keep: 5}
  - {epoch: 5, keep: 3}
  - {epoch: 10, keep: 2}
  - {epoch: 15, keep: 1}

final_epochs: 25
primary_metric: mAP_50_95
secondary_metric: APtiny
```

Only learning rate changes between candidates. The optional range test informs
the logarithmic nine-value grid but does not become a promotable candidate.
Successive halving resumes each candidate from its own checkpoint, retains a
15-epoch scheduler horizon, ranks moving-window mAP50–95, and breaks ties with
APtiny.

Search train and validation are deterministic, seed-42 subsets drawn only from
official train. Final fine-tuning discards search weights, reloads the original
pretrained model, uses every official training image for 25 epochs, and excludes
every official validation image.

`assert_final_training_uses_official_train` and the final workflow compare image
IDs for exact train identity and prove train/validation disjointness before
training. A compatible final run records
`run_kind=final_complete_official_train`.

Evaluation uses the complete official validation split, common COCO metrics,
per-class and object-size metrics, and standardized synchronized profiling. YOLOX
and multidimensional Optuna search are outside this controlled benchmark.
