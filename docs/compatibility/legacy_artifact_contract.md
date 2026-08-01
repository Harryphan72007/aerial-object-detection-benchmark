# Legacy artifact compatibility contract

Status: frozen as `legacy-artifacts-v1` by migration PR 1. This change documents
existing behavior; it does not alter producers, consumers, notebooks, or Drive
paths.

## Repository topology adjustment

The migration plan expected notebooks 01 through 06. They are not present in
this repository. The current canonical sequence is 00 through 03, model HPO
notebooks 10 through 13, fine-tuning notebooks 20 through 23, and evaluation and
publication notebooks 30 and 31. The three notebooks under `notebooks/optional`
are also consumers and are included in the checked inventory. The generated
snapshot `schemas/legacy/notebook_artifact_inventory_v1.json` records every
current notebook SHA, import, and artifact-reference line. It also records the
same evidence for artifact-producing or artifact-consuming Python sources under
`src` and `scripts`, closing the gap between thin notebooks and their workflows.

Regenerate and review the snapshot with:

```bash
python -m scripts.validation.inventory_legacy_artifacts \
  --output schemas/legacy/notebook_artifact_inventory_v1.json
```

CI-compatible drift check:

```bash
python -m scripts.validation.inventory_legacy_artifacts \
  --check schemas/legacy/notebook_artifact_inventory_v1.json
```

## Storage and discovery paths

All paths below are relative to the configured Drive root unless stated
otherwise. They are constructed by `src.paths.ProjectPaths`.

| Artifact | Existing path or pattern | Discovery consumer |
|---|---|---|
| Raw VisDrone splits | `datasets/VisDrone2019-DET/raw/VisDrone2019-DET-{train,val}` | dataset preparation and loaders |
| COCO annotations | `datasets/VisDrone2019-DET/processed/coco_{2class,10class}/annotations/instances_{split}.json` | training and evaluation |
| Dataset manifests | `datasets/VisDrone2019-DET/manifests/**.json` | data setup and LR workflow |
| Ordinary run directory | `checkpoints/<model_id>/<run_id>/` | trainer and `RunRegistry` |
| HPO search checkpoints | `checkpoints/lr_search/` | LR/HPO workflows |
| Final checkpoints | `checkpoints/final/<model_id>/<run_id>/` | comparison and publishing workflows |
| Registry JSON | `experiment_registry/checkpoint_registry.json` | `RunRegistry` and all run discovery |
| Registry CSV | `experiment_registry/runs.csv` | analysis/export convenience only |
| Predictions | `predictions/<run_id>__<split>__res<resolution>.json` | evaluators |
| Evaluation metrics | `evaluation/<run_id>__res<resolution>__metrics.json` | comparison, report, and bundle export |
| Profiles | `evaluation/<run_id>__profile.json` | comparison and bundle export |
| Failures | `evaluation/evaluation_failures.json` | evaluation workflow |
| Aggregate comparison | `evaluation/comparison_<track>_<split>.json` | report and notebook consumers |
| Reports | `reports/**` | comparison and publishing notebooks |
| Runtime bundles | `result_bundles/<bundle_id>/` | publisher |
| Git-published bundle | `results/bundles/<bundle_id>/` | repository result consumers |
| Latest Git manifest | `results/manifests/latest_result_manifest.json` | result validation |

`ProjectPaths.create()` initializes the registry as
`{"schema_version": 1, "runs": {}}` and initializes `runs.csv` with
`run_id,model_id,status`. These names are discovery contracts and must not be
renamed without an adapter.

## Dataset and prediction contract

COCO ground truth retains `images`, `annotations`, and `categories`. Each
annotation uses `id`, `image_id`, `category_id`, `bbox` in `[x, y, width,
height]`, `area`, and `iscrowd`. Predictions are a JSON array whose records use
`image_id`, `category_id`, `bbox`, and `score`. The frozen prediction schema is
`schemas/legacy/coco_predictions_v1.schema.json`.

For the two-class track, category ID 1 is `person` and category ID 2 is
`vehicle`. `person` collapses pedestrian and people. `vehicle` collapses
bicycle, car, van, truck, tricycle, awning-tricycle, bus, and motor. The 2-class
and 10-class tracks are not interchangeable.

## Checkpoints, run manifests, and registry

Checkpoint aliases are exactly `last.pth`, `best_map.pth`, and
`best_aptiny.pth`. Framework-native epoch/best checkpoints may coexist with
these aliases. Existing code saves checkpoints atomically and retains no
portable promise about framework-internal tensor keys across model families.
Legacy checkpoint files therefore remain evaluation or legacy-trainer inputs;
they must never be rewritten in place.

The required run-manifest keys are frozen by
`schemas/legacy/run_manifest_v1.schema.json` and the existing
`src.training.checkpointing.MANIFEST_REQUIRED` set:

```text
run_id, model_id, architecture_family, dataset_track, class_names, seed,
input_resolution, checkpoint_best_map, checkpoint_best_aptiny,
checkpoint_last, config_path, created_at, framework, framework_version,
pytorch_version, cuda_version, gpu_name, total_parameters,
trainable_parameters, frozen_parameters, best_validation_map,
best_validation_aptiny, best_epoch, total_training_seconds, status
```

Allowed status values are `created`, `running`, `completed`, `failed`, and
`interrupted`. Producers may add fields. Consumers must tolerate unknown fields.
The registry is `{"schema_version": 1, "runs": {run_id: manifest}}`; the map key
must equal the nested `run_id`. `runs.csv` is regenerated from the union of
manifest keys and is not authoritative.

## Metrics and reports

The evaluator writes `evaluation/<run_id>__res<resolution>__metrics.json`.
Existing consumers rely on these identity keys:

```text
run_id, model_id, architecture_family, dataset_track, training_resolution,
evaluation_resolution, seed, evaluation_image_count, prediction_file
```

The stable metric names are `mAP`, `AP50`, `AP75`, `APtiny`, `APsmall`,
`APmedium`, `APlarge`, `ARtiny`, `ARsmall`, `ARmedium`, `ARlarge`, and
`AR<maxDets>`. `per_class` is keyed by class name and contains `AP`, `AP50`, and
`AP75`. Optional detailed, calibration, confidence, error, latency, throughput,
memory, parameter-count, and training-time fields are additive. The frozen
minimum is in `schemas/legacy/evaluation_metrics_v1.schema.json`.

Controlled comparison additionally requires seed 42, evaluation resolution
640, a valid final config, and a completed batch-one profile. It emits:

```text
reports/comparison/comparison.csv
reports/comparison/comparison.json
reports/comparison/comparison.md
reports/comparison/accuracy_latency.png
reports/comparison/accuracy_memory.png
```

The comparison CSV columns are `Model`, `Architecture family`, `Selected LR`,
`mAP50-95`, `APtiny`, `person AP`, `vehicle AP`, `parameters`, `training time`,
`peak memory`, `latency`, `FPS`, `status`, and `run_id`.

## Lightweight result-bundle contract

Published bundles exclude checkpoints, raw predictions, databases, archives,
credentials, and private absolute paths. Existing schema-v2 bundles require:

```text
bundle_manifest.json
README.md
configs/selected_lr.yaml
configs/final_resolved_config.yaml
search/candidates.csv
search/promotion_history.csv
search/search_summary.json
metrics/final_metrics.json
metrics/per_class_metrics.csv
metrics/profiling_summary.json
reports/model_report.md
provenance/environment_summary.json
provenance/dataset_hashes.json
provenance/git_commit.txt
```

`bundle_manifest.json` remains the bundle discovery contract. Its required keys
are defined by `src.result_export.REQUIRED_MANIFEST_FIELDS`. HPO schema-v3
bundles use the separate required set in the same module. New producers must
dual-write these legacy layouts until all current consumers have adapters.

## Compatibility matrix and change rules

| Producer | Consumer | Frozen behavior |
|---|---|---|
| Existing trainer | `RunRegistry` | Manifest and aliases remain readable |
| Existing evaluator | COCO prediction JSON | Four required prediction fields remain unchanged |
| Existing evaluator | metrics consumers | Identity and stable metric keys remain unchanged |
| Existing registry | comparison notebooks | Registry v1 and run discovery patterns remain unchanged |
| Existing publisher | repository results | Lightweight bundle names and required files remain unchanged |
| New producer | existing consumer | Dual-write or adapt to this contract |

Any intentional contract change must update the applicable schema, fixture,
inventory snapshot, this document, and a compatibility test in one focused PR.
Never mutate a legacy artifact in place. Introduce a new schema version and keep
the old reader path until parity is proven.

## Representative fixture and verification

`tests/fixtures/legacy_artifacts` contains small JSON/CSV examples only; it
contains no images, model weights, checkpoints, or generated metrics. The
existing detailed evaluator reads the ground-truth and prediction fixtures in
`tests/test_legacy_artifact_contract.py`. The same test verifies the current
manifest validator, registry shape, CSV columns, schemas, and notebook snapshot.

PR 1 validation level is CPU/static only. It does not claim Colab, CUDA, GPU,
training, resume, latency, or mAP validation.
