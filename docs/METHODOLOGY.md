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

`assert_final_training_uses_official_train` and the authoritative data preflight
retain globally remapped numeric validation IDs and also compare stable identity:
`file_name`, `original_split`, and source archive SHA-256. They prove search
train/search validation filename disjointness, membership in official train,
complete final-train identity, and exclusion of every official validation image
before training. These assertions are saved in `split_summary.json`. A compatible
final run records
`run_kind=final_complete_official_train`.

Evaluation uses the complete official validation split, common COCO metrics,
per-class and object-size metrics, and standardized synchronized profiling. YOLOX
and multidimensional Optuna search are outside this controlled benchmark.

## Data provenance without methodology changes

Each extraction manifest binds image/annotation counts and the sorted relative
filename inventory to the archive SHA-256 and byte size. Each conversion
manifest binds its output hash and counts to that extraction inventory, class
mapping, light-vehicle policy, converter schema, and repository commit. Existing
artifacts are reused only when these fields remain current.

The persistent source images are always:

```text
$DRIVE_ROOT/datasets/VisDrone2019-DET/raw/VisDrone2019-DET-train/images
$DRIVE_ROOT/datasets/VisDrone2019-DET/raw/VisDrone2019-DET-val/images
```

Processed COCO storage contains annotations rather than copied images. An
optional verified local Colab cache changes only read performance; it does not
change any split, sample, metric, hyperparameter, checkpoint location, or result
location.

## End-to-end path audit

| Transition | Authoritative artifact or read root |
|---|---|
| download → archive storage | `$DRIVE_ROOT/datasets/VisDrone2019-DET/archives/*.zip` |
| archive validation | size, ZIP layout, CRC, SHA-256, and `manifests/{train,val}_archive.json` |
| extraction → raw storage | `raw/VisDrone2019-DET-{train,val}/{images,annotations}` |
| extraction validation | `manifests/{train,val}_extraction.json` |
| COCO conversion | `processed/coco_2class/annotations/instances_{train,val}.json` |
| conversion validation | adjacent `conversion_manifest_{train,val}.json` and audit JSON |
| LR-search manifest generation | `manifests/lr_search/` |
| adapter smoke and LR search | train images from official raw train; both search JSONs from `manifests/lr_search/` |
| final full-dataset training | complete official-train JSON plus official raw train images |
| final evaluation | official-validation JSON plus official raw validation images |

When local caching is enabled, only the image and JSON read roots in the last
three rows are replaced by their verified `/content/visdrone_cache` mirrors.
Adapter gates, candidate checkpoints, search state, final checkpoints,
predictions, metrics, profiling, and reports continue to write under
`$DRIVE_ROOT`.
