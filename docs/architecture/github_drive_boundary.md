# GitHub and Google Drive storage boundary

This boundary is enforced by `.gitignore`, repository validation, result-bundle
validation, and review. It applies to local development and hosted Colab.

## GitHub is authoritative for

- canonical notebooks and reusable Python source;
- configuration, schemas, small fixtures, and validation scripts;
- tests, CI workflows, documentation, licensing, and security policy;
- validated lightweight result bundles under `results/bundles` and their latest
  manifest under `results/manifests`.

## Google Drive is authoritative for

- raw or converted VisDrone datasets and generated tiles;
- pretrained weights, checkpoints, optimizer/scheduler/scaler/EMA states;
- predictions, evaluation outputs, profiling traces, plots, and reports;
- Optuna SQLite databases and snapshots;
- TensorBoard, framework clones, compiled extensions, logs, and caches.

The repository must contain references, hashes, schemas, and provenance for
these artifacts—not the artifacts themselves.

## Allowed exceptions

Small reviewed fixtures under `tests/fixtures` may use JSON, CSV, YAML, text, or
small media/archive files needed for deterministic tests. They are limited to
1 MiB each and must not contain real credentials, private paths, real model
weights, or copyrighted dataset samples.

Published result bundles are limited to portable, lightweight formats and are
validated separately by `scripts.validate_results`. Raw predictions and model
files remain prohibited even when small.

## Enforcement

Run against tracked files:

```bash
python -m scripts.validation.check_prohibited_files
```

Run against the Git index before committing:

```bash
python -m scripts.validation.check_prohibited_files --staged
```

The validator rejects prohibited directories and extensions, credential-like
filenames, known secret patterns, unapproved published-result types, files over
20 MiB, and fixtures over 1 MiB. Secret scanning remains a separate defense.

If a prohibited artifact is staged, unstage it and move it to the configured
Drive root. If a credential was committed or pushed, revoke it immediately and
follow `SECURITY.md`; do not attempt an uncoordinated history rewrite.
