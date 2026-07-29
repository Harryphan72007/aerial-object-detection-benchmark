# Published results

This directory contains only validated, lightweight benchmark outputs exported from a Drive result bundle. Checkpoints, datasets, raw predictions, profiling traces, TensorBoard logs, credentials, and private paths are intentionally excluded.

Each immutable per-model publication lives under
`bundles/<RESULT_BUNDLE_ID>/`. The current publication is described by
`manifests/latest_result_manifest.json`. Use
`python -m scripts.validate_results --repo-results results` before staging or
committing.
