# Repository cleanup manifest

Audit baseline: `origin/main` at `e104f29` (155 tracked files). Every baseline file was content-inspected alongside imports, Markdown/notebook links, shell commands, tests, workflow references, artifact policy, and Git ignore rules before removal.

Classifications: `KEEP` remains active; `MERGE` was consolidated into a replacement; `MOVE` keeps useful content at a clearer path; `DELETE` has no remaining live role; `GENERATED` is runtime output and must remain outside Git.

## Baseline file decisions

### `.gitattributes`

- Classification: **KEEP**
- Reason: Project metadata, licensing, packaging, or contributor infrastructure remains required.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `.github/pull_request_template.md`

- Classification: **KEEP**
- Reason: Repository automation retained and updated for the canonical workflow.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `.github/workflows/ci.yml`

- Classification: **KEEP**
- Reason: Repository automation retained and updated for the canonical workflow.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `.github/workflows/validate-results.yml`

- Classification: **KEEP**
- Reason: Repository automation retained and updated for the canonical workflow.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `.gitignore`

- Classification: **KEEP**
- Reason: Project metadata, licensing, packaging, or contributor infrastructure remains required.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `.pre-commit-config.yaml`

- Classification: **KEEP**
- Reason: Project metadata, licensing, packaging, or contributor infrastructure remains required.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `CITATION.cff`

- Classification: **KEEP**
- Reason: Project metadata, licensing, packaging, or contributor infrastructure remains required.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `CONTRIBUTING.md`

- Classification: **KEEP**
- Reason: Project metadata, licensing, packaging, or contributor infrastructure remains required.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `LICENSE`

- Classification: **KEEP**
- Reason: Project metadata, licensing, packaging, or contributor infrastructure remains required.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `LICENSES.md`

- Classification: **KEEP**
- Reason: Project metadata, licensing, packaging, or contributor infrastructure remains required.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `Makefile`

- Classification: **KEEP**
- Reason: Project metadata, licensing, packaging, or contributor infrastructure remains required.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `README.md`

- Classification: **KEEP**
- Reason: Project metadata, licensing, packaging, or contributor infrastructure remains required.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `REPOSITORY_MANIFEST.txt`

- Classification: **DELETE**
- Reason: A path-only snapshot did not document content, references, or cleanup decisions.
- Replacement: docs/REPOSITORY_CLEANUP_MANIFEST.md
- References that must be updated: All live references removed or redirected.

### `benchmark_data/README.md`

- Classification: **KEEP**
- Reason: Small tracked reference data, not a generated experiment artifact.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `benchmark_data/published_visdrone_results.csv`

- Classification: **KEEP**
- Reason: Small tracked reference data, not a generated experiment artifact.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `configs/common/dataset_10class.yaml`

- Classification: **KEEP**
- Reason: Model or dataset configuration required by adapters and controlled execution.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `configs/common/dataset_2class.yaml`

- Classification: **KEEP**
- Reason: Model or dataset configuration required by adapters and controlled execution.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `configs/common/evaluation_defaults.yaml`

- Classification: **KEEP**
- Reason: Model or dataset configuration required by adapters and controlled execution.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `configs/common/training_defaults.yaml`

- Classification: **KEEP**
- Reason: Model or dataset configuration required by adapters and controlled execution.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `configs/faster_rcnn_resnet50/10class.yaml`

- Classification: **KEEP**
- Reason: Model or dataset configuration required by adapters and controlled execution.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `configs/faster_rcnn_resnet50/2class.yaml`

- Classification: **KEEP**
- Reason: Model or dataset configuration required by adapters and controlled execution.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `configs/faster_rcnn_resnet50/model.yaml`

- Classification: **KEEP**
- Reason: Model or dataset configuration required by adapters and controlled execution.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `configs/faster_rcnn_swin_t/10class.yaml`

- Classification: **KEEP**
- Reason: Model or dataset configuration required by adapters and controlled execution.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `configs/faster_rcnn_swin_t/2class.yaml`

- Classification: **KEEP**
- Reason: Model or dataset configuration required by adapters and controlled execution.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `configs/faster_rcnn_swin_t/model.yaml`

- Classification: **KEEP**
- Reason: Model or dataset configuration required by adapters and controlled execution.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `configs/faster_rcnn_vmamba_t/10class.yaml`

- Classification: **KEEP**
- Reason: Model or dataset configuration required by adapters and controlled execution.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `configs/faster_rcnn_vmamba_t/2class.yaml`

- Classification: **KEEP**
- Reason: Model or dataset configuration required by adapters and controlled execution.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `configs/faster_rcnn_vmamba_t/UPSTREAM_CONFIG.md`

- Classification: **KEEP**
- Reason: Model or dataset configuration required by adapters and controlled execution.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `configs/faster_rcnn_vmamba_t/model.yaml`

- Classification: **KEEP**
- Reason: Model or dataset configuration required by adapters and controlled execution.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `configs/rtdetrv2_l/10class.yaml`

- Classification: **KEEP**
- Reason: Model or dataset configuration required by adapters and controlled execution.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `configs/rtdetrv2_l/2class.yaml`

- Classification: **KEEP**
- Reason: Model or dataset configuration required by adapters and controlled execution.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `configs/rtdetrv2_l/model.yaml`

- Classification: **KEEP**
- Reason: Model or dataset configuration required by adapters and controlled execution.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `configs/yolox_s/10class.yaml`

- Classification: **KEEP**
- Reason: Model or dataset configuration required by adapters and controlled execution.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `configs/yolox_s/2class.yaml`

- Classification: **KEEP**
- Reason: Model or dataset configuration required by adapters and controlled execution.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `configs/yolox_s/model.yaml`

- Classification: **KEEP**
- Reason: Model or dataset configuration required by adapters and controlled execution.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `docs/RUN_SEARCH_FINETUNE_AND_UPLOAD.md`

- Classification: **MERGE**
- Reason: Content duplicated the student run, environment, method, or publishing guidance.
- Replacement: docs/RUN.md
- References that must be updated: README, CONTRIBUTING, PR template, and internal documentation links updated.

### `docs/architecture_notes.md`

- Classification: **MOVE**
- Reason: Useful content remains valid but is no longer part of the required run sequence.
- Replacement: docs/reference/architecture_notes.md
- References that must be updated: Primary README/run-guide links updated or historical references isolated.

### `docs/checkpoint_format.md`

- Classification: **MOVE**
- Reason: Useful content remains valid but is no longer part of the required run sequence.
- Replacement: docs/reference/checkpoint_format.md
- References that must be updated: Primary README/run-guide links updated or historical references isolated.

### `docs/colab_runbook.md`

- Classification: **MERGE**
- Reason: Content duplicated the student run, environment, method, or publishing guidance.
- Replacement: docs/RUN.md and docs/ENVIRONMENTS.md
- References that must be updated: README, CONTRIBUTING, PR template, and internal documentation links updated.

### `docs/evaluation_protocol.md`

- Classification: **MERGE**
- Reason: Content duplicated the student run, environment, method, or publishing guidance.
- Replacement: docs/METHODOLOGY.md
- References that must be updated: README, CONTRIBUTING, PR template, and internal documentation links updated.

### `docs/experiment_protocol.md`

- Classification: **MERGE**
- Reason: Content duplicated the student run, environment, method, or publishing guidance.
- Replacement: docs/METHODOLOGY.md
- References that must be updated: README, CONTRIBUTING, PR template, and internal documentation links updated.

### `docs/github_colab_workflow.md`

- Classification: **MERGE**
- Reason: Content duplicated the student run, environment, method, or publishing guidance.
- Replacement: docs/RUN.md and docs/RESULTS.md
- References that must be updated: README, CONTRIBUTING, PR template, and internal documentation links updated.

### `docs/lr_controlled_benchmark_audit.md`

- Classification: **MOVE**
- Reason: Useful content remains valid but is no longer part of the required run sequence.
- Replacement: docs/reference/lr_controlled_benchmark_audit.md
- References that must be updated: Primary README/run-guide links updated or historical references isolated.

### `docs/result_storage_policy.md`

- Classification: **MERGE**
- Reason: Content duplicated the student run, environment, method, or publishing guidance.
- Replacement: docs/RESULTS.md
- References that must be updated: README, CONTRIBUTING, PR template, and internal documentation links updated.

### `notebooks/00_colab_repository_setup.ipynb`

- Classification: **MERGE**
- Reason: Useful cells were replaced by thin canonical notebook cells calling reusable workflows.
- Replacement: notebooks/00_prepare_visdrone.ipynb
- References that must be updated: README, run guide, status messages, tests, and notebook-to-notebook links updated.

### `notebooks/00_environment_and_data_setup.ipynb`

- Classification: **MERGE**
- Reason: Useful cells were replaced by thin canonical notebook cells calling reusable workflows.
- Replacement: notebooks/00_prepare_visdrone.ipynb
- References that must be updated: README, run guide, status messages, tests, and notebook-to-notebook links updated.

### `notebooks/00_visdrone_colab_setup.ipynb`

- Classification: **MERGE**
- Reason: Useful cells were replaced by thin canonical notebook cells calling reusable workflows.
- Replacement: notebooks/00_prepare_visdrone.ipynb
- References that must be updated: README, run guide, status messages, tests, and notebook-to-notebook links updated.

### `notebooks/01_dataset_analysis.ipynb`

- Classification: **MOVE**
- Reason: Useful content remains valid but is no longer part of the required run sequence.
- Replacement: notebooks/optional/dataset_analysis.ipynb
- References that must be updated: Primary README/run-guide links updated or historical references isolated.

### `notebooks/02_train_resnet50_faster_rcnn.ipynb`

- Classification: **MERGE**
- Reason: Useful cells were replaced by thin canonical notebook cells calling reusable workflows.
- Replacement: notebooks/01_run_model_day.ipynb
- References that must be updated: README, run guide, status messages, tests, and notebook-to-notebook links updated.

### `notebooks/03_train_swin_t_faster_rcnn.ipynb`

- Classification: **MERGE**
- Reason: Useful cells were replaced by thin canonical notebook cells calling reusable workflows.
- Replacement: notebooks/01_run_model_day.ipynb
- References that must be updated: README, run guide, status messages, tests, and notebook-to-notebook links updated.

### `notebooks/04_train_vmamba_t_faster_rcnn.ipynb`

- Classification: **MERGE**
- Reason: Useful cells were replaced by thin canonical notebook cells calling reusable workflows.
- Replacement: notebooks/01_run_model_day.ipynb
- References that must be updated: README, run guide, status messages, tests, and notebook-to-notebook links updated.

### `notebooks/05_train_rtdetrv2_l.ipynb`

- Classification: **MERGE**
- Reason: Useful cells were replaced by thin canonical notebook cells calling reusable workflows.
- Replacement: notebooks/01_run_model_day.ipynb
- References that must be updated: README, run guide, status messages, tests, and notebook-to-notebook links updated.

### `notebooks/06_train_yolox_s_optional.ipynb`

- Classification: **DELETE**
- Reason: YOLOX is outside the controlled four-model benchmark; its low-level adapter/config remain available for non-benchmark research.
- Replacement: None
- References that must be updated: All live references removed or redirected.

### `notebooks/07_evaluate_all_models.ipynb`

- Classification: **MERGE**
- Reason: Useful cells were replaced by thin canonical notebook cells calling reusable workflows.
- Replacement: notebooks/01_run_model_day.ipynb
- References that must be updated: README, run guide, status messages, tests, and notebook-to-notebook links updated.

### `notebooks/08_architecture_visualization.ipynb`

- Classification: **MOVE**
- Reason: Useful content remains valid but is no longer part of the required run sequence.
- Replacement: notebooks/optional/architecture_visualization.ipynb
- References that must be updated: Primary README/run-guide links updated or historical references isolated.

### `notebooks/09_error_analysis.ipynb`

- Classification: **MOVE**
- Reason: Useful content remains valid but is no longer part of the required run sequence.
- Replacement: notebooks/optional/error_analysis.ipynb
- References that must be updated: Primary README/run-guide links updated or historical references isolated.

### `notebooks/10_generate_final_report.ipynb`

- Classification: **MERGE**
- Reason: Useful cells were replaced by thin canonical notebook cells calling reusable workflows.
- Replacement: notebooks/01_run_model_day.ipynb
- References that must be updated: README, run guide, status messages, tests, and notebook-to-notebook links updated.

### `notebooks/11_sync_results_to_github.ipynb`

- Classification: **MERGE**
- Reason: Useful cells were replaced by thin canonical notebook cells calling reusable workflows.
- Replacement: notebooks/02_publish_results.ipynb
- References that must be updated: README, run guide, status messages, tests, and notebook-to-notebook links updated.

### `notebooks/12_learning_rate_search.ipynb`

- Classification: **MERGE**
- Reason: Useful cells were replaced by thin canonical notebook cells calling reusable workflows.
- Replacement: notebooks/01_run_model_day.ipynb
- References that must be updated: README, run guide, status messages, tests, and notebook-to-notebook links updated.

### `notebooks/13_full_dataset_finetune.ipynb`

- Classification: **MERGE**
- Reason: Useful cells were replaced by thin canonical notebook cells calling reusable workflows.
- Replacement: notebooks/01_run_model_day.ipynb
- References that must be updated: README, run guide, status messages, tests, and notebook-to-notebook links updated.

### `project_config.yaml`

- Classification: **KEEP**
- Reason: Project metadata, licensing, packaging, or contributor infrastructure remains required.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `pyproject.toml`

- Classification: **KEEP**
- Reason: Project metadata, licensing, packaging, or contributor infrastructure remains required.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `reports/colab_compatibility_matrix.md`

- Classification: **MOVE**
- Reason: Useful content remains valid but is no longer part of the required run sequence.
- Replacement: docs/reference/colab_compatibility_matrix.md
- References that must be updated: Primary README/run-guide links updated or historical references isolated.

### `reports/notebook_blocker_audit.md`

- Classification: **MOVE**
- Reason: Useful content remains valid but is no longer part of the required run sequence.
- Replacement: docs/reference/notebook_blocker_audit.md
- References that must be updated: Primary README/run-guide links updated or historical references isolated.

### `reports/notebook_smoke_results.json`

- Classification: **MOVE**
- Reason: Useful content remains valid but is no longer part of the required run sequence.
- Replacement: docs/reference/notebook_smoke_results.json
- References that must be updated: Primary README/run-guide links updated or historical references isolated.

### `requirements-colab.txt`

- Classification: **KEEP**
- Reason: Pinned dependency boundary retained for its distinct dataset/model/test runtime.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `requirements-dataset-colab.txt`

- Classification: **KEEP**
- Reason: Pinned dependency boundary retained for its distinct dataset/model/test runtime.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `requirements-notebook-test.txt`

- Classification: **KEEP**
- Reason: Pinned dependency boundary retained for its distinct dataset/model/test runtime.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `requirements-openmmlab-py310-cu118.txt`

- Classification: **KEEP**
- Reason: Pinned dependency boundary retained for its distinct dataset/model/test runtime.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `requirements-rtdetr-colab.txt`

- Classification: **KEEP**
- Reason: Pinned dependency boundary retained for its distinct dataset/model/test runtime.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `requirements.txt`

- Classification: **KEEP**
- Reason: Pinned dependency boundary retained for its distinct dataset/model/test runtime.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `results/.gitkeep`

- Classification: **KEEP**
- Reason: Approved lightweight-result scaffold; runtime artifacts remain excluded.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `results/README.md`

- Classification: **KEEP**
- Reason: Approved lightweight-result scaffold; runtime artifacts remain excluded.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `scripts/__init__.py`

- Classification: **KEEP**
- Reason: Low-level execution/validation entry point retained behind the unified CLI.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `scripts/build_colab_notebooks.py`

- Classification: **DELETE**
- Reason: The monolithic generator duplicated notebook logic and caused drift; canonical .ipynb files now call normal Python modules.
- Replacement: Canonical notebooks plus src/workflows/
- References that must be updated: All live references removed or redirected.

### `scripts/build_registry.py`

- Classification: **KEEP**
- Reason: Low-level execution/validation entry point retained behind the unified CLI.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `scripts/configure_git.sh`

- Classification: **MERGE**
- Reason: Compatibility/user wrapper was subsumed by the unified notebook/CLI workflow.
- Replacement: scripts/benchmark.py publish
- References that must be updated: README, docs, notebooks, and tests updated.

### `scripts/create_results_manifest.py`

- Classification: **KEEP**
- Reason: Low-level execution/validation entry point retained behind the unified CLI.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `scripts/evaluate.py`

- Classification: **KEEP**
- Reason: Low-level execution/validation entry point retained behind the unified CLI.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `scripts/export_onnx.py`

- Classification: **KEEP**
- Reason: Low-level execution/validation entry point retained behind the unified CLI.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `scripts/export_results.py`

- Classification: **MERGE**
- Reason: Compatibility/user wrapper was subsumed by the unified notebook/CLI workflow.
- Replacement: src/result_export.py and scripts/benchmark.py publish
- References that must be updated: README, docs, notebooks, and tests updated.

### `scripts/full_dataset_finetune.py`

- Classification: **KEEP**
- Reason: Low-level execution/validation entry point retained behind the unified CLI.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `scripts/generate_report.py`

- Classification: **KEEP**
- Reason: Low-level execution/validation entry point retained behind the unified CLI.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `scripts/lr_search.py`

- Classification: **KEEP**
- Reason: Low-level execution/validation entry point retained behind the unified CLI.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `scripts/next_benchmark_step.py`

- Classification: **MERGE**
- Reason: Compatibility/user wrapper was subsumed by the unified notebook/CLI workflow.
- Replacement: scripts/benchmark.py next
- References that must be updated: README, docs, notebooks, and tests updated.

### `scripts/prepare_data.py`

- Classification: **KEEP**
- Reason: Low-level execution/validation entry point retained behind the unified CLI.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `scripts/profile_model.py`

- Classification: **KEEP**
- Reason: Low-level execution/validation entry point retained behind the unified CLI.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `scripts/pull_latest.sh`

- Classification: **MERGE**
- Reason: Compatibility/user wrapper was subsumed by the unified notebook/CLI workflow.
- Replacement: notebook bootstrap cells
- References that must be updated: README, docs, notebooks, and tests updated.

### `scripts/run_mmdetection.py`

- Classification: **KEEP**
- Reason: Low-level execution/validation entry point retained behind the unified CLI.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `scripts/run_notebook_smoke.py`

- Classification: **KEEP**
- Reason: Low-level execution/validation entry point retained behind the unified CLI.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `scripts/run_rtdetr_training.py`

- Classification: **KEEP**
- Reason: Low-level execution/validation entry point retained behind the unified CLI.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `scripts/setup_colab.sh`

- Classification: **MERGE**
- Reason: Compatibility/user wrapper was subsumed by the unified notebook/CLI workflow.
- Replacement: notebooks/00_prepare_visdrone.ipynb
- References that must be updated: README, docs, notebooks, and tests updated.

### `scripts/show_benchmark_status.py`

- Classification: **MERGE**
- Reason: Compatibility/user wrapper was subsumed by the unified notebook/CLI workflow.
- Replacement: scripts/benchmark.py status
- References that must be updated: README, docs, notebooks, and tests updated.

### `scripts/sync_results_to_repo.py`

- Classification: **KEEP**
- Reason: Low-level execution/validation entry point retained behind the unified CLI.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `scripts/train.py`

- Classification: **KEEP**
- Reason: Low-level execution/validation entry point retained behind the unified CLI.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `scripts/tune.py`

- Classification: **DELETE**
- Reason: Legacy multidimensional tuning wrapper conflicts with LR-only search.
- Replacement: src/training/lr_search.py
- References that must be updated: All live references removed or redirected.

### `scripts/validate_results.py`

- Classification: **KEEP**
- Reason: Low-level execution/validation entry point retained behind the unified CLI.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `scripts/verify_model_environments.py`

- Classification: **KEEP**
- Reason: Low-level execution/validation entry point retained behind the unified CLI.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/__init__.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/benchmark_status.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/colab_setup.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/data/__init__.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/data/collapse_classes.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/data/convert_visdrone.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/data/dataloaders.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/data/download.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/data/smoke_dataset.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/data/statistics.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/data/validate_annotations.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/drive_sync.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/evaluation/__init__.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/evaluation/calibration.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/evaluation/coco_evaluator.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/evaluation/detection_metrics.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/evaluation/efficiency_metrics.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/evaluation/error_analysis.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/evaluation/mmdet_aerial_metric.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/evaluation/report_generator.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/evaluation/robustness.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/evaluation/visualization.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/git_utils.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/models/__init__.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/models/base_adapter.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/models/mmdetection_adapter.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/models/registry.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/models/rtdetr_adapter.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/models/yolox_adapter.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/notebook_utils.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/paths.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/reproducibility.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/result_export.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/training/__init__.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/training/callbacks.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/training/checkpointing.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/training/hyperparameter_search.py`

- Classification: **DELETE**
- Reason: Compatibility alias for the removed generic tuning path.
- Replacement: src/training/lr_search.py
- References that must be updated: All live references removed or redirected.

### `src/training/lr_range.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/training/lr_search.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/training/lr_workflow.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/training/profiling.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/training/recipes.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/training/trainer.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/utils/__init__.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/utils/environment.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/utils/logging.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `src/utils/serialization.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `tests/test_annotation_conversion.py`

- Classification: **KEEP**
- Reason: Automated contract, conversion, evaluation, registry, or export coverage retained.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `tests/test_benchmark_status.py`

- Classification: **KEEP**
- Reason: Automated contract, conversion, evaluation, registry, or export coverage retained.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `tests/test_checkpoint_registry.py`

- Classification: **KEEP**
- Reason: Automated contract, conversion, evaluation, registry, or export coverage retained.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `tests/test_class_mapping.py`

- Classification: **KEEP**
- Reason: Automated contract, conversion, evaluation, registry, or export coverage retained.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `tests/test_dataset_paths_and_download.py`

- Classification: **KEEP**
- Reason: Automated contract, conversion, evaluation, registry, or export coverage retained.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `tests/test_lr_search.py`

- Classification: **KEEP**
- Reason: Automated contract, conversion, evaluation, registry, or export coverage retained.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `tests/test_metric_calculation.py`

- Classification: **KEEP**
- Reason: Automated contract, conversion, evaluation, registry, or export coverage retained.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `tests/test_model_adapters.py`

- Classification: **KEEP**
- Reason: Automated contract, conversion, evaluation, registry, or export coverage retained.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `tests/test_result_export_workflow.py`

- Classification: **KEEP**
- Reason: Automated contract, conversion, evaluation, registry, or export coverage retained.
- Replacement: —
- References that must be updated: References retained or updated in place.

### `tests/test_workflow_utils.py`

- Classification: **KEEP**
- Reason: Automated contract, conversion, evaluation, registry, or export coverage retained.
- Replacement: —
- References that must be updated: References retained or updated in place.

## Post-cleanup canonical additions

### `docs/ENVIRONMENTS.md`

- Classification: **KEEP**
- Reason: Canonical current documentation.
- Replacement: —
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

### `docs/METHODOLOGY.md`

- Classification: **KEEP**
- Reason: Canonical current documentation.
- Replacement: —
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

### `docs/REPOSITORY_CLEANUP_MANIFEST.md`

- Classification: **KEEP**
- Reason: Canonical current documentation.
- Replacement: —
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

### `docs/RESULTS.md`

- Classification: **KEEP**
- Reason: Canonical current documentation.
- Replacement: —
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

### `docs/RUN.md`

- Classification: **KEEP**
- Reason: Canonical current documentation.
- Replacement: —
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

### `docs/reference/architecture_notes.md`

- Classification: **MOVE**
- Reason: Moved from `docs/architecture_notes.md` and removed from the required sequence.
- Replacement: docs/reference/architecture_notes.md
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

### `docs/reference/checkpoint_format.md`

- Classification: **MOVE**
- Reason: Moved from `docs/checkpoint_format.md` and removed from the required sequence.
- Replacement: docs/reference/checkpoint_format.md
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

### `docs/reference/colab_compatibility_matrix.md`

- Classification: **MOVE**
- Reason: Moved from `reports/colab_compatibility_matrix.md` and removed from the required sequence.
- Replacement: docs/reference/colab_compatibility_matrix.md
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

### `docs/reference/lr_controlled_benchmark_audit.md`

- Classification: **MOVE**
- Reason: Moved from `docs/lr_controlled_benchmark_audit.md` and removed from the required sequence.
- Replacement: docs/reference/lr_controlled_benchmark_audit.md
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

### `docs/reference/notebook_blocker_audit.md`

- Classification: **MOVE**
- Reason: Moved from `reports/notebook_blocker_audit.md` and removed from the required sequence.
- Replacement: docs/reference/notebook_blocker_audit.md
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

### `docs/reference/notebook_smoke_results.json`

- Classification: **MOVE**
- Reason: Moved from `reports/notebook_smoke_results.json` and removed from the required sequence.
- Replacement: docs/reference/notebook_smoke_results.json
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

### `notebooks/00_prepare_visdrone.ipynb`

- Classification: **KEEP**
- Reason: Canonical notebook-first user surface or optional exploration.
- Replacement: —
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

### `notebooks/01_run_model_day.ipynb`

- Classification: **KEEP**
- Reason: Canonical notebook-first user surface or optional exploration.
- Replacement: —
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

### `notebooks/02_publish_results.ipynb`

- Classification: **KEEP**
- Reason: Canonical notebook-first user surface or optional exploration.
- Replacement: —
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

### `notebooks/03_compare_all_models.ipynb`

- Classification: **KEEP**
- Reason: Canonical notebook-first user surface or optional exploration.
- Replacement: —
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

### `notebooks/optional/architecture_visualization.ipynb`

- Classification: **MOVE**
- Reason: Moved from `notebooks/08_architecture_visualization.ipynb` and removed from the required sequence.
- Replacement: notebooks/optional/architecture_visualization.ipynb
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

### `notebooks/optional/dataset_analysis.ipynb`

- Classification: **MOVE**
- Reason: Moved from `notebooks/01_dataset_analysis.ipynb` and removed from the required sequence.
- Replacement: notebooks/optional/dataset_analysis.ipynb
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

### `notebooks/optional/error_analysis.ipynb`

- Classification: **MOVE**
- Reason: Moved from `notebooks/09_error_analysis.ipynb` and removed from the required sequence.
- Replacement: notebooks/optional/error_analysis.ipynb
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

### `scripts/benchmark.py`

- Classification: **KEEP**
- Reason: Low-level execution/validation entry point retained behind the unified CLI.
- Replacement: —
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

### `scripts/clean_notebooks.py`

- Classification: **KEEP**
- Reason: Low-level execution/validation entry point retained behind the unified CLI.
- Replacement: —
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

### `scripts/scan_repository_secrets.py`

- Classification: **KEEP**
- Reason: Low-level execution/validation entry point retained behind the unified CLI.
- Replacement: —
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

### `scripts/validate_doc_links.py`

- Classification: **KEEP**
- Reason: Low-level execution/validation entry point retained behind the unified CLI.
- Replacement: —
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

### `scripts/validate_notebooks.py`

- Classification: **KEEP**
- Reason: Low-level execution/validation entry point retained behind the unified CLI.
- Replacement: —
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

### `src/workflows/__init__.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

### `src/workflows/comparison.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

### `src/workflows/contract.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

### `src/workflows/dataset_setup.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

### `src/workflows/environment.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

### `src/workflows/model_day.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

### `src/workflows/publishing.py`

- Classification: **KEEP**
- Reason: Reusable, single-responsibility implementation retained and covered by tests.
- Replacement: —
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

### `tests/test_comparison_workflow.py`

- Classification: **KEEP**
- Reason: Automated contract, conversion, evaluation, registry, or export coverage retained.
- Replacement: —
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

### `tests/test_notebook_first_workflow.py`

- Classification: **KEEP**
- Reason: Automated contract, conversion, evaluation, registry, or export coverage retained.
- Replacement: —
- References that must be updated: Resolved by the new README, run guide, tests, and CI.

## Later removals

The decisions above were made against the `e104f29` baseline and are kept as a
record. Two later rounds removed more; entries above naming a since-deleted file
as a `Replacement` should be read against this section.

### Retired `lr_controlled_v1` protocol

`lr_controlled_v1` stopped being runnable at PR-11 but kept shipping. Deleted:
`src/workflows/model_day.py`, `src/training/lr_workflow.py`,
`src/workflows/notebook_entrypoints.py`, `scripts/benchmark.py`,
`scripts/lr_search.py`, `scripts/full_dataset_finetune.py`, and notebooks
`00_bootstrap_colab`, `01_run_model_day`, `02_publish_results`,
`03_compare_all_models`.

Live pieces moved rather than died: `LRControlledBenchmark.prepare_manifests` →
`src/training/lr_search.py::ensure_lr_search_manifests`; `model_day._run_module`
→ `src/subprocess_utils.py::run_module_in_model_runtime`; `scripts.benchmark
publish` → `scripts/publish_results.py`.

### Twelve operator runs collapsed to five

One notebook per model plus one report replaced the twelve-launch flow. Deleted:
`00_prepare_visdrone`, `10_hpo_*`–`13_hpo_*`, `20_finetune_*`–`23_finetune_*`,
`30_evaluate_all_models`, `31_publish_results`. Added: `10_resnet50`,
`11_swin_t`, `12_vmamba_t`, `13_rtdetrv2`, `30_report`, backed by
`src/workflows/model_pipeline.py`, `evaluation_runner.py`, `reporting.py`, and
`adapter_smoke.py`. `07_performance_tiling` moved to `notebooks/optional/`.
Notebook 00's dataset knobs survive as `scripts/prepare_dataset.py`.

## Generated artifacts policy

Datasets, archives, checkpoints, optimizer/scheduler states, raw predictions, TensorBoard logs, compiled CUDA extensions, cloned framework trees, credentials, and generated experimental reports are classified **GENERATED** and excluded by `.gitignore`. Only validated lightweight bundles under `results/bundles/` may be tracked.
