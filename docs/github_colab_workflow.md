# GitHub → Colab → Drive → GitHub workflow

## Initial upload

Inspect the worktree, then choose public or private repository visibility:

```bash
git status
git init
git branch -M main
git add README.md LICENSE LICENSES.md CITATION.cff
git add project_config.yaml pyproject.toml
git add src/ scripts/ configs/ notebooks/ tests/ docs/
git add requirements.txt requirements-dataset-colab.txt
git add .gitignore .gitattributes .github/
git diff --cached --stat
git commit -m "chore: initialize VisDrone architecture benchmark"
git remote add origin https://github.com/Harryphan72007/aerial-object-detection-benchmark.git
git push -u origin main
```

Or, after choosing visibility in GitHub CLI:

```bash
gh repo create aerial-object-detection-benchmark --public --source=. --remote=origin --push
```

## Colab setup and training

Open `00_visdrone_colab_setup.ipynb` in Colab, review its configuration cell,
mount Drive, and run top to bottom. It downloads or restores validated archives
under `DRIVE_ROOT/datasets/VisDrone2019-DET/archives`, preserves extracted data
under `raw`, and produces both COCO tracks under `processed`. The two older
setup notebooks are compatibility preflights.

Notebooks 02–05 retain model-specific environment and adapter checks. Notebook
12 owns the shared LR-only protocol: 2-class data, 640-pixel input, seed 42,
AMP, effective batch 8, nine log-spaced candidates, and resumable 2/5/10/15
epoch rungs. Notebook 13 reloads the original pretrained checkpoint, trains on
complete official train for 25 epochs, and invokes the common evaluator once on
complete official validation. Search checkpoints stay outside the final run
registry. Training stops before starting when Drive is unavailable or unwritable.

Each run records `git_commit.txt`, `git_status.txt`, `run_manifest.json`, and (when dirty) `source_changes.patch`. A dirty source tree is marked in the manifest; it does not silently become an untraceable experiment.

## Evaluation and result bundles

Run `07_evaluate_all_models.ipynb` or `scripts/evaluate.py`. Evaluation discovers completed checkpoints through the existing registry, filters by track/model/resolution/seed, verifies class mappings, and writes raw predictions and complete metrics to Drive. A model failure is recorded and compatible models continue where safe.

Create a versioned per-model bundle with `scripts/create_results_manifest.py`.
Never mix models, runs, seeds, or dataset tracks in one bundle. The bundle
records the run identity, selected LR, resolved final configuration, search
history, checkpoint and annotation hashes, training/evaluation commits,
environment, official dataset proof, measured metrics, generated files,
exclusions, and single-seed status.

## Safe result synchronization

Open notebook `11_sync_results_to_github.ipynb`, list bundles, run the exporter with `--dry-run`, inspect copied destinations and exclusions, then run it for real. Inspect `git status --short`, `git diff --stat`, and `git diff -- results/`. Configure repository identity locally. Use Colab Secrets for a temporary `GITHUB_TOKEN` only during push; never print, save, or put it in Git config, notebook output, Drive, or manifests. A local computer is the safer alternative.

Use the separate `experiment-results` branch. Fetch and inspect remote changes
before updating it. Do not force-push by default. Stage only approved paths
(`git add -- results`), run validation, commit, and push. Create a PR only after
`gh auth status` succeeds; do not include private Drive URLs.

Interrupted pushes are safe to retry after inspecting `git status` and remote differences. If GitHub rejects a large file, remove it from staging/history as needed, add the appropriate ignore rule, and retain the artifact on Drive or an external release store. Do not use `git add .`.

## Notebook and provenance hygiene

Run `nbstripout` (included in `.pre-commit-config.yaml`) before commits. CI validates code and result pull requests but never trains models. Reports must state the bundle ID and whether results are single-seed or multi-seed. Results are not fabricated by this repository; empty Drive results remain empty until experiments run.

## Recovery and conflict handling

If Drive is unavailable or fails the write test, stop before training. A run's `last.pth` is the resume point after a runtime disconnect; rerun the same training notebook with its `RESUME_RUN_ID`. The checkpoint includes model, optimizer, scheduler, AMP scaler, epoch, best metrics, resolved arguments, run ID, and seed.

If the repository is dirty before a pull, the shared updater prints `git status --porcelain` and stops. During training, the original commit and `git_status.txt` remain in the run directory; dirty tracked changes are captured in `source_changes.patch`. Registry updates use a lock, a temporary file, `fsync`, atomic replacement, and a `.bak` copy.

If a remote results branch changed, fetch it and inspect the divergence before rebasing or merging. Never force-push by default. After an interrupted push, inspect local and remote status and retry the same branch push. If GitHub rejects a large file, remove it from staging/history as needed, add an ignore rule, and keep the artifact on Drive or publish a selected checkpoint through a Release, Hugging Face, or Zenodo.

Run `pre-commit install` once after installing development requirements. The configured `nbstripout` hook removes notebook outputs before commit.
