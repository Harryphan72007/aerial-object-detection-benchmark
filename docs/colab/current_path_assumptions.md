# Current Colab and local path assumptions

Status: migration PR 2 inventory. Paths are documented, not centralized or
changed. Centralization belongs to PR 6.

## Configured defaults

| Identity | Current value | Source |
|---|---|---|
| Repository URL | `https://github.com/Harryphan72007/aerial-object-detection-benchmark.git` | notebook setup cells and `project_config.yaml` identity |
| Branch | `main` | notebook setup cells and `project_config.yaml` |
| Colab repository | `/content/aerial-object-detection-benchmark` | `project_config.yaml` and notebooks |
| Drive mount | `/content/drive` | notebook setup cells |
| Drive project root | `/content/drive/MyDrive/visdrone_architecture_benchmark` | `project_config.yaml`, `src.paths`, notebooks |
| Disposable dataset cache | `/content/visdrone_cache` | notebook 01 |
| Isolated model environments | `/content/visdrone_model_envs` | environment provisioner |
| Local 00-03 fallback | `<repository>/local_artifacts` | notebooks 00-03 |

`VISDRONE_DRIVE_ROOT` overrides the Drive root. Optional notebooks also accept
`BENCHMARK_REPO_ROOT`, `BENCHMARK_REPOSITORY_URL`, and
`BENCHMARK_REPOSITORY_BRANCH`. `VISDRONE_MODEL_PYTHON` and
`VISDRONE_RUNTIME_MANIFEST` identify an isolated model process after
provisioning. They are cleared before setup and published only after the hashed
runtime reaches transactional state `READY`. `FAILED`, `INSTALLING`, and
`VERIFYING` runtimes are rebuilt only under `/content/visdrone_model_envs`.

## Notebook-family behavior

| Notebooks | Colab clone/update | Dependencies | Local path behavior |
|---|---|---|---|
| 00-03 | clone `main`; existing clones require a clean tree and fast-forward pull | dataset/shared stack where needed | repository is `Path.cwd()`; Drive defaults to `local_artifacts` |
| 10-13 HPO | clone `main`; clean-tree check and fast-forward pull | HPO shared stack plus editable package | repository is `Path.cwd()`; Drive still defaults to the absolute Colab path unless overridden |
| 20-23 fine-tune | clone when absent; no equivalent update step in the notebook cell | editable package; workflow provisions family environment | repository is `Path.cwd()`; Drive still defaults to the absolute Colab path unless overridden |
| 30-31 evaluate/publish | no clone/bootstrap; assumes launch from repository | existing environment | `Path.cwd()` must be repository root; Drive must be overridden outside Colab |
| optional notebooks | configurable clone URL/branch; repository-root override | dataset/shared stack | `BENCHMARK_REPO_ROOT` or current directory; Drive defaults to Colab path |

## Implicit preconditions

1. Hosted Colab has mounted Drive at `/content/drive` before Drive artifacts are
   accessed.
2. Local executions start with the repository as the current working directory,
   except where an explicit repository-root override exists.
3. Git and network access are available for clone, fetch/pull, and pinned
   upstream repositories.
4. The canonical clone is either absent or a Git repository. Dirty clones are
   refused by notebooks that implement the clean-tree guard.
5. Branch `main` exists remotely and can be fast-forwarded.
6. The Drive root is writable and persistent; `/content` caches and isolated
   environments are disposable.
7. `sys.path` is modified to put the cloned repository first. An unrelated
   installed package with the same module name must not win import resolution.
8. Large artifacts remain under the Drive root. The Git checkout is source code
   only.
9. Model subprocesses receive their Python executable and runtime manifest via
   environment variables.

## Path inconsistencies frozen for later repair

The current notebooks do not resolve local paths identically. Notebooks 00-03
fall back to `<repo>/local_artifacts`, while HPO, fine-tuning, evaluation,
publication, and optional notebooks retain a `/content/drive/...` default even
outside Colab. Local users must set `VISDRONE_DRIVE_ROOT` for those notebooks.

Some notebooks bootstrap or safely update the clone, some only clone when
missing, and notebooks 30-31 assume an already-correct checkout. These are
documented risks; PR 2 does not rewrite setup cells.

## Diagnostic interpretation

Run:

```bash
python -m scripts.diagnostics.report_current_environment
```

Path entries report existence, directory type, and access-bit readability and
writability without creating probe files. `writable_by_access_check` is only an
OS access check; it is not proof that Google Drive writes are reliable. A real
Drive write/integrity test belongs to later Colab validation.

Training must not be inferred safe solely because paths exist. The diagnostic
does not mount Drive, clone or pull repositories, install dependencies, create
directories, construct models, or start training.
