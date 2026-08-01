# Experiment namespace guards

New artifact producers must build a `RunPathIdentity`, resolve the run root with
`ArtifactPathResolver`, and call `guard_write_target` before creating a directory
or opening a file. The guard is read-only: it resolves the proposed location and
rejects any path outside the exact run namespace.

The allowed track/mode matrix is:

| Track | Modes |
|---|---|
| `smoke` | `smoke` |
| `controlled` | `full` |
| `performance` | `full`, `sliced`, `ensemble` |

Consequently, smoke artifacts cannot enter full runs, controlled artifacts cannot
enter performance runs, and full-image predictions cannot enter sliced or
ensemble directories. Artifact kinds such as checkpoints and predictions are also
separate roots.

Legacy producers remain unchanged until their model-specific migration PR. Do not
point new producers at legacy directories to bypass the guard.
