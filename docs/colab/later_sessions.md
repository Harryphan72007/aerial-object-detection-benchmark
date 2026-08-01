# Later Colab sessions

Reopen the same GitHub bootstrap notebook. The checkout under `/content` is
disposable and may no longer exist after a runtime reset; cloning again is safe.

When a checkout exists, bootstrap refuses to update it if tracked or untracked
files make the tree dirty. A branch update must fast-forward to `origin/<branch>`.
A tag or commit is checked out detached. Diverged branches, missing references,
and existing non-Git directories fail visibly; bootstrap never resets, cleans,
stashes, or force-updates them.

If the disposable clone is dirty, inspect it only if the changes matter. The
safest recovery is normally to save intentional source changes through a GitHub
branch, then remove or replace the disposable runtime outside this notebook and
reclone. Never treat Drive artifacts as source-tree changes.

After bootstrap, confirm the selected commit and clean status before discovering
resumable runs. Full resume compatibility is introduced later in the migration;
until then, use existing legacy behavior and do not infer compatibility from a
matching filename alone.
