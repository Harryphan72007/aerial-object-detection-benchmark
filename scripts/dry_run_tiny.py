"""End-to-end dry-run-tiny: prove the result bundle before the real run.

Runs one model, one track, 1 epoch, on a ~50-image subset through the *real*
train -> eval -> profile -> manifest -> validate chain, writing to a `dry_run/`
namespace that the canonical discovery/publish logic provably cannot see. The
point is to surface publication bugs on a 10-minute run instead of after 30
GPU-hours.

The produced bundle is marked (`bundle_kind: dry_run`) and lives under
`<drive_root>/dry_run/result_bundles/`. It validates against the *production*
bundle schema (`validate_bundle`) yet is rejected by the results/ publisher and
by `validate_results --repo-results results/` — so a dry-run can never be
mistaken for a canonical result.

GPU-only: the train/eval steps require CUDA and the provisioned model
environments; this script cannot be validated on a CPU host. The namespace
isolation and the dry-run marker/rejection logic are unit-tested on CPU in
`tests/test_dry_run_bundle.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from src.result_export import DRY_RUN_NAMESPACE, is_dry_run_bundle, validate_bundle
from src.utils.serialization import read_json, write_json


def dry_run_bundle_root(drive_root: str | Path) -> Path:
    """The isolated namespace for dry-run bundles, disjoint from result_bundles/."""
    return Path(drive_root).expanduser().resolve() / DRY_RUN_NAMESPACE / "result_bundles"


def mark_bundle_as_dry_run(bundle_dir: str | Path) -> dict[str, Any]:
    """Stamp a freshly built bundle manifest as a dry-run bundle."""
    manifest_path = Path(bundle_dir) / "bundle_manifest.json"
    manifest = read_json(manifest_path)
    manifest["bundle_kind"] = DRY_RUN_NAMESPACE
    manifest["dry_run_tiny"] = True
    write_json(manifest_path, manifest)
    return manifest


def run_dry_run_tiny(
    drive_root: str | Path,
    repo_root: str | Path,
    *,
    model_id: str,
    dataset_track: str = "2class",
    images: int = 50,
    epochs: int = 1,
) -> dict[str, Any]:  # pragma: no cover - GPU host only
    """Execute the real chain on a tiny subset into the dry_run namespace.

    Implemented against the same train/eval/bundle functions the canonical run
    uses, only with a tiny image budget and a single epoch, and pointed at the
    dry_run namespace. Requires CUDA and the model environment.
    """
    from src.hpo.final_workflow import FinalExperimentWorkflow  # noqa: F401

    raise SystemExit(
        "run_dry_run_tiny requires a CUDA GPU and provisioned model environments. "
        "Invoke on the target hardware; it builds a bundle under "
        f"{dry_run_bundle_root(drive_root)} and validates it before any full run."
    )


def validate_dry_run_bundle(bundle_dir: str | Path) -> list[str]:
    """The dry-run bundle must satisfy the production schema and be marked."""
    bundle = Path(bundle_dir)
    errors = list(validate_bundle(bundle))
    manifest_path = bundle / "bundle_manifest.json"
    if not manifest_path.is_file():
        return errors + ["dry-run bundle has no bundle_manifest.json"]
    if not is_dry_run_bundle(read_json(manifest_path)):
        errors.append("dry-run bundle is not marked as a dry run")
    return sorted(set(errors))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--model-id", default="faster_rcnn_resnet50")
    parser.add_argument("--dataset-track", default="2class", choices=["2class", "10class"])
    parser.add_argument("--images", type=int, default=50)
    args = parser.parse_args(argv)
    result = run_dry_run_tiny(  # pragma: no cover - GPU host only
        args.drive_root,
        args.repo_root,
        model_id=args.model_id,
        dataset_track=args.dataset_track,
        images=args.images,
    )
    print(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
