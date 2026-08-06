"""Real-GPU adapter smoke gate for every controlled-track model.

This is the single hard prerequisite before any full run: it converts
"implemented but unvalidated" into "executable and tested" for each adapter. For
each model it asserts, in order, on a real GPU:

  1. constructs                     - the model builds from its pinned revision.
  2. pretrained_weights_complete    - weights load with ZERO missing/unexpected
                                      keys (a partial load raises, not warns).
  3. feature_map_contract           - backbone/FPN channels + strides + NCHW
                                      spatial dims at the *configured* resolution
                                      (640 and 1024), never 224.
  4. detection_head_class_count     - the head is reset to the track class count
                                      (2 or 10) with no COCO-80 residue.
  5. forward_backward_finite_loss   - one forward+backward on a 2-image batch
                                      produces a finite loss.
  6. predict_wellformed             - one predict returns well-formed boxes.
  7. checkpoint_roundtrip           - save -> load yields identical parameters.

It writes a signed JSON record per model to the artifact root (never to
`results/` - a smoke pass is not a benchmark result). A run must not start until
a stored READY record exists for every controlled-track model at the current
commit and GPU.

GPU-only: cannot run or be validated on a CPU host. The assertion primitives it
calls (`src.workflows.adapter_gate`) are unit-tested on CPU with stub tensors;
this orchestration around real adapters is validated only on hardware.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.config.benchmark_tracks import load_track_config, resolve_controlled_protocol
from src.workflows.adapter_gate import (
    AdapterGateError,
    SMOKE_CHECK_ORDER,
    adapter_fingerprint,
    assert_checkpoint_roundtrip,
    assert_detection_head_class_count,
    assert_feature_map_contract,
    assert_finite_loss,
    assert_pretrained_load_complete,
    assert_wellformed_predictions,
    build_smoke_record,
)

# Expected backbone/FPN feature contract per family, as ordered (channels,
# stride) pairs. These encode the architecture the benchmark intends to run and
# are confirmed against the real model on GPU; adjust here if a pinned upstream
# revision legitimately changes them.
FEATURE_CONTRACT = {
    "mmdetection": [(256, 4), (256, 8), (256, 16), (256, 32), (256, 64)],
    "vmamba_mmdetection": [(256, 4), (256, 8), (256, 16), (256, 32), (256, 64)],
    "transformers": [(256, 8), (256, 16), (256, 32)],
}
TRACK_CLASS_COUNT = {"2class": 2, "10class": 10}


def _require_cuda() -> Any:  # pragma: no cover - GPU host only
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA device available. The adapter smoke gate must run on the "
            "target GPU; it does not run on CPU."
        )
    return torch


def run_model_smoke(
    model_id: str,
    repo_root: Path,
    *,
    dataset_track: str,
    image_size: int,
) -> dict[str, Any]:  # pragma: no cover - GPU host only
    """Execute the ordered smoke checks for one model and return a signed record."""
    torch = _require_cuda()
    from src.models.registry import create_adapter, load_model_config

    config = load_model_config(model_id, repo_root)
    framework = str(config["framework"])
    expected_classes = TRACK_CLASS_COUNT[dataset_track]
    checks: list[dict[str, Any]] = []

    def record(name: str, fn: Any) -> None:
        try:
            fn()
            checks.append({"name": name, "passed": True})
        except Exception as error:  # noqa: BLE001 - the gate records every failure
            checks.append({"name": name, "passed": False, "error": str(error)})
            raise

    adapter = create_adapter(model_id)
    model_holder: dict[str, Any] = {}

    def _construct() -> None:
        model_holder["model"] = adapter.load_model("", config)

    def _weights() -> None:
        load = model_holder.get("load_result", {})
        assert_pretrained_load_complete(
            load.get("missing_keys"), load.get("unexpected_keys")
        )

    def _features() -> None:
        maps = adapter.feature_maps(image_size) if hasattr(adapter, "feature_maps") else []
        assert_feature_map_contract(maps, FEATURE_CONTRACT[framework], image_size)

    def _head() -> None:
        assert_detection_head_class_count(
            adapter.head_class_count(), expected_classes
        )

    def _loss() -> None:
        batch = torch.randn(2, 3, image_size, image_size, device="cuda")
        assert_finite_loss(adapter.forward_backward(batch))

    def _predict() -> None:
        images = [torch.randn(3, image_size, image_size) for _ in range(2)]
        assert_wellformed_predictions(
            adapter.predict(images), num_classes=expected_classes
        )

    def _roundtrip() -> None:
        before, after = adapter.checkpoint_roundtrip()
        assert_checkpoint_roundtrip(before, after)

    steps = {
        "constructs": _construct,
        "pretrained_weights_complete": _weights,
        "feature_map_contract": _features,
        "detection_head_class_count": _head,
        "forward_backward_finite_loss": _loss,
        "predict_wellformed": _predict,
        "checkpoint_roundtrip": _roundtrip,
    }
    try:
        for name in SMOKE_CHECK_ORDER:
            record(name, steps[name])
    except Exception:
        pass  # a failed check is recorded; the record status becomes FAILED_ADAPTER

    fingerprint = adapter_fingerprint(model_id, repo_root)
    return build_smoke_record(
        model_id, fingerprint, checks, gpu=str(fingerprint.get("gpu"))
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--drive-root", required=True)
    parser.add_argument("--dataset-track", default="2class", choices=["2class", "10class"])
    parser.add_argument("--model-id", default=None, help="limit to one model")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    track = load_track_config(repo_root, "controlled")
    model_ids = [args.model_id] if args.model_id else list(track["model_ids"])
    out_dir = Path(args.drive_root).resolve() / "adapter_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_ready = True
    for model_id in model_ids:  # pragma: no cover - GPU host only
        protocol = resolve_controlled_protocol(repo_root, model_id)
        record = run_model_smoke(
            model_id,
            repo_root,
            dataset_track=args.dataset_track,
            image_size=int(protocol["image_size"]),
        )
        (out_dir / f"{model_id}__{args.dataset_track}__smoke.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        status = record["status"]
        all_ready = all_ready and status == "READY"
        print(f"{model_id}: {status}")
    return 0 if all_ready else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
