"""Real-GPU adapter smoke gate for every controlled-track model.

This is the single hard prerequisite before any full run: it converts
"implemented but unvalidated" into "executable and tested" for each adapter. For
each model it asserts, in order, on a real GPU:

  1. constructs                     - the configured model builds from its pinned
                                      revision, through the same code path the
                                      training backend uses.
  2. pretrained_weights_complete    - the pretrained checkpoint is present and
                                      actually reached the constructed model.
                                      A swallowed or partial load fails here.
  3. feature_map_contract           - backbone/FPN (or hybrid-encoder) channels +
                                      strides + NCHW spatial dims at the
                                      *configured* resolution, never 224.
  4. detection_head_class_count     - the constructed head is reset to the track
                                      class count (2 or 10), no COCO-80 residue.
  5. forward_backward_finite_loss   - one real forward+backward on a 2-image
                                      batch yields a finite loss and gradients.
  6. predict_wellformed             - one predict returns well-formed boxes.
  7. checkpoint_roundtrip           - save -> reload into a compatible model
                                      yields identical parameters.

Each model's checks run inside that model's isolated runtime (the same
interpreter the training backend is launched with), because the notebook kernel
does not have MMDetection, VMamba, or the pinned Transformers stack installed.

It writes a signed JSON record per model to the artifact root (never to
`results/` - a smoke pass is not a benchmark result). A run must not start until
a stored READY record exists for the model at the current commit, environment,
dataset track, and resolution; `src.workflows.adapter_gate.require_ready_adapter_gate`
enforces that from the live HPO and final-training workflows.

GPU-only: cannot run or be validated on a CPU host. The assertion primitives it
calls (`src.workflows.adapter_gate`) are unit-tested on CPU with stub tensors;
this orchestration around real adapters is validated only on hardware.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from src.config.benchmark_tracks import load_track_config
from src.utils.serialization import write_json
from src.workflows.adapter_gate import (
    SMOKE_CHECK_ORDER,
    adapter_fingerprint,
    assert_checkpoint_roundtrip,
    assert_detection_head_class_count,
    assert_feature_map_contract,
    assert_finite_loss,
    assert_pretrained_load_complete,
    assert_wellformed_predictions,
    build_smoke_record,
    smoke_check_result,
    smoke_record_path,
)
from src.workflows.adapter_smoke import run_adapter_gate, unexpected_failure_record

# Expected feature contract per framework, as ordered (channels, stride) pairs.
# These encode the architecture the benchmark intends to run and are confirmed
# against the real model on GPU; adjust here if a pinned upstream revision
# legitimately changes them.
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
    """Execute the ordered smoke checks for one model inside its own runtime."""
    torch = _require_cuda()
    import numpy as np

    from src.models.adapter_validation import AdapterValidationContract
    from src.models.registry import create_adapter, load_model_config

    config = load_model_config(model_id, repo_root)
    framework = str(config["framework"])
    expected_classes = TRACK_CLASS_COUNT[dataset_track]
    device = "cuda:0"
    adapter = create_adapter(model_id, device=device, repo_root=repo_root)
    if not isinstance(adapter, AdapterValidationContract):
        raise TypeError(
            f"{type(adapter).__name__} does not implement AdapterValidationContract; "
            "a model without the validation contract cannot be smoke-tested"
        )
    checks: list[dict[str, Any]] = []
    checkpoint: dict[str, Any] = {}

    def _construct() -> dict[str, Any]:
        adapter.build_for_validation(
            num_classes=expected_classes, image_size=image_size, device=device
        )
        return {"device": device, "num_classes": expected_classes}

    def _weights() -> dict[str, Any]:
        result = assert_pretrained_load_complete(adapter.checkpoint_load_result())
        checkpoint.update(result.as_dict())
        return result.as_dict()

    def _features() -> dict[str, Any]:
        maps = [level.as_dict() for level in adapter.feature_maps(image_size)]
        assert_feature_map_contract(maps, FEATURE_CONTRACT[framework], image_size)
        return {"levels": maps}

    def _head() -> dict[str, Any]:
        observed = adapter.head_class_count()
        assert_detection_head_class_count(observed, expected_classes)
        return {"head_class_count": observed}

    def _loss() -> dict[str, Any]:
        result = adapter.forward_backward(image_size=image_size, batch_size=2)
        assert_finite_loss(result.loss)
        if result.parameters_with_gradient == 0:
            raise RuntimeError("backward produced no gradients on any parameter")
        if not (result.maximum_absolute_gradient > 0.0):
            raise RuntimeError(
                "every gradient is exactly zero after one backward pass"
            )
        return result.as_dict()

    def _predict() -> dict[str, Any]:
        images = [
            np.random.default_rng(seed).integers(
                0, 255, (image_size, image_size, 3), dtype=np.uint8
            )
            for seed in (17, 42)
        ]
        predictions = adapter.predict(images)
        assert_wellformed_predictions(predictions, num_classes=expected_classes)
        return {"detections": [len(item["boxes"]) for item in predictions]}

    def _roundtrip() -> dict[str, Any]:
        before, after = adapter.checkpoint_roundtrip()
        assert_checkpoint_roundtrip(before, after)
        return {"parameters_compared": len(before)}

    steps: dict[str, Callable[[], dict[str, Any]]] = {
        "constructs": _construct,
        "pretrained_weights_complete": _weights,
        "feature_map_contract": _features,
        "detection_head_class_count": _head,
        "forward_backward_finite_loss": _loss,
        "predict_wellformed": _predict,
        "checkpoint_roundtrip": _roundtrip,
    }
    failure: dict[str, Any] | None = None
    try:
        for name in SMOKE_CHECK_ORDER:
            started = time.perf_counter()
            try:
                evidence = steps[name]()
            except Exception as error:  # noqa: BLE001 - the gate records every failure
                checks.append(
                    smoke_check_result(
                        name,
                        passed=False,
                        error=error,
                        duration_seconds=round(time.perf_counter() - started, 3),
                    )
                )
                failure = {
                    "check": name,
                    "exception_type": type(error).__name__,
                    "message": str(error),
                }
                break
            checks.append(
                smoke_check_result(
                    name,
                    passed=True,
                    evidence=evidence,
                    duration_seconds=round(time.perf_counter() - started, 3),
                )
            )
    finally:
        adapter.release()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fingerprint = adapter_fingerprint(model_id, repo_root)
    return build_smoke_record(
        model_id,
        fingerprint,
        checks,
        gpu=str(fingerprint.get("gpu")),
        dataset_track=dataset_track,
        image_size=image_size,
        checkpoint=checkpoint,
        failure=failure,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--drive-root")
    parser.add_argument("--dataset-track", default="2class", choices=["2class", "10class"])
    parser.add_argument("--model-id", default=None, help="limit to one model")
    parser.add_argument(
        "--in-runtime",
        action="store_true",
        help="run the checks in this interpreter (it must be the model runtime)",
    )
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--record-path", default=None)
    parser.add_argument(
        "--skip-provisioning",
        action="store_true",
        help="assume the model runtime is already provisioned and selected",
    )
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve()

    if args.in_runtime:  # pragma: no cover - GPU host only
        if not args.model_id or not args.record_path or not args.image_size:
            parser.error("--in-runtime requires --model-id, --image-size, --record-path")
        record_path = Path(args.record_path)
        try:
            record = run_model_smoke(
                args.model_id,
                repo_root,
                dataset_track=args.dataset_track,
                image_size=int(args.image_size),
            )
        except Exception as error:  # noqa: BLE001 - a crash still produces evidence
            record = unexpected_failure_record(
                args.model_id,
                repo_root,
                dataset_track=args.dataset_track,
                image_size=int(args.image_size),
                error=error,
                stage="constructs",
            )
            write_json(record_path, record)
            traceback.print_exc()
            print(f"{args.model_id}: {record['status']}")
            return 1
        write_json(record_path, record)
        print(f"{args.model_id}: {record['status']}")
        return 0 if record["status"] == "READY" else 1

    if not args.drive_root:
        parser.error("--drive-root is required")
    drive_root = Path(args.drive_root).expanduser().resolve()
    track = load_track_config(repo_root, "controlled")
    model_ids = [args.model_id] if args.model_id else list(track["model_ids"])

    all_ready = True
    for model_id in model_ids:  # pragma: no cover - GPU host only
        record = run_adapter_gate(
            model_id,
            repo_root,
            drive_root,
            dataset_track=args.dataset_track,
            image_size=args.image_size,
            skip_provisioning=args.skip_provisioning,
        )
        status = str(record.get("status"))
        all_ready = all_ready and status == "READY"
        print(f"{model_id}: {status}")
        print(f"  record: {smoke_record_path(drive_root, model_id, args.dataset_track)}")
    return 0 if all_ready else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
