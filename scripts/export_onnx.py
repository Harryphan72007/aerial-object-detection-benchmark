#!/usr/bin/env python
"""Attempt ONNX export and preserve failures as deployment measurements."""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from src.models.rtdetr_adapter import RTDetrV2Adapter
from src.paths import ProjectPaths
from src.training.checkpointing import RunRegistry
from src.utils.serialization import read_yaml, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    paths = ProjectPaths.from_value(args.drive_root).create()
    registry = RunRegistry(paths)
    run = next(
        item
        for item in registry.list_available_runs(status=None)
        if item["run_id"] == args.run_id
    )
    run_dir = paths.run_dir(run["model_id"], run["run_id"])
    result = {"run_id": args.run_id, "model_id": run["model_id"], "status": "failed"}
    try:
        if run["model_id"] != "rtdetrv2_l":
            raise RuntimeError(
                "MMDetection export is backend-specific. Use MMDeploy with an "
                "explicit deployment config and record that command rather than "
                "claiming a generic torch.onnx export."
            )
        import torch

        config = read_yaml(run_dir / "model_config.yaml")
        config["input_resolution"] = int(run["input_resolution"])
        adapter = RTDetrV2Adapter(device="cpu")
        model = adapter.load_model(
            registry.load_checkpoint_from_registry(args.run_id), config
        )

        class ExportWrapper(torch.nn.Module):
            def __init__(self, detector):
                super().__init__()
                self.detector = detector

            def forward(self, pixel_values):
                outputs = self.detector(pixel_values=pixel_values)
                return outputs.logits, outputs.pred_boxes

        wrapper = ExportWrapper(model).eval()
        resolution = int(run["input_resolution"])
        dummy = torch.randn(1, 3, resolution, resolution)
        output = paths.evaluation / f"{args.run_id}.onnx"
        torch.onnx.export(
            wrapper,
            (dummy,),
            output,
            opset_version=args.opset,
            input_names=["pixel_values"],
            output_names=["logits", "pred_boxes"],
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "logits": {0: "batch"},
                "pred_boxes": {0: "batch"},
            },
        )
        result.update(
            {
                "status": "completed",
                "onnx_path": str(output),
                "onnx_size_bytes": output.stat().st_size,
                "opset": args.opset,
            }
        )
    except Exception as error:
        result.update(
            {"error": repr(error), "traceback": traceback.format_exc()}
        )
    write_json(paths.evaluation / f"{args.run_id}__onnx_export.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
