"""Dataset statistics and density-bin derivation."""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import numpy as np


def compute_statistics(annotation_file: str | Path) -> dict[str, Any]:
    data = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
    category_names = {int(c["id"]): c["name"] for c in data["categories"]}
    class_counts: Counter[str] = Counter(); areas=[]; widths=[]; heights=[]; per_image=defaultdict(int)
    truncation_counts: Counter[str] = Counter()
    occlusion_counts: Counter[str] = Counter()
    for ann in data["annotations"]:
        x, y, w, h = map(float, ann["bbox"])
        class_counts[str(category_names[int(ann["category_id"])])] += 1
        widths.append(w); heights.append(h); areas.append(w*h); per_image[int(ann["image_id"])] += 1
        attributes = ann.get("attributes", {})
        truncation_counts[str(attributes.get("truncation", "unknown"))] += 1
        occlusion_counts[str(attributes.get("occlusion", "unknown"))] += 1
    counts=np.array([per_image.get(int(img["id"]),0) for img in data["images"]], dtype=float)
    area_arr=np.asarray(areas, dtype=float)
    def describe(arr: np.ndarray) -> dict[str,float]:
        if arr.size == 0: return {k:0.0 for k in ("min","q25","median","q75","max","mean")}
        return {"min":float(arr.min()),"q25":float(np.quantile(arr,.25)),"median":float(np.median(arr)),"q75":float(np.quantile(arr,.75)),"max":float(arr.max()),"mean":float(arr.mean())}
    thresholds = [float(np.quantile(counts, q)) if counts.size else 0.0 for q in (.25,.50,.75)]
    return {
        "image_count": len(data["images"]), "annotation_count": len(data["annotations"]),
        "class_counts": dict(class_counts), "bbox_area": describe(area_arr),
        "truncation_counts": dict(truncation_counts),
        "occlusion_counts": dict(occlusion_counts),
        "images_without_valid_objects": int((counts == 0).sum()),
        "bbox_width": describe(np.asarray(widths)), "bbox_height": describe(np.asarray(heights)),
        "objects_per_image": describe(counts),
        "density_thresholds": {"low_max":thresholds[0],"medium_max":thresholds[1],"high_max":thresholds[2],"extremely_dense_above":thresholds[2]},
        "custom_size_counts": {
            "tiny": int((area_arr < 16**2).sum()),
            "small": int(((area_arr >= 16**2) & (area_arr < 32**2)).sum()),
            "medium": int(((area_arr >= 32**2) & (area_arr < 96**2)).sum()),
            "large": int((area_arr >= 96**2).sum()),
        },
    }
