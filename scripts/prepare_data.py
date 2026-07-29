#!/usr/bin/env python
"""Convert VisDrone train/val splits to validated COCO tracks."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from src.data.collapse_classes import ClassMapping
from src.data.convert_visdrone import convert_split
from src.data.download import ensure_visdrone_layout
from src.data.statistics import compute_statistics
from src.data.validate_annotations import validate_coco
from src.paths import ProjectPaths

def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--drive-root",required=True);p.add_argument("--raw-root");p.add_argument("--tracks",nargs="+",choices=["2class","10class"],default=["2class","10class"]);p.add_argument("--exclude-light-vehicles",action="store_true");p.add_argument("--validate",action="store_true");args=p.parse_args()
    paths=ProjectPaths.from_value(args.drive_root).create();raw=Path(args.raw_root) if args.raw_root else paths.raw;ensure_visdrone_layout(raw)
    for track in args.tracks:
        mapping=ClassMapping(track,exclude_light_vehicles=args.exclude_light_vehicles);out=paths.coco(track);(out/"annotations").mkdir(parents=True,exist_ok=True)
        for split,source in (("train","VisDrone2019-DET-train"),("val","VisDrone2019-DET-val")):
            src=raw/source;dest_images=out/split
            if dest_images.exists() or dest_images.is_symlink():
                if dest_images.is_symlink() and dest_images.resolve()==(src/"images").resolve():pass
                elif dest_images.exists():pass
            else: dest_images.symlink_to((src/"images").resolve(),target_is_directory=True)
            ann=out/"annotations"/f"instances_{split}.json";summary=convert_split(src/"images",src/"annotations",ann,mapping)
            print(track,split,summary)
            if args.validate:
                report=validate_coco(ann,dest_images);print(json.dumps(report.__dict__,indent=2));report.raise_for_errors()
            stats=compute_statistics(ann);(out/"annotations"/f"statistics_{split}.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
if __name__=="__main__":main()
