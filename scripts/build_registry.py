#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
from src.paths import ProjectPaths
from src.training.checkpointing import RunRegistry

def main()->None:
 p=argparse.ArgumentParser();p.add_argument("--drive-root",required=True);args=p.parse_args();paths=ProjectPaths.from_value(args.drive_root).create();reg=RunRegistry(paths);count=0
 for manifest in paths.checkpoints.glob("*/*/run_manifest.json"):
  reg.register_run(manifest);count+=1
 print(f"registered {count} manifests")
if __name__=="__main__":main()
