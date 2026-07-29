#!/usr/bin/env python
"""Compatibility entry point for the LR-only controlled search."""
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.lr_search import main


if __name__ == "__main__":
    main()
