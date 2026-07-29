#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_PATH="${1:-/content/aerial-object-detection-benchmark}"
python -m pip install -e "$REPOSITORY_PATH"
if [[ -f "$REPOSITORY_PATH/requirements-colab.txt" ]]; then
  python -m pip install -r "$REPOSITORY_PATH/requirements-colab.txt"
fi
python -c "from src.drive_sync import initialize_drive_directories; initialize_drive_directories('/content/drive/MyDrive/visdrone_architecture_benchmark')"
