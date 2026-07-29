#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_PATH="${1:-.}"
BRANCH="${2:-main}"
if [[ -n "$(git -C "$REPOSITORY_PATH" status --porcelain)" ]]; then
  echo "Repository is dirty; inspect changes before pulling." >&2
  git -C "$REPOSITORY_PATH" status --short
  exit 2
fi
git -C "$REPOSITORY_PATH" fetch origin
git -C "$REPOSITORY_PATH" checkout "$BRANCH"
git -C "$REPOSITORY_PATH" pull --ff-only origin "$BRANCH"
