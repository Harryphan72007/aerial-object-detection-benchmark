#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_PATH="${1:-.}"
GIT_USER_NAME="${GIT_USER_NAME:-<YOUR_NAME>}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-<YOUR_EMAIL>}"
git -C "$REPOSITORY_PATH" config user.name "$GIT_USER_NAME"
git -C "$REPOSITORY_PATH" config user.email "$GIT_USER_EMAIL"
echo "Configured identity for this repository only: $GIT_USER_NAME <$GIT_USER_EMAIL>"
