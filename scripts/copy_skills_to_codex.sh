#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$REPO_ROOT/skills"
DEST_DIR="$HOME/.codex/skills"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "Source skills directory not found: $SRC_DIR" >&2
  exit 1
fi

mkdir -p "$DEST_DIR"

# Mirror repo skills into Codex skills (overwrite existing entries)
cp -R "$SRC_DIR/." "$DEST_DIR/"

echo "Copied skills from $SRC_DIR to $DEST_DIR"
