#!/usr/bin/env bash
# Stage every non-test runtime .baml source from baml_src/ into <dest_baml_src>,
# preserving the directory layout so future baml_src/ subdirectories work.
# Excludes *_test.baml files and everything under any tests/ directory; this is
# the exact test/runtime classification enforced by scripts/check_architecture.py.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/baml_src"

if [ $# -ne 1 ]; then
  echo "usage: scripts/stage_runtime.sh <dest_baml_src>" >&2
  exit 2
fi
DEST="$1"
mkdir -p "$DEST"

(
  cd "$SRC"
  find . -type f -name '*.baml' ! -name '*_test.baml' ! -path '*/tests/*' -print0
) |
while IFS= read -r -d '' file; do
  rel="${file#./}"
  mkdir -p "$DEST/$(dirname "$rel")"
  cp "$SRC/$rel" "$DEST/$rel"
done
