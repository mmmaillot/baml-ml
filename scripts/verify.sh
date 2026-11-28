#!/usr/bin/env bash
# Single-command local verification. Fast pre-PR checks by default; --full
# additionally runs the experiment harnesses under experiments/.
set -euo pipefail
export LC_ALL=C DO_NOT_TRACK=1
unset BAML_TELEMETRY BOUNDARY_API_KEY BOUNDARY_PROJECT

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

usage() {
  cat >&2 <<'EOF'
usage: scripts/verify.sh [--full]

  (default)   BAML_TELEMETRY=off baml-cli check + test --max-concurrency 1 --test-timeout 5m
  --full      also runs experiments/imported_inference/run.sh,
              experiments/native_ml_observability/run.sh and
              experiments/heterogeneous_pipelines/run.sh; additionally
              experiments/standalone/run.sh when a baml-pack-host is available
              (BAML_PACK_HOST or .toolchain/ from ./scripts/install_baml.sh --pack-host)

The baml-cli is resolved from BAML_CLI, else .toolchain/ (see ./scripts/install_baml.sh).
EOF
}

FULL=0
for arg in "$@"; do
  case "$arg" in
    --full) FULL=1;;
    -h|--help) usage; exit 0;;
    *) echo "unknown argument: $arg" >&2; usage; exit 2;;
  esac
done

find_tool() {
  # $1: environment variable name, $2: executable name under .toolchain/
  local path=""
  if [ -n "${!1:-}" ]; then
    path="$(realpath "${!1}")"
  elif [ -d "$ROOT/.toolchain" ]; then
    path="$(find "$ROOT/.toolchain" -type f -name "$2" -print -quit)"
  fi
  if [ -z "$path" ] || [ ! -x "$path" ]; then
    echo "error: $2 not found (set $1 or run ./scripts/install_baml.sh)" >&2
    exit 1
  fi
  printf '%s\n' "$path"
}

CLI="$(find_tool BAML_CLI baml-cli)"
export BAML_CLI="$CLI"
echo "Using BAML_CLI=$BAML_CLI ($("$CLI" --version 2>/dev/null))"

echo "==> baml check"
BAML_TELEMETRY=off "$CLI" check
echo "==> baml test"
BAML_TELEMETRY=off "$CLI" test --max-concurrency 1 --test-timeout 5m

if [ "$FULL" -eq 1 ]; then
  RESULTS="$(mktemp -d)"
  trap 'rm -rf "$RESULTS"' EXIT
  echo "==> experiments/imported_inference/run.sh (results in $RESULTS/imported-inference)"
  bash "$ROOT/experiments/imported_inference/run.sh" "$RESULTS/imported-inference"
  echo "==> experiments/native_ml_observability/run.sh (results in $RESULTS/native-ml-observability)"
  bash "$ROOT/experiments/native_ml_observability/run.sh" "$RESULTS/native-ml-observability"
  echo "==> experiments/heterogeneous_pipelines/run.sh (results in $RESULTS/heterogeneous-pipelines)"
  bash "$ROOT/experiments/heterogeneous_pipelines/run.sh" "$RESULTS/heterogeneous-pipelines"
  HOST="${BAML_PACK_HOST:-}"
  if [ -z "$HOST" ] && [ -d "$ROOT/.toolchain" ]; then
    HOST="$(find "$ROOT/.toolchain" -type f -name baml-pack-host -print -quit)"
  fi
  if [ -n "$HOST" ] && [ -x "$HOST" ]; then
    export BAML_PACK_HOST="$(realpath "$HOST")"
    echo "==> experiments/standalone/run.sh (results in $RESULTS/standalone)"
    bash "$ROOT/experiments/standalone/run.sh" "$RESULTS/standalone"
  else
    echo "==> skipping experiments/standalone/run.sh (no baml-pack-host; run ./scripts/install_baml.sh --pack-host to enable)"
  fi
fi

echo "All requested checks passed."
