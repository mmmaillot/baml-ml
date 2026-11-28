#!/usr/bin/env bash
# Download and SHA-256-verify the pinned BAML toolchain release, then print
# the installed executable paths. Supports linux-x86_64, linux-arm64 and
# macos-arm64. In GitHub Actions the paths are also written to GITHUB_ENV
# (BAML_CLI, BAML_PACK_HOST) and GITHUB_OUTPUT (baml-cli, baml-pack-host).
set -euo pipefail

BAML_VERSION="0.26.0"
BASE_URL="https://github.com/BoundaryML/baml/releases/download/baml-language-${BAML_VERSION}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

usage() {
  cat >&2 <<'EOF'
usage: scripts/install_baml.sh [--dest DIR] [--platform PLATFORM] [--pack-host] [--force]

  --dest DIR        install destination (default: .toolchain/ in the repo, or
                    $RUNNER_TEMP/baml-toolchain when RUNNER_TEMP is set)
  --platform P      linux-x86_64 | linux-arm64 | macos-arm64 (default: auto-detect)
  --pack-host       also install and export baml-pack-host (BAML_PACK_HOST)
  --force           re-download even if a matching toolchain is already present
EOF
}

DEST=""
PLATFORM=""
PACK_HOST=0
FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dest) DEST="${2:?--dest needs a value}"; shift 2;;
    --platform) PLATFORM="${2:?--platform needs a value}"; shift 2;;
    --pack-host) PACK_HOST=1; shift;;
    --force) FORCE=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown argument: $1" >&2; usage; exit 2;;
  esac
done

target_for() {
  case "$1" in
    linux-x86_64) echo "x86_64-unknown-linux-gnu x86_64";;
    linux-arm64) echo "aarch64-unknown-linux-gnu aarch64";;
    macos-arm64) echo "aarch64-apple-darwin arm64";;
    *) echo "unsupported platform: $1 (expected linux-x86_64, linux-arm64 or macos-arm64)" >&2; return 1;;
  esac
}

if [ -z "$PLATFORM" ]; then
  case "$(uname -s)/$(uname -m)" in
    Linux/x86_64) PLATFORM=linux-x86_64;;
    Linux/aarch64) PLATFORM=linux-arm64;;
    Darwin/arm64) PLATFORM=macos-arm64;;
    *) echo "unsupported host: $(uname -s)/$(uname -m) (expected linux-x86_64, linux-arm64 or macos-arm64)" >&2; exit 1;;
  esac
fi
read -r TARGET MACHINE <<< "$(target_for "$PLATFORM")"
if [ "$(uname -m)" != "$MACHINE" ]; then
  echo "platform $PLATFORM expects machine $MACHINE, but this host is $(uname -m)" >&2
  exit 1
fi

if [ -z "$DEST" ]; then
  if [ -n "${RUNNER_TEMP:-}" ]; then DEST="$RUNNER_TEMP/baml-toolchain"; else DEST="$ROOT/.toolchain"; fi
fi
mkdir -p "$DEST"

find_tool() {
  find "$DEST" -type f -name "$1" -print -quit
}

CLI="$(find_tool baml-cli)"
HOST="$(find_tool baml-pack-host)"
if [ "$FORCE" -eq 0 ] && [ -n "$CLI" ] && [ -x "$CLI" ] \
  && [ "$("$CLI" --version 2>/dev/null)" = "baml-cli ${BAML_VERSION}" ] \
  && { [ "$PACK_HOST" -eq 0 ] || { [ -n "$HOST" ] && [ -x "$HOST" ]; }; }; then
  echo "BAML ${BAML_VERSION} toolchain already installed in $DEST"
else
  ARCHIVE="baml-language-${BAML_VERSION}-${TARGET}.tar.gz"
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT
  for asset in "$ARCHIVE" "$ARCHIVE.sha256"; do
    curl --proto '=https' --tlsv1.2 --retry 3 --fail --silent --show-error --location \
      "$BASE_URL/$asset" --output "$WORK/$asset"
  done
  (cd "$WORK"; if [ "$(uname -s)" = Darwin ]; then shasum -a 256 -c "$ARCHIVE.sha256"; else sha256sum --check --strict "$ARCHIVE.sha256"; fi)
  tar -C "$DEST" -xzf "$WORK/$ARCHIVE"
  CLI="$(find_tool baml-cli)"
  test -n "$CLI"
  chmod +x "$CLI"
  echo "verified and installed: $CLI"
fi

"$CLI" --version

if [ "$PACK_HOST" -eq 1 ]; then
  if [ -z "$HOST" ] || [ ! -x "$HOST" ] || [ "$FORCE" -eq 1 ]; then
    HOST="$(find_tool baml-pack-host)"
    test -n "$HOST"
    chmod +x "$HOST"
    echo "verified and installed: $HOST"
  else
    echo "BAML ${BAML_VERSION} pack host already installed: $HOST"
  fi
fi

if [ -n "${GITHUB_ENV:-}" ]; then
  echo "BAML_CLI=$CLI" >> "$GITHUB_ENV"
  if [ "$PACK_HOST" -eq 1 ]; then echo "BAML_PACK_HOST=$HOST" >> "$GITHUB_ENV"; fi
fi
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "baml-cli=$CLI" >> "$GITHUB_OUTPUT"
  if [ "$PACK_HOST" -eq 1 ]; then echo "baml-pack-host=$HOST" >> "$GITHUB_OUTPUT"; fi
fi
echo "export BAML_CLI=$CLI"
if [ "$PACK_HOST" -eq 1 ]; then echo "export BAML_PACK_HOST=$HOST"; fi
