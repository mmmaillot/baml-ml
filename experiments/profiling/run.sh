#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
HERE="$(cd "$(dirname "$0")" && pwd)"
ARCHIVE="$1"
OUT="$2"
mkdir -p "$OUT"
for trial in 0 1 2 3; do
  for workload in d4 d32 d128 d512 b32 b256 b1024 repeat100 distinct100 distinct1000 fit150 fit1000 fit10000; do
    unzip -p "$ARCHIVE" "$trial-default-$workload/profiler.json" |
      jq -c --arg workload "$workload" --argjson trial "$trial" '{workload:$workload,trial:$trial,profile:.}'
  done
done > "$OUT/profiles.jsonl"
jq -s -f "$HERE/analyze.jq" "$OUT/profiles.jsonl" > "$OUT/report.json"
jq -r '.[] | .workload as $w | .functions[] | [$w,.name,.self_ns,.inclusive_ns,.calls,.share,.amdahl_infinite,.amdahl_10x] | @tsv' "$OUT/report.json" > "$OUT/functions.tsv"
