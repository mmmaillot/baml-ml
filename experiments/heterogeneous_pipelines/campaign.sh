#!/usr/bin/env bash
# Reproducible storage/latency campaign for heterogeneous v3 RandomForest
# batches (Semantic vs Adaptive vs Full vs shadow). Writes a JSON report to
# "$1/report.json"; the committed copy is evidence/report.json.
set -euo pipefail
export LC_ALL=C DO_NOT_TRACK=1
unset BAML_TELEMETRY BOUNDARY_API_KEY BOUNDARY_PROJECT
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CLI="$(realpath "$BAML_CLI")"
OUT="$(realpath -m "$1")"
mkdir -p "$OUT"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

PROJECT="$WORK/project"
mkdir -p "$PROJECT/baml_src" "$PROJECT/artifacts/fixtures/v3"
for src in "$ROOT"/baml_src/*.baml; do
  case "$(basename "$src")" in *_test.baml) continue;; esac
  cp "$src" "$PROJECT/baml_src/"
done
cp "$ROOT/experiments/heterogeneous_pipelines/probes.baml" "$PROJECT/baml_src/"
cp "$ROOT"/artifacts/fixtures/v3/v3-ct-hetero-forest.json "$PROJECT/artifacts/fixtures/v3/"
python3 "$ROOT/experiments/heterogeneous_pipelines/generate_batch.py" "$PROJECT/batch.json"
printf '[package]\nname = "hetero-campaign-probe"\n' > "$PROJECT/baml.toml"

REPS=3
run_case() {
  local name="$1" fn="$2" args="$3"
  local best_ms=999999999 total_bytes=0 spans=0
  for rep in $(seq 1 $REPS); do
    local dir="$WORK/$name-$rep"
    rm -rf "$dir"
    mkdir -p "$dir"
    cp -R "$PROJECT/baml_src" "$PROJECT/artifacts" "$PROJECT/baml.toml" "$PROJECT/batch.json" "$dir/"
    (
      cd "$dir"
      export HOME="$dir/home" BAML_HOME="$dir/home/baml"
      export XDG_CONFIG_HOME="$HOME/config" XDG_DATA_HOME="$HOME/data" XDG_CACHE_HOME="$HOME/cache"
      mkdir -p "$HOME"
      local start end
      start=$(date +%s%N)
      "$CLI" run "$fn" $args --output-format json > "$dir/result.json" 2> "$dir/run.stderr" || { cat "$dir/run.stderr"; exit 1; }
      end=$(date +%s%N)
      echo $(( (end - start) / 1000000 )) > "$dir/elapsed_ms"
      "$CLI" local recordings query --output-format json "SELECT count(*) AS n FROM spans" > "$dir/spans.json" 2>/dev/null || echo '{"rows":[[0]]}' > "$dir/spans.json"
      du -sb "$dir/.baml" 2>/dev/null | cut -f1 > "$dir/bytes" || echo 0 > "$dir/bytes"
    )
    local ms bytes
    ms=$(cat "$dir/elapsed_ms")
    bytes=$(cat "$dir/bytes")
    total_bytes=$bytes
    if [ "$ms" -lt "$best_ms" ]; then best_ms=$ms; fi
    spans=$(jq -r '.rows[0][0]' "$dir/spans.json")
    # All repetitions must produce identical predictions.
    if [ "$rep" = 1 ]; then cp "$dir/result.json" "$WORK/$name-result.json"; else
      diff <(jq -S . "$WORK/$name-result.json") <(jq -S . "$dir/result.json") >/dev/null
    fi
  done
  jq -n --arg name "$name" --argjson ms "$best_ms" --argjson bytes "$total_bytes" --argjson spans "$spans" --argjson reps $REPS \
    '{mode: $name, best_wall_ms: $ms, recording_bytes: $bytes, span_count: $spans, repetitions: $reps}'
}

rm -f "$OUT/results.jsonl"
run_case semantic HeteroCampaignSemanticProbe "" >> "$OUT/results.jsonl"
run_case adaptive HeteroCampaignAdaptiveProbe "" >> "$OUT/results.jsonl"
run_case full HeteroCampaignFullProbe "" >> "$OUT/results.jsonl"
run_case shadow HeteroCampaignShadowProbe "" >> "$OUT/results.jsonl"
jq -s '{campaign: "heterogeneous v3 RandomForestClassifier batch (500 rows)",
  model: "artifacts/fixtures/v3/v3-ct-hetero-forest.json",
  batch: "experiments/heterogeneous_pipelines/batch.json",
  runtime_commit: "'"$(git -C "$ROOT" rev-parse HEAD)"'",
  baml_src_sha256: "'"$(cat "$ROOT"/baml_src/*.baml | sha256sum | cut -d" " -f1)"'",
  platform: "'"$(uname -s)/$(uname -m)"'",
  baml_cli: "'"$("$CLI" --version 2>/dev/null | awk '{print $2}')"'",
  results: .}' "$OUT/results.jsonl" > "$OUT/report.json"
jq -r '.results[] | [.mode, .best_wall_ms, .recording_bytes, .span_count] | @tsv' "$OUT/report.json" \
  | awk -F'\t' 'BEGIN{printf "%-10s %12s %16s %8s\n","mode","best_ms","rec_bytes","spans"} {printf "%-10s %12s %16s %8s\n",$1,$2,$3,$4}'
echo "campaign report written to $OUT/report.json"
