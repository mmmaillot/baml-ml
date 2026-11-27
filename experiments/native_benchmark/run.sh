#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C DO_NOT_TRACK=1
unset BAML_TELEMETRY BOUNDARY_API_KEY BOUNDARY_PROJECT
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SOURCE="$ROOT/experiments/native_benchmark"
CLI="$(realpath "$BAML_CLI")"
OUT="$(realpath -m "$1")"
mkdir -p "$OUT"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
"$CLI" --version > "$OUT/version.txt" 2>&1
{ uname -a; lscpu; free -m; jq --version; /usr/bin/time --version; } > "$OUT/environment.txt"
printf '' > "$OUT/trials.jsonl"

inventory() {
  { for dir in "$PROJECT/.baml" "$HOME"; do
      if [ -d "$dir" ]; then find "$dir" -type f -printf '%s\t%p\n'; fi
    done; } | sort -k2 > "$CASE/storage-$1.tsv"
  awk -F '\t' '
    { total += $1; if ($2 ~ /\.btel$/) btel += $1;
      else if ($2 ~ /\/cas\/|\/blobs\//) cas += $1;
      else if ($2 ~ /\.sqlite|\.db(-wal|-shm)?$|\/index\//) idx += $1;
      else other += $1 }
    END {printf "{\"total\":%d,\"btel\":%d,\"cas\":%d,\"index\":%d,\"other\":%d}\n",total,btel,cas,idx,other}
  ' "$CASE/storage-$1.tsv" > "$CASE/storage-$1.json"
}

run_trial() (
  local trial="$1" mode="$2" workload="$3" fn="$4" calls="$5" rows="$6"
  CASE="$OUT/$trial-$mode-$workload"
  mkdir -p "$CASE"
  PROJECT="$WORK/project"
  export HOME="$WORK/home" BAML_HOME="$WORK/home/baml"
  export XDG_CONFIG_HOME="$HOME/config" XDG_DATA_HOME="$HOME/data" XDG_CACHE_HOME="$HOME/cache"
  rm -rf "$PROJECT" "$HOME"
  mkdir -p "$PROJECT/baml_src" "$HOME"
  cd "$PROJECT"
  printf '[package]\nname = "native-benchmark"\n' > baml.toml
  for src in "$ROOT"/migration/baml_src/*.baml; do
    case "$(basename "$src")" in *_test.baml|test_*.baml|example_checks.baml|generated_examples.baml) continue;; esac
    cp "$src" baml_src/
  done
  cp "$ROOT/experiments/native_ml_observability/iris_fixture.baml" "$SOURCE/"*.baml baml_src/
  if [ "$mode" = empty ] || [ "$mode" = span ]; then
    directive=span
    if [ "$mode" = empty ]; then directive=empty_span; fi
    for func in ClassificationPipelinePredictProba ImputerTransform ScalerTransformCore LogRegPredictProbaCore ClassificationPipelineFit ImputerFit ScalerFit LogRegFit; do
      file="$(grep -l "^function $func(" baml_src/*.baml)"
      test "$(printf '%s\n' "$file" | wc -l)" -eq 1
      sed -i "/^function $func(/i /// baml:\$trace=trace.$directive" "$file"
    done
  fi
  BAML_TELEMETRY=off "$CLI" check > "$CASE/check.stdout" 2> "$CASE/check.stderr" || { cat "$CASE/check.stderr"; exit 1; }
  inventory pre-run
  if [ "$mode" = off ]; then export BAML_TELEMETRY=off; fi
  /usr/bin/time -f '%e\t%U\t%S\t%M' -o "$CASE/process.tsv" "$CLI" run "$fn" --output-format json > "$CASE/result.json" 2> "$CASE/run.stderr" || { cat "$CASE/run.stderr"; exit 1; }
  # bigint output is version-specific; explicitly accept documented encodings.
  jq -e --argjson calls "$calls" --argjson rows "$rows" '
    def ns: if type=="object" then .["$bigint"]|tonumber else tonumber end;
    .ok and .calls==$calls and .rows==$rows and (.elapsed_ns|ns)>0 and (.checksum|isfinite)
  ' "$CASE/result.json" >/dev/null || { cat "$CASE/result.json"; exit 1; }
  inventory pre-query
  "$CLI" local recordings query --output-format json "SELECT COUNT(*) AS n FROM spans" > "$CASE/spans.json" 2> "$CASE/spans.stderr"
  "$CLI" local recordings query --output-format json "SELECT function_name, total_time, self_time, invocation_count, error_count FROM profiler WHERE function_name LIKE 'user.%'" > "$CASE/profiler.json" 2> "$CASE/profiler.stderr"
  jq -e '.outcome.status=="complete"' "$CASE/spans.json" "$CASE/profiler.json" >/dev/null
  if [ "$mode" = off ]; then
    jq -e '.rows == [[0]]' "$CASE/spans.json" >/dev/null
    jq -e '.rows == []' "$CASE/profiler.json" >/dev/null
  else
    jq -e '.rows[0][0]>0' "$CASE/spans.json" >/dev/null
    if [ "$workload" = fit ]; then name=ClassificationPipelineFit; expected=4; else name=ClassificationPipelinePredictProba; expected=$((calls+10)); fi
    jq -e --arg fn "user.$name" --argjson expected "$expected" '[.rows[]|select(.[0]==$fn)|.[3]]|add==$expected' "$CASE/profiler.json" >/dev/null
  fi
  inventory post-query
  IFS=$'\t' read -r wall user sys rss < "$CASE/process.tsv"
  jq -cn --arg mode "$mode" --arg workload "$workload" --argjson trial "$trial" --argjson wall "$wall" --argjson user "$user" --argjson sys "$sys" --argjson rss "$rss" --slurpfile result "$CASE/result.json" --slurpfile pre "$CASE/storage-pre-run.json" --slurpfile post "$CASE/storage-pre-query.json" --slurpfile indexed "$CASE/storage-post-query.json" --slurpfile spans "$CASE/spans.json" '
    def ns: if type=="object" then .["$bigint"]|tonumber else tonumber end;
    $result[0] as $r |
    {mode:$mode,workload:$workload,trial:$trial,calls:$r.calls,rows:$r.rows,checksum:$r.checksum,elapsed_ns:($r.elapsed_ns|ns),wall_s:$wall,user_s:$user,sys_s:$sys,peak_rss_kib:$rss,spans:$spans[0].rows[0][0],btel_bytes:($post[0].btel-$pre[0].btel),cas_bytes:($post[0].cas-$pre[0].cas),index_bytes:($indexed[0].index-$post[0].index),other_bytes:($post[0].other-$pre[0].other),recording_total_delta:($post[0].total-$pre[0].total)}
  ' | tee -a "$OUT/trials.jsonl"
  # Preserve raw recordings for a representative trial; all trials retain JSON
  # and full file-size inventory. Avoid archiving identical toolchain caches.
  if [ "$trial" = 0 ] && [ -d "$PROJECT/.baml/btel" ]; then
    tar -C "$PROJECT/.baml" -czf "$CASE/btel.tar.gz" btel
  fi
)

modes=(off default empty span)
# Four repetitions rotate mode order; each workload gets equal treatment.
for trial in 0 1 2 3; do
  for spec in "zero BenchZero 0 0" "single100 Bench100 100 100" "single1000 Bench1000 1000 1000" "single5000 Bench5000 5000 5000" "batch30 BenchBatch 100 3000" "fit BenchFit 3 360"; do
    read -r workload fn calls rows <<< "$spec"
    for offset in 0 1 2 3; do
      mode="${modes[$(((trial+offset)%4))]}"
      run_trial "$trial" "$mode" "$workload" "$fn" "$calls" "$rows"
    done
  done
done
# Exact numeric checksum parity across all modes/trials; no latency gate.
jq -s -e 'length==96 and (group_by(.workload)|all(length==16 and ([.[].checksum]|unique|length)==1))' "$OUT/trials.jsonl" >/dev/null
jq -s -f "$SOURCE/report.jq" "$OUT/trials.jsonl" > "$OUT/report.json"
cat "$OUT/report.json"
