#!/usr/bin/env bash
# Small matched baseline/candidate comparison, not the full scaling campaign.
set -euo pipefail
export LC_ALL=C DO_NOT_TRACK=1
unset BAML_TELEMETRY BOUNDARY_API_KEY BOUNDARY_PROJECT
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CLI="$(realpath "$BAML_CLI")"
OUT="$(realpath -m "$1")"
BASE="${2:-40d58ba5de4e68a5a23182f6798f962339aa232e}"
mkdir -p "$OUT"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
git -C "$ROOT" archive "$BASE" migration/baml_src | tar -xf - -C "$WORK"
git -C "$ROOT" rev-parse HEAD > "$OUT/candidate.txt"
git -C "$ROOT" rev-parse "$BASE" > "$OUT/baseline.txt"
{ "$CLI" --version; uname -a; lscpu; free -m; } > "$OUT/environment.txt" 2>&1
: > "$OUT/trials.jsonl"
run_case() (
  local trial="$1" variant="$2" mode="$3" workload="$4" fn="$5" calls="$6" rows="$7"
  local case_dir="$OUT/$trial-$variant-$mode-$workload"
  mkdir -p "$case_dir"
  local project="$WORK/project"
  export HOME="$WORK/home" BAML_HOME="$WORK/home/baml"
  export XDG_CONFIG_HOME="$HOME/config" XDG_DATA_HOME="$HOME/data" XDG_CACHE_HOME="$HOME/cache"
  rm -rf "$project" "$HOME"
  mkdir -p "$project/baml_src" "$HOME"
  local src="$ROOT/migration/baml_src"
  if [ "$variant" = baseline ]; then src="$WORK/migration/baml_src"; fi
  for f in "$src/"*.baml; do
    case "$(basename "$f")" in *_test.baml|test_*.baml|example_checks.baml|generated_examples.baml) continue;; esac
    cp "$f" "$project/baml_src/"
  done
  cp "$ROOT/experiments/scaling/benchmark.baml" "$project/baml_src/"
  cd "$project"
  printf '[package]\nname = "validation-comparison"\n' > baml.toml
  BAML_TELEMETRY=off "$CLI" check > "$case_dir/check.log" 2>&1 || { cat "$case_dir/check.log"; exit 1; }
  if [ "$mode" = off ]; then export BAML_TELEMETRY=off; fi
  "$CLI" run "$fn" --output-format json > "$case_dir/result.json" 2> "$case_dir/run.stderr" || { cat "$case_dir/run.stderr"; exit 1; }
  jq -e --argjson calls "$calls" --argjson rows "$rows" '.ok and .calls==$calls and .rows==$rows and (.checksum|isfinite)' "$case_dir/result.json" >/dev/null
  "$CLI" local recordings query --output-format json "SELECT function_name, invocation_count FROM profiler WHERE function_name LIKE 'user.%'" > "$case_dir/profile.json" 2> "$case_dir/profile.stderr"
  jq -e '.outcome.status=="complete"' "$case_dir/profile.json" >/dev/null
  jq -cn --arg variant "$variant" --arg mode "$mode" --arg workload "$workload" --argjson trial "$trial" --slurpfile result "$case_dir/result.json" --slurpfile profile "$case_dir/profile.json" '
    def ns: if type=="object" then .["$bigint"]|tonumber else tonumber end;
    def count($name): ([$profile[0].rows[]|select(.[0]==("user."+$name))|.[1]]|add)//0;
    $result[0] as $r |
    {variant:$variant,mode:$mode,workload:$workload,trial:$trial,calls:$r.calls,rows:$r.rows,checksum:$r.checksum,elapsed_ns:($r.elapsed_ns|ns),finite_calls:(if $mode=="off" then null else count("IsFinite") end),matrix_scans:(if $mode=="off" then null else count("CheckFiniteMatrix") end)}
  ' | tee -a "$OUT/trials.jsonl"
)
for trial in 0 1 2; do
  for spec in "d512 D512 1000 1000" "b256 B256 100 25600"; do
    read -r workload fn calls rows <<< "$spec"
    for mode in off default; do
      if [ "$((trial%2))" = 0 ]; then variants=(baseline candidate); else variants=(candidate baseline); fi
      for variant in "${variants[@]}"; do run_case "$trial" "$variant" "$mode" "$workload" "$fn" "$calls" "$rows"; done
    done
  done
done
jq -s -e 'length==24 and all(.elapsed_ns>0) and (group_by(.workload)|all(length==12 and ([.[].checksum]|unique|length)==1)) and (map(select(.mode=="default"))|group_by([.workload,.trial])|all((map(select(.variant=="baseline"))[0]) as $b | (map(select(.variant=="candidate"))[0]) as $c | ($b.matrix_scans-$c.matrix_scans)==2*($b.calls+10) and $c.finite_calls<$b.finite_calls))' "$OUT/trials.jsonl" >/dev/null
jq -s 'group_by([.workload,.mode,.variant])|map({workload:.[0].workload,mode:.[0].mode,variant:.[0].variant,median_ns:(map(.elapsed_ns)|sort|.[1]),finite_calls:.[0].finite_calls,matrix_scans:.[0].matrix_scans})' "$OUT/trials.jsonl" | tee "$OUT/report.json"
