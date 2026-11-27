#!/usr/bin/env bash
# Native BAML CLI only; Bash/jq are external test harness, no SDK.
set -euo pipefail
export LC_ALL=C DO_NOT_TRACK=1
unset BAML_TELEMETRY BOUNDARY_API_KEY BOUNDARY_PROJECT
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SOURCE="$ROOT/experiments/native_ml_observability"
CLI="$(realpath "$BAML_CLI")"
OUT="$(realpath -m "$1")"
mkdir -p "$OUT"
"$CLI" --version > "$OUT/version.txt" 2>&1
uname -a > "$OUT/environment.txt"
jq --version >> "$OUT/environment.txt"
ORACLE="$SOURCE/oracle.json"
cp "$ORACLE" "$OUT/oracle.json"

query() {
  "$CLI" local recordings query --output-format json "$2" > "$CASE/$1.json" 2> "$CASE/$1.stderr"
  jq -e '.outcome.status == "complete"' "$CASE/$1.json" >/dev/null
}

inventory() {
  # Count all files under native runtime storage paths, including CAS and indexes,
  # separately before and after the first SQL query; logical bytes, not disk blocks.
  { for dir in "$PROJECT/.baml" "$HOME"; do
      if [ -d "$dir" ]; then find "$dir" -type f -printf '%s\t%p\n'; fi
    done; } | sort -k2 > "$CASE/storage-$1.tsv"
  awk -F '\t' '
    { total += $1; if ($2 ~ /\.btel$/) btel += $1;
      else if ($2 ~ /\/cas\/|\/blobs\//) cas += $1;
      else if ($2 ~ /\.sqlite|\.db(-wal|-shm)?$|\/index\//) idx += $1;
      else other += $1 }
    END {printf "{\"total\":%d,\"btel\":%d,\"cas_named_paths\":%d,\"index_named_paths\":%d,\"other\":%d}\n",total,btel,cas,idx,other}
  ' "$CASE/storage-$1.tsv" > "$CASE/storage-$1.json"
}

prepare() {
  CASE="$OUT/$1"
  PROJECT="$CASE/project"
  mkdir -p "$PROJECT" "$CASE/home"
  export HOME="$CASE/home" BAML_HOME="$CASE/home/baml"
  export XDG_CONFIG_HOME="$HOME/config" XDG_DATA_HOME="$HOME/data" XDG_CACHE_HOME="$HOME/cache"
  printf '[package]\nname = "native-ml-probe"\n' > "$PROJECT/baml.toml"
  cd "$PROJECT"
}

collect() {
  inventory before-query
  "$CLI" local recordings schema > "$CASE/schema.txt" 2> "$CASE/schema.stderr"
  query spans "SELECT span_id, parent_span_id, span_name, status, duration, input_args, output_value, error_value FROM spans ORDER BY start_time"
  query profiler "SELECT profiler_node_id, parent_profiler_node_id, function_name, total_time, self_time, invocation_count, error_count FROM profiler"
  inventory after-query
  echo "CASE $(basename "$CASE")"
  cat "$CASE/storage-before-query.json" "$CASE/storage-after-query.json"
  jq -c '.rows | map(.[0:5])' "$CASE/spans.json"
  jq -c '.rows' "$CASE/profiler.json"
}

for mode in off default empty span; do
  for sample in missing complete; do
    (
      prepare "$mode-$sample"
      mkdir -p baml_src
      # Exact copies of existing library sources; omit tests and generated fixtures.
      for src in "$ROOT"/baml_src/*.baml; do
        case "$(basename "$src")" in *_test.baml|test_*.baml|example_checks.baml|generated_examples.baml) continue;; esac
        cp "$src" baml_src/
      done
      cp "$SOURCE/iris_fixture.baml" baml_src/
      if [ "$mode" = empty ] || [ "$mode" = span ]; then
        directive=span
        if [ "$mode" = empty ]; then directive=empty_span; fi
        for fn in ClassificationPipelinePredictProba ImputerTransform ScalerTransformCore LogRegPredictProbaCore; do
          file="$(grep -l "^function $fn(" baml_src/*.baml)"
          test "$(printf '%s\n' "$file" | wc -l)" -eq 1
          sed -i "/^function $fn(/i /// baml:\$trace=trace.$directive" "$file"
        done
      fi
      # Compile without recordings so only the invocation under test is collected.
      BAML_TELEMETRY=off "$CLI" check > "$CASE/check.stdout" 2> "$CASE/check.stderr" || { cat "$CASE/check.stderr"; exit 1; }
      if [ "$mode" = off ]; then export BAML_TELEMETRY=off; fi
      fn=IrisMissing
      if [ "$sample" = complete ]; then fn=IrisComplete; fi
      "$CLI" run "$fn" --output-format json > "$CASE/prediction.json" 2> "$CASE/run.stderr" || { cat "$CASE/run.stderr"; exit 1; }
      jq -e --arg s "$sample" --slurpfile ref "$ORACLE" '
        .ok and .error == "" and (.probs|length)==1 and
        ([range(0;3) as $i | (.probs[0][$i] - $ref[0][$s].probs[$i]) | fabs < 1e-10] | all)
      ' "$CASE/prediction.json"
      collect
      jq -c '.rows[] | select(.[2] | test("ImputerTransform|ScalerTransformCore|LogRegPredictProbaCore|ClassificationPipelinePredictProba"))' "$CASE/spans.json"
      if [ "$mode" = off ]; then
        jq -e '.rows == []' "$CASE/spans.json" "$CASE/profiler.json" >/dev/null
      else
        jq -e '[.rows[][2] | split(".")[-1]] | (index("ImputerTransform") != null and index("ScalerTransformCore") != null and index("LogRegPredictProbaCore") != null)' "$CASE/profiler.json" >/dev/null
      fi
      if [ "$mode" = empty ] || [ "$mode" = span ]; then
        jq -e '[.rows[][2] | split(".")[-1]] | (index("ImputerTransform") != null and index("ScalerTransformCore") != null and index("LogRegPredictProbaCore") != null)' "$CASE/spans.json" >/dev/null
      fi
      if [ "$mode" = span ]; then
        jq -e --arg s "$sample" --slurpfile ref "$ORACLE" '
          def row($n): [.rows[] | select(.[2] == ("user."+$n))] | if length == 1 then .[0] else error("missing or duplicate span "+$n) end;
          def near($a;$b): ($a|length)==($b|length) and ([range(0;($b|length)) as $i | ($a[$i]-$b[$i])|fabs < 1e-10]|all);
          $ref[0][$s] as $e |
          row("ClassificationPipelinePredictProba") as $p |
          row("ImputerTransform") as $i |
          row("ScalerTransformCore") as $z |
          row("LogRegPredictProbaCore") as $m |
          ([$i,$z,$m] | all(.[1]==$p[0] and .[3]=="return" and .[4]>0)) and
          ($i[5].x[0] | if $s=="missing" then .[1]["$float"]=="NaN" and .[0]==$e.input[0] and .[2:4]==$e.input[2:4] else .==$e.input end) and
          near($i[6].x[0];$e.imputed) and near($z[5].x[0];$e.imputed) and
          near($z[6].x[0];$e.scaled) and near($m[5].x[0];$e.scaled) and
          near($m[6].probs[0];$e.probs) and near($p[6].probs[0];$e.probs) and
          ($m[5].model.weights|length)==3
        ' "$CASE/spans.json" > "$CASE/payload-assertion.json"
      elif [ "$mode" = empty ]; then
        jq -e '[.rows[] | select((.[2]|split(".")[-1]) as $n | ["ClassificationPipelinePredictProba","ImputerTransform","ScalerTransformCore","LogRegPredictProbaCore"] | index($n)!=null)] | length==4 and all(.[5]==null and .[6]==null)' "$CASE/spans.json" > "$CASE/payload-assertion.json"
      fi
    )
  done
done
# Independent scalar, distinct-value matrix and failure payloads.
for sample in scalar matrix failure; do
  (
    prepare "payload-$sample"
    cp "$ROOT/experiments/observability/probe.baml" "$SOURCE/payload_probe.baml" .
    BAML_TELEMETRY=off "$CLI" check > "$CASE/check.stdout" 2> "$CASE/check.stderr" || { cat "$CASE/check.stderr"; exit 1; }
    fn=PayloadScalar
    if [ "$sample" = matrix ]; then fn=PayloadMatrix; fi
    if [ "$sample" = failure ]; then fn=Failure; fi
    rc=0
    "$CLI" run "$fn" --output-format json > "$CASE/prediction.json" 2> "$CASE/run.stderr" || rc=$?
    if [ "$sample" = failure ]; then test "$rc" -eq 1; else test "$rc" -eq 0; fi
    collect
    if [ "$sample" = matrix ]; then
      jq -e '[.rows[] | select(.[2]=="user.matrix_echo")] | length==1 and (.[0] | .[5].x==.[6] and (.[6]|length)==100 and ([range(0;100) as $i | range(0;100) as $j | .[6][$i][$j]==(($i*100+$j+1)*0.125)]|all))' "$CASE/spans.json" > "$CASE/payload-assertion.json"
      jq -c '.rows[] | select(.[2]|endswith("matrix_echo")) | {name:.[2],input_type:(.[5]|type),output_type:(.[6]|type),input_preview:(.[5]|tostring|.[0:300]),output_preview:(.[6]|tostring|.[0:300])}' "$CASE/spans.json"
    else
      if [ "$sample" = scalar ]; then
        jq -e '[.rows[] | select(.[2]=="user.span_leaf")] | length==1 and (.[0] | .[5].x==20 and .[6]==21)' "$CASE/spans.json" > "$CASE/payload-assertion.json"
      else
        jq -e '[.rows[] | select(.[2]=="user.fail_leaf")] | length==1 and (.[0] | .[3]=="panic_error" and .[5].x==[1] and .[7].error["$class"]=="baml.panics.IndexOutOfBounds" and .[7].error.index==5 and .[7].error.length==1 and ([.[7].stack_trace.frames[].function_name]|index("user.fail_leaf")!=null))' "$CASE/spans.json" > "$CASE/payload-assertion.json"
      fi
      jq -c '.rows[] | select(.[2]|test("span_leaf|span_middle|fail_leaf|Failure"))' "$CASE/spans.json"
    fi
  )
done
for sample in missing complete; do
  for mode in default empty span; do
    diff <(jq -S . "$OUT/off-$sample/prediction.json") <(jq -S . "$OUT/$mode-$sample/prediction.json")
  done
done
echo "All native Iris predictions match the independent oracle and each other."
