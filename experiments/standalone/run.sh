#!/usr/bin/env bash
# Build with verified, matching CLI and pack host; run in a clean offline container.
set -euo pipefail
export LC_ALL=C DO_NOT_TRACK=1
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CLI="$(realpath "$BAML_CLI")"
HOST="$(realpath "$BAML_PACK_HOST")"
OUT="$(realpath -m "$1")"
mkdir -p "$OUT/dist" "$OUT/evidence"
BIN="$OUT/dist/baml-ml"
FIX="$ROOT/migration/artifacts/fixtures"
# Build without source tests or fixture data in the binary.
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT
mkdir -p "$BUILD/baml_src"
cp "$ROOT/migration/baml.toml" "$BUILD/"
for src in "$ROOT"/migration/baml_src/*.baml; do
  case "$(basename "$src")" in *_test.baml) continue;; esac
  cp "$src" "$BUILD/baml_src/"
done
(
  cd "$BUILD"
  BAML_TELEMETRY=off "$CLI" check
  BAML_TELEMETRY=off "$CLI" pack --function predict --host-binary "$HOST" --output "$BIN"
)
test -x "$BIN"
file "$BIN" | tee "$OUT/evidence/binary-format.txt"
ldd "$BIN" | tee "$OUT/evidence/dynamic-libraries.txt"
stat -c '%s' "$BIN" > "$OUT/evidence/binary-size.txt"
cp "$ROOT/LICENSE" "$OUT/dist/LICENSE"
(cd "$OUT/dist"; sha256sum baml-ml > SHA256SUMS)
docker pull ubuntu:24.04
# No repository, source, interpreter SDK, toolchain or credentials mounted.
# Mount each case under its original path so recorded paths remain resolvable.
run_case() {
  local case_dir="$1" telemetry="$2"
  shift 2
  docker run --rm --network none --read-only \
    --user "$(id -u):$(id -g)" --tmpfs /tmp \
    -e HOME="$case_dir/home" -e DO_NOT_TRACK=1 \
    -e BAML_TELEMETRY="$telemetry" \
    -v "$BIN:/opt/baml-ml:ro" -v "$case_dir:$case_dir:rw" \
    -w "$case_dir" ubuntu:24.04 /opt/baml-ml "$@"
}
prepare_case() {
  CASE="$OUT/evidence/$1"
  mkdir -p "$CASE/home"
  cp "$FIX/$2.json" "$CASE/model.json"
  jq -n --arg sample "$2" --slurpfile ref "$FIX/reference.json" --slurpfile a "$CASE/model.json" \
    '{features:$a[0].features,rows:($ref[0][$sample].imputed | .[0][0]=null)}' > "$CASE/input.json"
}
query() {
  (
    cd "$CASE"
    env -i PATH="$PATH" HOME="$CASE/home" DO_NOT_TRACK=1 \
      "$CLI" local recordings query --output-format json "$2"
  ) > "$CASE/$1.json" 2> "$CASE/$1.stderr"
}
for sample in iris diabetes binary; do
  for operation in predict predict-proba; do
    if [ "$sample" = diabetes ] && [ "$operation" = predict-proba ]; then continue; fi
    prepare_case "$sample-$operation" "$sample"
    run_case "$CASE" medium predict --model model.json --input input.json \
      --operation "$operation" --observation full --output-format json \
      > "$CASE/output.json" 2> "$CASE/run.stderr" || { cat "$CASE/run.stderr"; exit 1; }
    jq -e --arg op "$operation" --arg s "$sample" --slurpfile a "$CASE/model.json" --slurpfile ref "$FIX/reference.json" '
      def near($a;$b): if ($b|type)=="array" then (($a|type)=="array" and ($a|length)==($b|length) and ([range(0;($b|length)) as $i|near($a[$i];$b[$i])]|all)) else (($a|type)=="number" and (($a-$b)|fabs)<1e-9) end;
      .schema_version==1 and .model_id==$a[0].model_id and .estimator_kind==$a[0].estimator_kind and .classes==$a[0].classes and .operation==$op and near(.predictions;(if $op=="predict-proba" then $ref[0][$s].probs else $ref[0][$s].prediction end))
    ' "$CASE/output.json"
    query spans "SELECT span_id, parent_span_id, span_name, status, input_args, output_value, context_metadata['baml_ml.model_id'] FROM spans ORDER BY start_time"
    query profiler "SELECT function_name, invocation_count FROM profiler"
    jq -e --arg op "$operation" --arg s "$sample" --slurpfile a "$CASE/model.json" '
      def row($n): [.rows[]|select(.[2]==("user."+$n))]|if length==1 then .[0] else error("missing or duplicate span "+$n) end;
      .outcome.status=="complete" and (
        (if $s=="diabetes" then "PredictLoadedLinear" elif $op=="predict-proba" then "PredictLoadedProba" else "PredictLoadedLabels" end) as $root |
        row($root) as $p | row("ObservedImpute") as $i | row("ObservedScale") as $z |
        row("ModelLoaded")[4].artifact.model_id==$a[0].model_id and
        ([$p,$i,$z]|all(.[6]==$a[0].model_id and .[3]=="return")) and
        $i[1]==$p[0] and $z[1]==$p[0] and $i[4].x[0][0]["$float"]=="NaN" and
        (if $s=="diabetes" then row("ObservedLinear")[1]==$p[0] else row("ObservedDecisionScores")[1]==$p[0] and (if $op=="predict-proba" then row("ObservedSoftmax")[1]==$p[0] else row("ObservedLabels")[1]==$p[0] end) end)
      )
    ' "$CASE/spans.json" || { cat "$CASE/spans.json"; exit 1; }
    jq -e '
      def calls($n): ([.rows[]|select(.[0]==("user."+$n))|.[1]]|add // 0);
      .outcome.status=="complete" and calls("DecodeFittedArtifact")==1 and calls("ArtifactError")==1 and calls("ModelLoaded")==1
    ' "$CASE/profiler.json" || { cat "$CASE/profiler.json"; exit 1; }
    echo "Packed offline reference and traces passed: $sample $operation"
  done
done
# Default metadata-only semantic mode, plus explicit native engine off.
for mode in semantic timing off; do
  prepare_case "mode-$mode" iris
  telemetry=medium
  args=(predict --model model.json --input input.json --operation predict-proba --output-format json)
  if [ "$mode" = timing ]; then args+=(--observation timing); fi
  if [ "$mode" = off ]; then telemetry=off; args+=(--observation full); fi
  run_case "$CASE" "$telemetry" "${args[@]}" > "$CASE/output.json" 2> "$CASE/run.stderr"
  diff <(jq -S . "$OUT/evidence/iris-predict-proba/output.json") <(jq -S . "$CASE/output.json")
  if [ "$mode" = off ]; then
    test -z "$(find "$CASE/home" -type f -name '*.btel' -print -quit)"
  else
    query spans "SELECT span_name, input_args, output_value FROM spans"
    if [ "$mode" = semantic ]; then
      jq -e '[.rows[]|select(.[0]=="user.PredictLoadedProba" or .[0]=="user.ObservedImpute" or .[0]=="user.ObservedScale" or .[0]=="user.ObservedDecisionScores" or .[0]=="user.ObservedSoftmax")]|length==5 and all(.[1]==null and .[2]==null)' "$CASE/spans.json"
    else
      jq -e '[.rows[]|select(.[0]=="user.PredictLoadedProba" or .[0]=="user.ObservedImpute")]|length==0' "$CASE/spans.json"
    fi
  fi
  echo "Packed mode passed: $mode"
done
for error in malformed wrong-order ragged empty nonnumeric invalid-model wrong-operation; do
  prepare_case "error-$error" iris
  args=(predict --model model.json --input input.json --output-format json)
  case "$error" in
    malformed) printf '{' > "$CASE/input.json";;
    wrong-order) jq '.features|=reverse' "$CASE/input.json" > "$CASE/tmp"; mv "$CASE/tmp" "$CASE/input.json";;
    ragged) jq '.rows[1]=[1]' "$CASE/input.json" > "$CASE/tmp"; mv "$CASE/tmp" "$CASE/input.json";;
    empty) jq '.rows=[]' "$CASE/input.json" > "$CASE/tmp"; mv "$CASE/tmp" "$CASE/input.json";;
    nonnumeric) jq '.rows[0][0]="1"' "$CASE/input.json" > "$CASE/tmp"; mv "$CASE/tmp" "$CASE/input.json";;
    invalid-model) jq '.schema_version=2' "$CASE/model.json" > "$CASE/tmp"; mv "$CASE/tmp" "$CASE/model.json";;
    wrong-operation) args+=(--operation fit);;
  esac
  rc=0
  run_case "$CASE" off "${args[@]}" > "$CASE/output.json" 2> "$CASE/run.stderr" || rc=$?
  test "$rc" -ne 0
  # Docker failure/launch failure must not masquerade as an application rejection.
  test "$rc" -lt 125
  if jq -e 'has("predictions")' "$CASE/output.json" >/dev/null 2>&1; then
    echo "Unexpected successful result for $error"; exit 1
  fi
  echo "Packed rejection passed: $error (exit $rc)"
done
echo "Standalone offline smoke passed"
