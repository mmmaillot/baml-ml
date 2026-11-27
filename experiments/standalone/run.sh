#!/usr/bin/env bash
# Build with verified, matching native CLI and pack host; exercise in isolation.
set -euo pipefail
export LC_ALL=C DO_NOT_TRACK=1
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CLI="$(realpath "$BAML_CLI")"
HOST="$(realpath "$BAML_PACK_HOST")"
mkdir -p "$1"
OUT="$(cd "$1" && pwd -P)"
mkdir -p "$OUT/dist" "$OUT/evidence"
OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS/$ARCH" in
  Linux/x86_64) PLATFORM=linux-x86_64;;
  Linux/aarch64) PLATFORM=linux-arm64;;
  Darwin/arm64) PLATFORM=macos-arm64;;
  *) echo "Unsupported native platform: $OS/$ARCH" >&2; exit 1;;
esac
printf '%s\n' "$PLATFORM" > "$OUT/evidence/platform.txt"
uname -a > "$OUT/evidence/environment.txt"
BIN="$OUT/dist/baml-ml"
FIX="$ROOT/artifacts/fixtures"
# Build without source tests or fixture data in the binary.
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT
mkdir -p "$BUILD/baml_src"
cp "$ROOT/baml.toml" "$BUILD/"
for src in "$ROOT"/baml_src/*.baml; do
  case "$(basename "$src")" in *_test.baml) continue;; esac
  cp "$src" "$BUILD/baml_src/"
done
(
  cd "$BUILD"
  BAML_TELEMETRY=off "$CLI" check
  BAML_TELEMETRY=off "$CLI" pack --function predict --function shadow --host-binary "$HOST" --output "$BIN"
)
test -x "$BIN"
file "$BIN" | tee "$OUT/evidence/binary-format.txt"
case "$PLATFORM" in
  linux-x86_64) grep -E 'ELF.*x86-64' "$OUT/evidence/binary-format.txt";;
  linux-arm64) grep -Ei 'ELF.*(aarch64|ARM aarch64)' "$OUT/evidence/binary-format.txt";;
  macos-arm64) grep -E 'Mach-O.*arm64' "$OUT/evidence/binary-format.txt";;
esac
if [ "$OS" = Darwin ]; then
  otool -L "$BIN" | tee "$OUT/evidence/dynamic-libraries.txt"
  otool -l "$BIN" > "$OUT/evidence/macho-load-commands.txt"
  sw_vers >> "$OUT/evidence/environment.txt"
  codesign --verify --verbose=2 "$BIN"
  stat -f '%z' "$BIN" > "$OUT/evidence/binary-size.txt"
else
  ldd "$BIN" | tee "$OUT/evidence/dynamic-libraries.txt"
  stat -c '%s' "$BIN" > "$OUT/evidence/binary-size.txt"
fi
cp "$ROOT/LICENSE" "$OUT/dist/LICENSE"
(cd "$OUT/dist"; if [ "$OS" = Darwin ]; then shasum -a 256 baml-ml > SHA256SUMS; else sha256sum baml-ml > SHA256SUMS; fi)
if [ "$OS" = Linux ]; then
  docker pull ubuntu:24.04
else
  # Native macOS process, fresh home and no inherited credentials or PATH to
  # developer tools. Deny network via the platform sandbox; no Rosetta/Docker.
  test -x /usr/bin/sandbox-exec
fi
# No repository, source, interpreter SDK, toolchain or credentials mounted.
# Mount each case under its original path so recorded paths remain resolvable.
run_case() {
  local case_dir="$1" telemetry="$2"
  shift 2
  if [ "$OS" = Darwin ]; then
    (
      cd "$case_dir"
      /usr/bin/env -i HOME="$case_dir/home" TMPDIR="$case_dir/tmp" PATH=/usr/bin:/bin LC_ALL=C DO_NOT_TRACK=1 BAML_TELEMETRY="$telemetry" \
        /usr/bin/sandbox-exec -p '(version 1) (allow default) (deny network*)' "$BIN" "$@"
    )
    return
  fi
  docker run --rm --network none --read-only \
    --user "$(id -u):$(id -g)" --tmpfs /tmp \
    -e HOME="$case_dir/home" -e DO_NOT_TRACK=1 \
    -e BAML_TELEMETRY="$telemetry" \
    -v "$BIN:/opt/baml-ml:ro" -v "$case_dir:$case_dir:rw" \
    -w "$case_dir" ubuntu:24.04 /opt/baml-ml "$@"
}
prepare_case() {
  CASE="$OUT/evidence/$1"
  mkdir -p "$CASE/home" "$CASE/tmp"
  cp "$FIX/$2.json" "$CASE/model.json"
  jq -n --arg sample "$2" --slurpfile ref "$FIX/reference.json" --slurpfile a "$CASE/model.json" \
    '{features:$a[0].features,rows:($ref[0][$sample].imputed | .[0][0]=null)}' > "$CASE/input.json"
}
query() {
  (
    # Packed hosts record under HOME/.baml/btel, while the development
    # query command resolves its index from a project root, not HOME.
    # This manifest is created by the analyzer AFTER the offline execution.
    printf '[package]\nname = "packed-recording-analysis"\n' > "$CASE/home/baml.toml"
    cd "$CASE/home"
    find .baml -type f -name '*.btel' -print >&2
    env -i PATH="$PATH" HOME="$CASE/home" DO_NOT_TRACK=1 \
      "$CLI" local recordings query --project-dir "$CASE/home" --output-format json "$2"
  ) > "$CASE/$1.json" 2> "$CASE/$1.stderr" || { cat "$CASE/$1.stderr"; return 1; }
  if ! jq -e '.outcome.source_missing == false' "$CASE/$1.json" >/dev/null; then
    cat "$CASE/$1.stderr"; find "$CASE/home" -type f; return 1
  fi
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
# Tree smoke: every qualified native platform, all capture modes, no Python on target.
for mode in semantic full timing off; do
  CASE="$OUT/evidence/tree-$mode"
  mkdir -p "$CASE/home" "$CASE/tmp"
  cp "$FIX/tree-smoke.json" "$CASE/model.json"
  printf '%s\n' '{"features":["x"],"rows":[[null],[-1],[1]]}' > "$CASE/input.json"
  telemetry=medium
  capture="$mode"
  if [ "$mode" = off ]; then telemetry=off; capture=full; fi
  run_case "$CASE" "$telemetry" predict --model model.json --input input.json --operation predict-proba --observation "$capture" --output-format json > "$CASE/output.json" 2> "$CASE/run.stderr"
  jq -e '.schema_version==1 and .estimator_kind=="decision_tree_classifier" and .classes==[-3,7] and .predictions==[[0.5,0.5],[0.5,0.5],[0,1]]' "$CASE/output.json"
  if [ "$mode" = off ]; then
    test -z "$(find "$CASE/home" -type f -name '*.btel' -print -quit)"
  else
    query spans "SELECT span_name, input_args, output_value FROM spans"
    if [ "$mode" = full ]; then
      jq -e '[.rows[]|select(.[0]=="user.ObservedTreeTraversal")]|length==1 and .[0][2].paths==[[0,1],[0,1],[0,2]] and .[0][2].leaves==[1,1,2]' "$CASE/spans.json"
    elif [ "$mode" = semantic ]; then
      jq -e '[.rows[]|select(.[0]=="user.ObservedTreeTraversal")]|length==1 and .[0][1]==null and .[0][2]==null' "$CASE/spans.json"
    else
      jq -e '[.rows[]|select(.[0]=="user.ObservedTreeTraversal")]|length==0' "$CASE/spans.json"
    fi
  fi
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
    wrong-order) jq '.features|=reverse' "$CASE/input.json" > "$CASE/mutation.json"; mv "$CASE/mutation.json" "$CASE/input.json";;
    ragged) jq '.rows[1]=[1]' "$CASE/input.json" > "$CASE/mutation.json"; mv "$CASE/mutation.json" "$CASE/input.json";;
    empty) jq '.rows=[]' "$CASE/input.json" > "$CASE/mutation.json"; mv "$CASE/mutation.json" "$CASE/input.json";;
    nonnumeric) jq '.rows[0][0]="1"' "$CASE/input.json" > "$CASE/mutation.json"; mv "$CASE/mutation.json" "$CASE/input.json";;
    invalid-model) jq '.schema_version=2' "$CASE/model.json" > "$CASE/mutation.json"; mv "$CASE/mutation.json" "$CASE/model.json";;
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
# Both selective entry points get the same isolation as the reference cases.
prepare_case "mode-adaptive" iris
printf '%s\n' '{"max_escalated_rows":1}' > "$CASE/policy.json"
run_case "$CASE" medium predict --model model.json --input input.json --operation predict-proba \
  --observation adaptive --policy policy.json --output-format json > "$CASE/output.json" 2> "$CASE/run.stderr"
diff <(jq -S . "$OUT/evidence/iris-predict-proba/output.json") <(jq -S . "$CASE/output.json")
query spans "SELECT span_name, context_metadata['baml_ml.row_index'] FROM spans WHERE span_name = 'user.EscalatedRow'"
jq -e '.rows==[["user.EscalatedRow",0]]' "$CASE/spans.json"

prepare_case "shadow-iris" iris
printf '%s\n' '{"max_escalated_rows":1}' > "$CASE/policy.json"
run_case "$CASE" medium shadow --champion model.json --challenger model.json --input input.json \
  --operation predict-proba --policy policy.json --output-format json > "$CASE/output.json" 2> "$CASE/run.stderr"
jq -e --slurpfile expected "$OUT/evidence/iris-predict-proba/output.json" '
  .predictions==$expected[0].predictions and .summary.challenger_ok and
  .summary.escalated_rows==1 and .summary.label_disagreements==0
' "$CASE/output.json"
query spans "SELECT span_name, context_metadata['baml_ml.row_index'] FROM spans WHERE span_name = 'user.ShadowEscalatedRow'"
jq -e '.rows==[["user.ShadowEscalatedRow",0]]' "$CASE/spans.json"
# The generated CLI must accept both entry points with an omitted policy flag.
for command in predict shadow; do
  CASE="$OUT/evidence/default-policy-$command"
  mkdir -p "$CASE/home" "$CASE/tmp"
  cp "$FIX/tree-smoke.json" "$CASE/model.json"
  printf '%s\n' '{"features":["x"],"rows":[[null],[1]]}' > "$CASE/input.json"
  if [ "$command" = predict ]; then
    args=(predict --model model.json --observation adaptive)
    replay=EscalatedRow
  else
    args=(shadow --champion model.json --challenger model.json)
    replay=ShadowEscalatedRow
  fi
  run_case "$CASE" medium "${args[@]}" --input input.json --output-format json \
    > "$CASE/output.json" 2> "$CASE/run.stderr"
  jq -e '.predictions==[-3,7]' "$CASE/output.json"
  query spans "SELECT context_metadata['baml_ml.row_index'] FROM spans WHERE span_name = 'user.$replay'"
  jq -e '.rows==[[0]]' "$CASE/spans.json"
done
# Schema v3 heterogeneous RandomForest pipeline smoke: predict, adaptive
# replay and shadow on the packed offline binary, checked against the
# committed sklearn 1.8.0 oracle.
V3FIX="$FIX/v3"
jq -e '."v3-ct-hetero-forest" | has("input") and has("predict_proba")' "$V3FIX/oracle.json" >/dev/null
for mode in predict adaptive shadow; do
  CASE="$OUT/evidence/v3-ct-forest-$mode"
  mkdir -p "$CASE/home" "$CASE/tmp"
  cp "$V3FIX/v3-ct-hetero-forest.json" "$CASE/model.json"
  jq '."v3-ct-hetero-forest".input' "$V3FIX/oracle.json" > "$CASE/input.json"
  case "$mode" in
    predict) args=(predict --model model.json --input input.json --operation predict-proba --observation full);;
    adaptive) args=(predict --model model.json --input input.json --operation predict-proba --observation adaptive);;
    shadow) args=(shadow --champion model.json --challenger model.json --input input.json --operation predict-proba);;
  esac
  run_case "$CASE" medium "${args[@]}" --output-format json > "$CASE/output.json" 2> "$CASE/run.stderr" || { cat "$CASE/run.stderr"; exit 1; }
  jq -e --arg mode "$mode" --slurpfile oracle "$V3FIX/oracle.json" '
    def near($a;$b): if ($b|type)=="array" then (($a|type)=="array" and ($a|length)==($b|length) and ([range(0;($b|length)) as $i|near($a[$i];$b[$i])]|all)) else (($a|type)=="number" and (($a-$b)|fabs)<1e-9) end;
    .schema_version==1 and .estimator_kind=="random_forest_classifier" and .classes==[0,1] and
    near(if $mode=="shadow" then .predictions else .predictions end; $oracle[0]["v3-ct-hetero-forest"].predict_proba) and
    (if $mode=="shadow" then .summary.challenger_ok and .summary.triggers.missing_input==2 and .summary.triggered_rows==2 else true end)
  ' "$CASE/output.json" || { cat "$CASE/output.json"; exit 1; }
  query spans "SELECT span_name, parent_span_id, context_metadata['baml_ml.step_index'], context_metadata['baml_ml.step_kind'], context_metadata['baml_ml.branch_name'], context_metadata['baml_ml.shadow_role'], context_metadata['baml_ml.row_index'] FROM spans ORDER BY start_time"
  if [ "$mode" = predict ]; then
    jq -e '
      def rows($n): [.rows[]|select(.[0]==("user."+$n))];
      (rows("PipelineModelLoaded")|length==1) and
      (rows("ObservedColumnTransformer") as $ct |
        ($ct|length==1) and ($ct[0][2]==0) and ($ct[0][3]=="column_transformer") and
        (rows("ObservedColumnTransformerBranch") as $br |
          ($br|length==2) and ([$br[]|select(.[2]==0 and .[4]!=null and .[1]==$ct[0][0])]|length==2)) and
        (rows("ObservedImputeStep") as $im | ($im|length==2) and ([$im[]|select(.[3]=="simple_imputer" and .[2]==0)]|length==2)) and
        (rows("ObservedScaleStep") as $sc | ($sc|length==1) and ($sc[0][3]=="standard_scaler" and $sc[0][2]==0)) and
        (rows("ObservedOneHotEncode") as $oh | ($oh|length==1) and ($oh[0][3]=="one_hot_encoder" and $oh[0][2]==0)) and
        (rows("ObservedForestTraversal")|length==1))
    ' "$CASE/spans.json" || { cat "$CASE/spans.json"; exit 1; }
  fi
  if [ "$mode" = adaptive ]; then
    jq -e '[.rows[]|select(.[0]=="user.EscalatedRow")]|length==2 and all(.[6]!=null)' "$CASE/spans.json" || { cat "$CASE/spans.json"; exit 1; }
  fi
  if [ "$mode" = shadow ]; then
    jq -e '[.rows[]|select(.[0]=="user.ShadowEscalatedRow")]|length>=1' "$CASE/spans.json" || { cat "$CASE/spans.json"; exit 1; }
    jq -e '[.rows[]|select(.[0]=="user.ObservedColumnTransformer" and .[5]!=null)]|length==2' "$CASE/spans.json" || { cat "$CASE/spans.json"; exit 1; }
  fi
  echo "Packed v3 heterogeneous forest passed: $mode"
done
echo "Packed adaptive and shadow passed"
tar -czf "$OUT/baml-ml-$PLATFORM.tar.gz" -C "$OUT/dist" baml-ml LICENSE SHA256SUMS
echo "Standalone offline smoke passed: $PLATFORM"
