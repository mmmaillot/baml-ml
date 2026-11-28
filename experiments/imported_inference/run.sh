#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C DO_NOT_TRACK=1
unset BAML_TELEMETRY BOUNDARY_API_KEY BOUNDARY_PROJECT
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CLI="$(realpath "$BAML_CLI")"
mkdir -p "$1"
OUT="$(cd "$1" && pwd -P)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
for sample in iris diabetes binary; do
  (
    CASE="$OUT/$sample"
    PROJECT="$WORK/$sample"
    mkdir -p "$CASE" "$PROJECT/baml_src" "$PROJECT/artifacts/fixtures"
    export HOME="$WORK/$sample-home" BAML_HOME="$WORK/$sample-home/baml"
    export XDG_CONFIG_HOME="$HOME/config" XDG_DATA_HOME="$HOME/data" XDG_CACHE_HOME="$HOME/cache"
    mkdir -p "$HOME"
    cp "$ROOT"/artifacts/fixtures/*.json "$PROJECT/artifacts/fixtures/"
    "$ROOT/scripts/stage_runtime.sh" "$PROJECT/baml_src"
    cp "$ROOT/baml_src/artifact_test.baml" "$PROJECT/baml_src/"
    cd "$PROJECT"
    printf '[package]\nname = "imported-inference-probe"\n' > baml.toml
    fn=ImportedIrisProbe
    outer=PredictLoadedProba
    model=ObservedSoftmax
    if [ "$sample" = binary ]; then fn=ImportedBinaryProbe; fi
    if [ "$sample" = diabetes ]; then fn=ImportedDiabetesProbe; outer=PredictLoadedLinear; model=ObservedLinear; fi
    BAML_TELEMETRY=off "$CLI" check > "$CASE/check.log" 2>&1 || { cat "$CASE/check.log"; exit 1; }
    "$CLI" run "$fn" --output-format json > "$CASE/result.json" 2> "$CASE/run.stderr" || { cat "$CASE/run.stderr"; exit 1; }
    "$CLI" local recordings query --output-format json "SELECT span_id AS span_id, parent_span_id AS parent_span_id, span_name AS span_name, status AS status, duration AS duration, input_args AS input_args, output_value AS output_value, error_value AS error_value, context_distinct_id AS context_distinct_id, context_metadata['baml_ml.model_id'] AS model_id, context_metadata['baml_ml.contract_version'] AS contract_version FROM spans ORDER BY start_time" > "$CASE/spans.json" 2> "$CASE/query.stderr"
    jq -e -L "$ROOT/scripts" --arg sample "$sample" --arg outer "$outer" --arg model "$model" --slurpfile ref "$PROJECT/artifacts/fixtures/reference.json" --slurpfile artifact "$PROJECT/artifacts/fixtures/$sample.json" --slurpfile result "$CASE/result.json" '
      import "btel_jq" as btel;
      def row($n): [btel::named_rows[]|select(.span_name==("user."+$n))]|if length==1 then .[0] else error("missing or duplicate span "+$n) end;
      def near($a;$b): if ($b|type)=="array" then (($a|type)=="array" and ($a|length)==($b|length) and ([range(0;($b|length)) as $i|near($a[$i];$b[$i])]|all)) else (($a|type)=="number" and (($a-$b)|fabs)<1e-9) end;
      .outcome.status=="complete" and (
        $ref[0][$sample] as $e |
        row($outer) as $p | row("ObservedImpute") as $i | row("ObservedScale") as $s | row($model) as $m |
        ([$i,$s,$m]|all(.parent_span_id==$p.span_id and .status=="return" and .duration>0 and .model_id==$artifact[0].model_id and .contract_version==1)) and
        (row("ModelLoaded").input_args.artifact as $loaded | $artifact[0] | to_entries | all(. as $entry | $loaded[$entry.key]==$entry.value)) and
        ([$p,$i,$s,$m]|all(.model_id==$artifact[0].model_id and (.input_args|[..|objects|keys[]]|index("weights")==null and index("means")==null and index("coefficients")==null))) and
        $p.input_args.model_id==$artifact[0].model_id and
        $i.input_args.x[0][0]["$float"]=="NaN" and
        near($i.output_value.x;$e.imputed) and near($s.input_args.x;$e.imputed) and near($s.output_value.x;$e.scaled) and
        (if $sample=="diabetes" then near($m.input_args.x;$e.scaled) else row("ObservedDecisionScores") as $d | $d.parent_span_id==$p.span_id and $d.model_id==$artifact[0].model_id and near($d.input_args.x;$e.scaled) and near($d.output_value;$m.input_args.scores) and ($d.output_value|length)==2 end) and
        (if $sample=="diabetes" then near($m.output_value.value;$e.prediction) and near($p.output_value.value;$e.prediction) and near($result[0].value;$e.prediction)
         else near($m.output_value.probs;$e.probs) and near($p.output_value.probs;$e.probs) and near($result[0].probs;$e.probs) end)
      )
    ' "$CASE/spans.json" || { cat "$CASE/spans.json"; exit 1; }
    jq -c '{result:.,sample:"'"$sample"'"}' "$CASE/result.json"
  )
done

# Independent recording: one factory load, two single predictions and a batch.
(
  # Isolate project-local recordings too; HOME alone does not isolate BTEL.
  mkdir -p "$WORK/repeat-project"
  cp -R "$WORK/iris/baml_src" "$WORK/iris/artifacts" "$WORK/iris/baml.toml" "$WORK/repeat-project/"
  cd "$WORK/repeat-project"
  export HOME="$WORK/repeat-home" BAML_HOME="$WORK/repeat-home/baml"
  export XDG_CONFIG_HOME="$HOME/config" XDG_DATA_HOME="$HOME/data" XDG_CACHE_HOME="$HOME/cache"
  mkdir -p "$HOME" "$OUT/repeat"
  "$CLI" run RepeatImportedIrisProbe --output-format json > "$OUT/repeat/result.json" 2> "$OUT/repeat/run.stderr" || { cat "$OUT/repeat/run.stderr"; exit 1; }
  "$CLI" local recordings query --output-format json "SELECT function_name AS function_name, invocation_count AS invocation_count FROM profiler" > "$OUT/repeat/profiler.json"
  jq -e -L "$ROOT/scripts" '
    import "btel_jq" as btel;
    def calls($n): ([btel::named_rows[]|select(.function_name==("user."+$n))|.invocation_count]|add // 0);
    .outcome.status=="complete" and calls("ReadFittedArtifact")==0 and calls("DecodeFittedArtifact")==1 and calls("ArtifactError")==1 and calls("ArtifactClassification")==1 and calls("PredictLoadedProba")==3 and calls("ImputerTransformCore")==3 and calls("ScalerTransformCore")==3 and calls("ObservedDecisionScores")==3 and calls("ObservedSoftmax")==3 and calls("ModelLoaded")==1
  ' "$OUT/repeat/profiler.json" || { cat "$OUT/repeat/profiler.json"; exit 1; }
  jq -c -L "$ROOT/scripts" 'import "btel_jq" as btel; [btel::named_rows[] | select(.function_name|test("ArtifactError$|ArtifactClassification$|DecodeFittedArtifact$|PredictLoadedProba$|ImputerTransformCore$|ScalerTransformCore$|ObservedDecisionScores$|ObservedSoftmax$|ModelLoaded$"))]' "$OUT/repeat/profiler.json"
)

# Run the actual runtime without editing its source or injecting trace directives.
for mode in semantic timing full off labels failure load-failure; do
  (
    mkdir -p "$WORK/mode-$mode" "$OUT/mode-$mode"
    cp -R "$WORK/iris/baml_src" "$WORK/iris/artifacts" "$WORK/iris/baml.toml" "$WORK/mode-$mode/"
    cd "$WORK/mode-$mode"
    export HOME="$WORK/mode-$mode-home" BAML_HOME="$WORK/mode-$mode-home/baml"
    export XDG_CONFIG_HOME="$HOME/config" XDG_DATA_HOME="$HOME/data" XDG_CACHE_HOME="$HOME/cache"
    mkdir -p "$HOME"
    CASE="$OUT/mode-$mode"
    fn=ObservationFullProbe
    case "$mode" in
      semantic) fn=ObservationSemanticProbe;;
      timing) fn=ObservationTimingProbe;;
      labels) fn=ObservationLabelProbe;;
      failure) fn=ObservationFailureProbe;;
      load-failure) fn=ObservationLoadFailureProbe;;
      off) export BAML_TELEMETRY=off;;
    esac
    "$CLI" run "$fn" --output-format json > "$CASE/result.json" 2> "$CASE/run.stderr" || { cat "$CASE/run.stderr"; exit 1; }
    "$CLI" local recordings query --output-format json "SELECT span_id AS span_id, parent_span_id AS parent_span_id, span_name AS span_name, status AS status, duration AS duration, input_args AS input_args, output_value AS output_value, error_value AS error_value, context_distinct_id AS context_distinct_id, context_metadata['baml_ml.model_id'] AS model_id, context_metadata['baml_ml.contract_version'] AS contract_version FROM spans ORDER BY start_time" > "$CASE/spans.json"
    "$CLI" local recordings query --output-format json "SELECT function_name AS function_name, invocation_count AS invocation_count FROM profiler" > "$CASE/profiler.json"
    if [ "$mode" = off ]; then
      jq -e '.rows==[]' "$CASE/spans.json" "$CASE/profiler.json"
    elif [ "$mode" = timing ]; then
      jq -e -L "$ROOT/scripts" 'import "btel_jq" as btel; [btel::named_rows[]|select(.span_name|test("user\\.(ModelLoaded|PredictLoadedProba|ObservedImpute|ObservedScale|ObservedDecisionScores|ObservedSoftmax)$"))]|length==0' "$CASE/spans.json"
      jq -e -L "$ROOT/scripts" 'import "btel_jq" as btel; [btel::named_rows[]|select(.function_name=="user.PredictLoadedProba" and .invocation_count==1)]|length==1' "$CASE/profiler.json"
    elif [ "$mode" = semantic ] || [ "$mode" = full ]; then
      jq -e -L "$ROOT/scripts" --arg mode "$mode" '
        import "btel_jq" as btel;
        [btel::named_rows[]|select(.span_name|test("user\\.(PredictLoadedProba|ObservedImpute|ObservedScale|ObservedDecisionScores|ObservedSoftmax)$"))] |
        length==5 and all(.context_distinct_id=="request-probe" and .model_id!=null and .contract_version==1 and (if $mode=="semantic" then .input_args==null and .output_value==null and .error_value==null else .input_args!=null and .output_value!=null end))
      ' "$CASE/spans.json"
      jq -e -L "$ROOT/scripts" --arg mode "$mode" 'import "btel_jq" as btel; [btel::named_rows[]|select(.span_name=="user.ModelLoaded")]|length==1 and (.[0]|if $mode=="semantic" then .input_args==null and .output_value==null else .input_args.artifact.model_id!=null end)' "$CASE/spans.json"
    elif [ "$mode" = labels ]; then
      jq -e -L "$ROOT/scripts" 'import "btel_jq" as btel; [btel::named_rows[]|select(.span_name=="user.ObservedLabels")]|length==1 and .[0].output_value.value==[1,1]' "$CASE/spans.json"
      jq -e -L "$ROOT/scripts" 'import "btel_jq" as btel; [btel::named_rows[]|select(.span_name=="user.ObservedSoftmax")]|length==0' "$CASE/spans.json"
    elif [ "$mode" = failure ]; then
      jq -e -L "$ROOT/scripts" '
        import "btel_jq" as btel;
        def roots: [btel::named_rows[]|select(.span_name=="user.PredictLoadedProba")];
        roots|length==3 and all(.status!="return" and .error_value!=null) and
        ([.[]|.error_value|..|objects|.stage? // empty]|sort)==["imputation","input_validation","standardization"]
      ' "$CASE/spans.json" || { cat "$CASE/spans.json"; exit 1; }
      jq -e -L "$ROOT/scripts" 'import "btel_jq" as btel; [btel::named_rows[]|select(.span_name=="user.ObservedImpute" and .status!="return")]|length==1' "$CASE/spans.json"
      jq -e -L "$ROOT/scripts" 'import "btel_jq" as btel; [btel::named_rows[]|select(.span_name=="user.ObservedScale" and .status!="return")]|length==1' "$CASE/spans.json"
      jq -e -L "$ROOT/scripts" 'import "btel_jq" as btel; [btel::named_rows[]|select(.span_name=="user.ObservedDecisionScores")]|length==0' "$CASE/spans.json"
    elif [ "$mode" = load-failure ]; then
      jq -e -L "$ROOT/scripts" 'import "btel_jq" as btel; [btel::named_rows[]|select(.span_name=="user.PrepareArtifact")]|length==1 and (.[0]|.status!="return" and .input_args==null and .output_value==null and .error_value!=null)' "$CASE/spans.json"
      jq -e -L "$ROOT/scripts" 'import "btel_jq" as btel; [btel::named_rows[]|select(.span_name=="user.ModelLoaded")]|length==0' "$CASE/spans.json"
    fi
    echo "Observation contract mode $mode passed"
  )
done
for mode in semantic timing off; do
  diff <(jq -S . "$OUT/mode-full/result.json") <(jq -S . "$OUT/mode-$mode/result.json")
done
