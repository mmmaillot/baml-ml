#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C DO_NOT_TRACK=1
unset BAML_TELEMETRY BOUNDARY_API_KEY BOUNDARY_PROJECT
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CLI="$(realpath "$BAML_CLI")"
OUT="$(realpath -m "$1")"
mkdir -p "$OUT"
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
    cp "$ROOT"/migration/artifacts/fixtures/*.json "$PROJECT/artifacts/fixtures/"
    for src in "$ROOT"/migration/baml_src/*.baml; do
      case "$(basename "$src")" in artifact_test.baml) ;; *_test.baml|test_*.baml|example_checks.baml|generated_examples.baml) continue;; esac
      cp "$src" "$PROJECT/baml_src/"
    done
    cd "$PROJECT"
    printf '[package]\nname = "imported-inference-probe"\n' > baml.toml
    fn=ImportedIrisProbe
    outer=PredictLoadedProba
    model=LogRegPredictProbaCore
    if [ "$sample" = binary ]; then fn=ImportedBinaryProbe; fi
    if [ "$sample" = diabetes ]; then fn=ImportedDiabetesProbe; outer=PredictLoadedLinear; model=LinearRegressionPredictCore; fi
    for target in "$outer" ImputerTransformCore ScalerTransformCore "$model"; do
      file="$(grep -l "^function $target(" baml_src/*.baml)"
      test "$(printf '%s\n' "$file" | wc -l)" -eq 1
      sed -i "/^function $target(/i /// baml:\$trace=trace.span" "$file"
    done
    BAML_TELEMETRY=off "$CLI" check > "$CASE/check.log" 2>&1 || { cat "$CASE/check.log"; exit 1; }
    "$CLI" run "$fn" --output-format json > "$CASE/result.json" 2> "$CASE/run.stderr" || { cat "$CASE/run.stderr"; exit 1; }
    "$CLI" local recordings query --output-format json "SELECT span_id, parent_span_id, span_name, status, duration, input_args, output_value FROM spans ORDER BY start_time" > "$CASE/spans.json" 2> "$CASE/query.stderr"
    jq -e --arg sample "$sample" --arg outer "$outer" --arg model "$model" --slurpfile ref "$PROJECT/artifacts/fixtures/reference.json" --slurpfile artifact "$PROJECT/artifacts/fixtures/$sample.json" --slurpfile result "$CASE/result.json" '
      def row($n): [.rows[]|select(.[2]==("user."+$n))]|if length==1 then .[0] else error("missing or duplicate span "+$n) end;
      def near($a;$b): if ($b|type)=="array" then (($a|type)=="array" and ($a|length)==($b|length) and ([range(0;($b|length)) as $i|near($a[$i];$b[$i])]|all)) else (($a|type)=="number" and (($a-$b)|fabs)<1e-9) end;
      .outcome.status=="complete" and (
        $ref[0][$sample] as $e |
        row($outer) as $p | row("ImputerTransformCore") as $i | row("ScalerTransformCore") as $s | row($model) as $m |
        ([$i,$s,$m]|all(.[1]==$p[0] and .[3]=="return" and .[4]>0)) and
        $p[5].model_id==$artifact[0].model_id and
        $i[5].x[0][0]["$float"]=="NaN" and
        near($i[6].x;$e.imputed) and near($s[5].x;$e.imputed) and near($s[6].x;$e.scaled) and near($m[5].x;$e.scaled) and
        (if $sample=="diabetes" then near($m[6].value;$e.prediction) and near($p[6].value;$e.prediction) and near($result[0].value;$e.prediction)
         else near($m[6].probs;$e.probs) and near($p[6].probs;$e.probs) and near($result[0].probs;$e.probs) end)
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
  "$CLI" local recordings query --output-format json "SELECT function_name, invocation_count FROM profiler" > "$OUT/repeat/profiler.json"
  jq -e '
    def calls($n): ([.rows[]|select(.[0]==("user."+$n))|.[1]]|add // 0);
    .outcome.status=="complete" and calls("ReadFittedArtifact")==0 and calls("DecodeFittedArtifact")==1 and calls("ArtifactError")==1 and calls("ArtifactClassification")==1 and calls("PredictLoadedProba")==3 and calls("ImputerTransformCore")==3 and calls("ScalerTransformCore")==3 and calls("LogRegPredictProbaCore")==3
  ' "$OUT/repeat/profiler.json" || { cat "$OUT/repeat/profiler.json"; exit 1; }
  jq -c '[.rows[] | select(.[0]|test("ArtifactError$|ArtifactClassification$|DecodeFittedArtifact$|PredictLoadedProba$|ImputerTransformCore$|ScalerTransformCore$|LogRegPredictProbaCore$"))]' "$OUT/repeat/profiler.json"
)
