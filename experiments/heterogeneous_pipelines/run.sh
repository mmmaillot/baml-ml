#!/usr/bin/env bash
# BTEL SQL verification for schema v3 heterogeneous pipelines + RandomForest.
# Runs unmodified runtime sources plus probes.baml in fresh projects and
# asserts span inventory, hierarchy, metadata, redaction and replay semantics.
set -euo pipefail
export LC_ALL=C DO_NOT_TRACK=1
unset BAML_TELEMETRY BOUNDARY_API_KEY BOUNDARY_PROJECT
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CLI="$(realpath "$BAML_CLI")"
mkdir -p "$1"
OUT="$(cd "$1" && pwd -P)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

PROJECT="$WORK/project"
mkdir -p "$PROJECT/baml_src" "$PROJECT/artifacts/fixtures"
"$ROOT/scripts/stage_runtime.sh" "$PROJECT/baml_src"
cp "$ROOT/experiments/heterogeneous_pipelines/probes.baml" "$PROJECT/baml_src/"
cp "$ROOT"/artifacts/fixtures/v3/*.json "$PROJECT/artifacts/fixtures/v3/" 2>/dev/null || {
  mkdir -p "$PROJECT/artifacts/fixtures/v3"
  cp "$ROOT"/artifacts/fixtures/v3/*.json "$PROJECT/artifacts/fixtures/v3/"
}
printf '[package]\nname = "heterogeneous-pipelines-probe"\n' > "$PROJECT/baml.toml"

SPAN_QUERY="SELECT span_id AS span_id, parent_span_id AS parent_span_id, span_name AS span_name, status AS status, duration AS duration, input_args AS input_args, output_value AS output_value, error_value AS error_value, context_metadata['baml_ml.model_id'] AS model_id, context_metadata['baml_ml.step_index'] AS step_index, context_metadata['baml_ml.step_kind'] AS step_kind, context_metadata['baml_ml.branch_name'] AS branch_name, context_metadata['baml_ml.shadow_role'] AS shadow_role, context_metadata['baml_ml.row_index'] AS row_index, context_metadata['baml_ml.escalated'] AS escalated, context_metadata['baml_ml.trigger_reasons'] AS trigger_reasons FROM spans ORDER BY start_time"

run_probe() {
  local name="$1" fn="$2" mode="$3"
  (
    local dir="$WORK/run-$name"
    mkdir -p "$dir"
    cp -R "$PROJECT/baml_src" "$PROJECT/artifacts" "$PROJECT/baml.toml" "$dir/"
    cd "$dir"
    export HOME="$dir/home" BAML_HOME="$dir/home/baml"
    export XDG_CONFIG_HOME="$HOME/config" XDG_DATA_HOME="$HOME/data" XDG_CACHE_HOME="$HOME/cache"
    mkdir -p "$HOME" "$OUT/$name"
    if [ "$mode" = off ]; then export BAML_TELEMETRY=off; fi
    "$CLI" run "$fn" --output-format json > "$OUT/$name/result.json" 2> "$OUT/$name/run.stderr" || { cat "$OUT/$name/run.stderr"; exit 1; }
    "$CLI" local recordings query --output-format json "$SPAN_QUERY" > "$OUT/$name/spans.json" 2> "$OUT/$name/query.stderr"
    "$CLI" local recordings query --output-format json "SELECT function_name, invocation_count FROM profiler" > "$OUT/$name/profiler.json" 2>> "$OUT/$name/query.stderr"
  )
}

near() {
  cat > /tmp/near-8eae5c.jq <<'EOF'
def near($a;$b): if ($b|type)=="array" then (($a|type)=="array" and ($a|length)==($b|length) and ([range(0;($b|length)) as $i|near($a[$i];$b[$i])]|all)) else (($a|type)=="number" and (($a-$b)|fabs)<1e-9) or ($a==$b) end;
near($a;$b)
EOF
}

run_probe full HeteroFullProbe full
run_probe semantic HeteroSemanticProbe semantic
run_probe timing HeteroTimingProbe timing
run_probe adaptive HeteroAdaptiveProbe adaptive
run_probe shadow HeteroShadowProbe semantic
run_probe off HeteroFullProbe off

# Predictions are identical across modes and match the sklearn 1.8.0 oracle.
for mode in semantic timing adaptive; do
  jq -e --slurpfile ref "$PROJECT/artifacts/fixtures/v3/oracle.json" --slurpfile a "$OUT/full/result.json" --slurpfile b "$OUT/$mode/result.json" '
    def near($a;$b): if ($b|type)=="array" then (($a|type)=="array" and ($a|length)==($b|length) and ([range(0;($b|length)) as $i|near($a[$i];$b[$i])]|all)) else (($a|type)=="number" and (($a-$b)|fabs)<1e-9) end;
    $a[0].ok and $b[0].ok and near($a[0].probs; $ref[0]["v3-ct-hetero-forest"].predict_proba) and $a[0].probs==$b[0].probs
  ' "$OUT/$mode/result.json" || { cat "$OUT/$mode/result.json"; exit 1; }
done
jq -e --slurpfile ref "$PROJECT/artifacts/fixtures/v3/oracle.json" --slurpfile s "$OUT/shadow/result.json" '
  def near($a;$b): if ($b|type)=="array" then (($a|type)=="array" and ($a|length)==($b|length) and ([range(0;($b|length)) as $i|near($a[$i];$b[$i])]|all)) else (($a|type)=="number" and (($a-$b)|fabs)<1e-9) end;
  near($s[0].predictions; $ref[0]["v3-ct-hetero-forest"].predict_proba) and
  $s[0].summary.challenger_ok and $s[0].summary.triggers.missing_input==2 and $s[0].summary.escalated_rows==2
' "$OUT/shadow/result.json"

# Full: complete stage inventory with ordered step metadata, branch hierarchy,
# redacted fitted state and captured values.
jq -e -L "$ROOT/scripts" '
  import "btel_jq" as btel;
  def rows($n): [btel::named_rows[]|select(.span_name==("user."+$n))];
  .outcome.status=="complete" and
  (rows("PipelineModelLoaded")|length==1) and
  (rows("ObservedColumnTransformer") as $ct |
    ($ct|length==1) and ($ct[0].step_index==0) and ($ct[0].step_kind=="column_transformer") and ($ct[0].model_id!=null) and
    (rows("ObservedColumnTransformerBranch") as $br |
      ($br|length==2) and ([$br[]|select(.step_index==0 and .branch_name!=null and .parent_span_id==$ct[0].span_id and .step_kind=="column_transformer")]|length==2)) and
    (rows("ObservedImputeStep") as $im | ($im|length==2) and ([$im[]|select(.step_kind=="simple_imputer" and .step_index==0 and .branch_name!=null)]|length==2)) and
    (rows("ObservedScaleStep") as $sc | ($sc|length==1) and ($sc[0].step_kind=="standard_scaler" and $sc[0].step_index==0)) and
    (rows("ObservedOneHotEncode") as $oh | ($oh|length==1) and ($oh[0].step_kind=="one_hot_encoder" and $oh[0].step_index==0)) and
    (rows("ObservedForestTraversal")|length==1 and .[0].status=="return" and .[0].output_value!=null and .[0].input_args!=null))
' "$OUT/full/spans.json" || { cat "$OUT/full/spans.json"; exit 1; }
# Full redaction: no fitted-state keys in any step payload; values are captured.
jq -e -L "$ROOT/scripts" '
  import "btel_jq" as btel;
  def rows($n): [btel::named_rows[]|select(.span_name==("user."+$n))];
  ((rows("ObservedImputeStep")+rows("ObservedScaleStep")+rows("ObservedOneHotEncode")+rows("ObservedColumnTransformer")+rows("ObservedColumnTransformerBranch")+rows("ObservedForestTraversal"))|all(
    ((.input_args|tostring) + (.output_value|tostring)) as $p |
    ($p|test("statistics|fill_strings|categories|\"trees\"|\"nodes\"|\"scaler\"|\"imputer\"|\"one_hot\"|column_transformer|\"means\"|\"scales\"")|not) and
    (.input_args!=null) and (.output_value!=null)
  ))
' "$OUT/full/spans.json" || { cat "$OUT/full/spans.json"; exit 1; }
# Semantic: same spans, no payloads.
jq -e -L "$ROOT/scripts" '
  import "btel_jq" as btel;
  def rows($n): [btel::named_rows[]|select(.span_name==("user."+$n))];
  (rows("ObservedColumnTransformer")+rows("ObservedColumnTransformerBranch")+rows("ObservedImputeStep")+rows("ObservedScaleStep")+rows("ObservedOneHotEncode")+rows("ObservedForestTraversal")|length==8) and
  ((rows("ObservedColumnTransformer")+rows("ObservedImputeStep")+rows("ObservedScaleStep")+rows("ObservedOneHotEncode")+rows("ObservedForestTraversal"))|all(.input_args==null and .output_value==null and .status=="return"))
' "$OUT/semantic/spans.json" || { cat "$OUT/semantic/spans.json"; exit 1; }
# Timing and off: no semantic spans at all.
jq -e -L "$ROOT/scripts" 'import "btel_jq" as btel; [btel::named_rows[]|select(.span_name|test("user\\.(Observed|PipelineModelLoaded|EscalatedRow)"))]|length==0' "$OUT/timing/spans.json"
jq -e '.rows==[]' "$OUT/off/spans.json" "$OUT/off/profiler.json"
# Adaptive: summary plus per-row Full replays with trigger reasons; the lazy
# model snapshot is captured exactly once, after the semantic pass.
jq -e -L "$ROOT/scripts" '
  import "btel_jq" as btel;
  def rows($n): [btel::named_rows[]|select(.span_name==("user."+$n))];
  (rows("AdaptiveSummary")|length==1) and
  (rows("EscalatedRow") as $er | ($er|length==2) and ([$er[]|select(.escalated==true and .trigger_reasons=="missing_input" and (.row_index==0 or .row_index==1))]|length==2)) and
  (rows("PipelineModelLoaded")|length==2) and
  (rows("ObservedForestTraversal")|length==3)
' "$OUT/adaptive/spans.json" || { cat "$OUT/adaptive/spans.json"; exit 1; }
# Shadow: champion/challenger roles, selective Full replays, summary metadata.
jq -e -L "$ROOT/scripts" '
  import "btel_jq" as btel;
  def rows($n): [btel::named_rows[]|select(.span_name==("user."+$n))];
  (rows("ShadowEscalatedRow")|length==2) and
  (rows("ObservedColumnTransformer") as $ct | ($ct|length==6) and ([$ct[]|select(.shadow_role=="champion" or .shadow_role=="challenger")]|length==6)) and
  (rows("PipelineModelLoaded")|length==4) and
  (rows("ObservedForestTraversal")|length==6)
' "$OUT/shadow/spans.json" || { cat "$OUT/shadow/spans.json"; exit 1; }

for name in full semantic timing adaptive shadow off; do
  echo "heterogeneous v3 forest probe passed: $name"
done
