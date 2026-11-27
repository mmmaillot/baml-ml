// Native observation policy, fixed for the lifetime of a loaded model.
// No exporter, collector or manual span bookkeeping.
enum ObservationMode {
  Timing,
  Semantic,
  Full,
}

class InferenceFailure {
  stage string
  message string
}

// Redaction protects the captured state in the semantic call payloads, not
// arbitrary copies made by a caller or by additional tracing instrumentation.
class TracePrivate<T> {
  value: T @trace.redact
}

function ObservationOptions(mode: ObservationMode) -> trace.Options {
  if (mode == ObservationMode.Full) { return trace.span(); }
  if (mode == ObservationMode.Semantic) { return trace.empty_span(); }
  trace.timing()
}

function ModelObservation(mode: ObservationMode, model_id: string, schema_version: int, source_version: string) -> trace.Options {
  ObservationOptions(mode).context(metadata = {
    "baml_ml.contract_version": 1,
    "baml_ml.model_id": model_id,
    "baml_ml.schema_version": schema_version,
    "baml_ml.source_version": source_version
  })
}

// Called only after successful validation and conversion. Full mode records
// the artifact once; no parameter snapshot is returned to the caller.
function ModelLoaded(artifact: FittedArtifact) -> void {}

function ObservedImpute(x: float[][], state: TracePrivate<MeanImputer>) -> ImputerTransformResult throws InferenceFailure {
  let r = ImputerTransformCore(x, state.value, validated_state = true);
  if (!r.ok) { throw InferenceFailure { stage: "imputation", message: r.error }; }
  r
}

function ObservedScale(x: float[][], state: TracePrivate<StandardScaler>) -> ScalerTransformResult throws InferenceFailure {
  let r = ScalerTransformCore(x, state.value, true, validated_state = true);
  if (!r.ok) { throw InferenceFailure { stage: "standardization", message: r.error }; }
  r
}

function ObservedLinear(x: float[][], state: TracePrivate<LinearRegressionModel>) -> NumArrayResult throws InferenceFailure {
  let r = LinearRegressionPredictCore(x, state.value, true, validated_state = true);
  if (!r.ok) { throw InferenceFailure { stage: "linear_prediction", message: r.error }; }
  r
}

function ObservedDecisionScores(x: float[][], state: TracePrivate<LogisticRegressionModel>) -> float[][] throws InferenceFailure {
  let scores: float[][] = [];
  for (let i = 0; i < x.length(); i += 1) {
    let row = ModelRowLogits(x[i],state.value);
    for (let k = 0; k < row.length(); k += 1) {
      if (!IsFinite(row[k])) { throw InferenceFailure { stage: "decision_scores", message: "artifact: non-finite decision score" }; }
    }
    scores.push(row);
  }
  scores
}

function ObservedSoftmax(scores: float[][]) -> ProbaResult throws InferenceFailure {
  let probs: float[][] = [];
  for (let i = 0; i < scores.length(); i += 1) {
    let row = SoftmaxRow(scores[i]);
    for (let k = 0; k < row.length(); k += 1) {
      if (!IsFinite(row[k])) { throw InferenceFailure { stage: "softmax", message: "logreg_predict_proba: probability is not finite" }; }
    }
    probs.push(row);
  }
  ProbaResult { ok: true, probs: probs, error: "" }
}

function ObservedLabels(scores: float[][], classes: int[]) -> IntArrayResult {
  let labels: int[] = [];
  for (let i = 0; i < scores.length(); i += 1) {
    let best = 0;
    for (let k = 1; k < scores[i].length(); k += 1) {
      if (scores[i][k] > scores[i][best]) { best = k; }
    }
    labels.push(classes[best]);
  }
  IntArrayResult { ok: true, value: labels, error: "" }
}
