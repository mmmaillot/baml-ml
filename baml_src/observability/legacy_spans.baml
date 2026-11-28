// Frozen BTEL contract span wrappers for schema 1 & 2 models.
// @contract(btel-v1)
function PrepareTreeArtifact(raw: string) -> FittedTreeArtifact {
  let a = DecodeTreeArtifact(raw);
  let err = TreeArtifactError(a);
  if (err != "") { throw baml.errors.InvalidArgument { message: err }; }
  a
}

// @contract(btel-v1)
function TreeModelLoaded(artifact: FittedTreeArtifact) -> void {}

// A single semantic batch span, not one full span for every visited node.
// Full mode exposes ordered node IDs and leaf IDs; together with the load
// record and scaled inputs these reconstruct split features and thresholds.

// @contract(btel-v1)
function ObservedTreeTraversal(x: float[][], state: TracePrivate<TreeClassifierState>, capture_path: bool) -> TreeBatchResult throws InferenceFailure {
  TreePredictCore(x,state.value,capture_path, $trace = trace.timing())
}

// @contract(btel-v1)
function ModelLoaded(artifact: FittedArtifact) -> void {}


// @contract(btel-v1)
function ObservedImpute(x: float[][], state: TracePrivate<MeanImputer>) -> ImputerTransformResult throws InferenceFailure {
  let r = ImputerTransformCore(x, state.value, validated_state = true);
  if (!r.ok) { throw InferenceFailure { stage: "imputation", message: r.error }; }
  r
}


// @contract(btel-v1)
function ObservedScale(x: float[][], state: TracePrivate<StandardScaler>) -> ScalerTransformResult throws InferenceFailure {
  let r = ScalerTransformCore(x, state.value, true, validated_state = true);
  if (!r.ok) { throw InferenceFailure { stage: "standardization", message: r.error }; }
  r
}


// @contract(btel-v1)
function ObservedLinear(x: float[][], state: TracePrivate<LinearRegressionModel>) -> NumArrayResult throws InferenceFailure {
  let r = LinearRegressionPredictCore(x, state.value, true, validated_state = true);
  if (!r.ok) { throw InferenceFailure { stage: "linear_prediction", message: r.error }; }
  r
}


// @contract(btel-v1)
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


// @contract(btel-v1)
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


// @contract(btel-v1)
function ObservedLabels(scores: float[][], classes: int[]) -> IntArrayResult {
  IntArrayResult { ok: true, value: ClassLabels(scores, classes), error: "" }
}


function ClassLabels(scores: float[][], classes: int[]) -> int[] {
  let labels: int[] = [];
  for (let i = 0; i < scores.length(); i += 1) {
    let best = 0;
    for (let k = 1; k < scores[i].length(); k += 1) {
      if (scores[i][k] > scores[i][best]) { best = k; }
    }
    labels.push(classes[best]);
  }
  labels
}

// @contract(btel-v1)
function PrepareArtifact(raw: string, expected_linear: bool?) -> FittedArtifact {
  let a = DecodeFittedArtifact(raw);
  let err = ArtifactError(a);
  if (err != "") {
    throw baml.errors.InvalidArgument { message: err };
  }
  if (expected_linear == true && a.estimator_kind != "linear") {
    throw baml.errors.InvalidArgument { message: "artifact: expected linear estimator" };
  }
  if (expected_linear == false && a.estimator_kind == "linear") {
    throw baml.errors.InvalidArgument { message: "artifact: expected logistic estimator" };
  }
  a
}


function PreparedPreprocess(x: float[][], imputer: MeanImputer, scaler: StandardScaler, opts: trace.Options) -> float[][] throws InferenceFailure {
  let t1 = ObservedImpute(x, TracePrivate<MeanImputer> { value: imputer }, $trace = opts);
  let t2 = ObservedScale(t1.x, TracePrivate<StandardScaler> { value: scaler }, $trace = opts);
  t2.x
}
