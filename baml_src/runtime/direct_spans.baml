// BTEL contract entry points for direct inference. The execution core calls stage spans in observability/.

// @contract(btel-v1)
function PredictLoadedTree(x: float[][], features: string[], model_id: string, expected: string[], state: TracePrivate<ExecutableModel>, observation: ObservationMode) -> TreeBatchResult throws InferenceFailure {
  let err = LoadedFeatureError(features,expected);
  if (err != "") { throw InferenceFailure { stage: "input_validation", message: err }; }
  // Direct execution: delegate to the engine core with no signal collection,
  // so no RunSelectiveModel span is inserted in non-adaptive modes.
  let r = state.value.run(NumericExecutionInput { rows: x }, Operation.Predict, observation, signals = null);
  TreeBatchResult { labels: r.labels, probs: r.probabilities, leaves: r.leaves, paths: r.paths }
}


// @contract(btel-v1)
function PredictLoadedLinear(x: float[][], features: string[], model_id: string, expected: string[], state: TracePrivate<ExecutableModel>, observation: ObservationMode) -> NumArrayResult throws InferenceFailure {
  let err = LoadedFeatureError(features, expected);
  if (err != "") { throw InferenceFailure { stage: "input_validation", message: err }; }
  let r = state.value.run(NumericExecutionInput { rows: x }, Operation.Predict, observation, signals = null);
  NumArrayResult { ok: true, value: r.values, error: "" }
}


// @contract(btel-v1)
function PredictLoadedProba(x: float[][], features: string[], model_id: string, expected: string[], state: TracePrivate<ExecutableModel>, observation: ObservationMode) -> ProbaResult throws InferenceFailure {
  let err = LoadedFeatureError(features, expected);
  if (err != "") { throw InferenceFailure { stage: "input_validation", message: err }; }
  let r = state.value.run(NumericExecutionInput { rows: x }, Operation.PredictProba, observation, signals = null);
  ProbaResult { ok: true, probs: r.probabilities, error: "" }
}


// @contract(btel-v1)
function PredictLoadedLabels(x: float[][], features: string[], model_id: string, expected: string[], state: TracePrivate<ExecutableModel>, observation: ObservationMode) -> IntArrayResult throws InferenceFailure {
  let err = LoadedFeatureError(features, expected);
  if (err != "") { throw InferenceFailure { stage: "input_validation", message: err }; }
  let r = state.value.run(NumericExecutionInput { rows: x }, Operation.Predict, observation, signals = null);
  IntArrayResult { ok: true, value: r.labels, error: "" }
}


// @contract(btel-v1)
function PredictLoadedPipeline(cells: Cell[][], features: string[], model_id: string, expected: string[], state: TracePrivate<ExecutableModel>, observation: ObservationMode, operation: string) -> PipelineResult throws InferenceFailure {
  let err = LoadedFeatureError(features, expected);
  if (err != "") { throw InferenceFailure { stage: "input_validation", message: err }; }
  // The span records the wire string; the enum lives behind the boundary.
  // Non-throwing parse: the public entry already validated the operation.
  let op = if (operation == "predict-proba") { Operation.PredictProba } else { Operation.Predict };
  // Direct execution: delegate to the engine core with no signal collection,
  // so no RunSelectiveModel span is inserted in non-adaptive modes.
  let r = state.value.run(cells, op, observation, signals = null);
  PipelineResult { ok: true, values: r.values, labels: r.labels, probs: r.probabilities, error: "" }
}
