// Factory-owned state. A BAML class/array is mutable by reference; do not
// expose an artifact, parameter arrays or the pipeline captured by these closures.
// Accept JSON strings, not caller-owned mutable objects. Each decode owns its state.
// The callable fields themselves can be replaced by the caller, but this does not
// expose or change the factory's captured parameters.

class LoadedLinearModel {
  predict_fn: (float[][], string[]) -> NumArrayResult throws never
  function predict(self, x: float[][], features: string[]) -> NumArrayResult {
    let f = self.predict_fn;
    f(x, features)
  }
}

class LoadedLogisticModel {
  proba_fn: (float[][], string[]) -> ProbaResult throws never
  labels_fn: (float[][], string[]) -> IntArrayResult throws never
  function predict_proba(self, x: float[][], features: string[]) -> ProbaResult {
    let f = self.proba_fn;
    f(x, features)
  }
  function predict(self, x: float[][], features: string[]) -> IntArrayResult {
    let f = self.labels_fn;
    f(x, features)
  }
}

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

function PrepareLinearModel(raw: string, observation: ObservationMode = ObservationMode.Semantic) -> LoadedLinearModel {
  // Do not record raw artifact text or repeat the decoded output.
  let opts = ObservationOptions(observation);
  let a = PrepareArtifact(raw, true, $trace = if (observation == ObservationMode.Full) { trace.span(inputs = false, output = false) } else { opts });
  BuildLoadedLinear(a, observation)
}

// Internal: caller owns the freshly decoded and validated artifact.
function BuildLoadedLinear(a: FittedArtifact, observation: ObservationMode) -> LoadedLinearModel {
  let p = ArtifactRegression(a);
  let names = a.features;
  let id = a.model_id;
  let model_opts = ModelObservation(observation,id,a.schema_version,a.source_version);
  ModelLoaded(a, $trace = model_opts);
  let state = TracePrivate<RegressionPipeline> { value: p };
  LoadedLinearModel {
    predict_fn: (x: float[][], features: string[]) -> NumArrayResult {
      PredictLoadedLinear(x, features, id, names, state, observation, $trace = model_opts) catch (e) {
        let f: InferenceFailure => NumArrayResult { ok: false, value: [], error: f.message }
      }
    }
  }
}

function PrepareLogisticModel(raw: string, observation: ObservationMode = ObservationMode.Semantic) -> LoadedLogisticModel {
  let opts = ObservationOptions(observation);
  let a = PrepareArtifact(raw, false, $trace = if (observation == ObservationMode.Full) { trace.span(inputs = false, output = false) } else { opts });
  BuildLoadedLogistic(a, observation)
}

// Internal: caller owns the freshly decoded and validated artifact.
function BuildLoadedLogistic(a: FittedArtifact, observation: ObservationMode) -> LoadedLogisticModel {
  let p = ArtifactClassification(a);
  let names = a.features;
  let id = a.model_id;
  let model_opts = ModelObservation(observation,id,a.schema_version,a.source_version);
  ModelLoaded(a, $trace = model_opts);
  let state = TracePrivate<ClassificationPipeline> { value: p };
  LoadedLogisticModel {
    proba_fn: (x: float[][], features: string[]) -> ProbaResult {
      PredictLoadedProba(x, features, id, names, state, observation, $trace = model_opts) catch (e) {
        let f: InferenceFailure => ProbaResult { ok: false, probs: [], error: f.message }
      }
    },
    labels_fn: (x: float[][], features: string[]) -> IntArrayResult {
      PredictLoadedLabels(x, features, id, names, state, observation, $trace = model_opts) catch (e) {
        let f: InferenceFailure => IntArrayResult { ok: false, value: [], error: f.message }
      }
    }
  }
}

function LoadLinearModel(path: string, observation: ObservationMode = ObservationMode.Semantic) -> LoadedLinearModel {
  PrepareLinearModel(baml.fs.read(path), observation = observation)
}

function LoadLogisticModel(path: string, observation: ObservationMode = ObservationMode.Semantic) -> LoadedLogisticModel {
  PrepareLogisticModel(baml.fs.read(path), observation = observation)
}

// Internal execution helpers. Model ID is an explicit argument for tracing.
// These assume the factory has validated and privately captured the state.
function LoadedFeatureError(features: string[], expected: string[]) -> string {
  if (features.length() != expected.length()) { return "artifact: feature order mismatch"; }
  for (let j = 0; j < features.length(); j += 1) {
    if (features[j] != expected[j]) { return "artifact: feature order mismatch"; }
  }
  ""
}

function PreparedPreprocess(x: float[][], imputer: MeanImputer, scaler: StandardScaler, opts: trace.Options) -> float[][] throws InferenceFailure {
  let t1 = ObservedImpute(x, TracePrivate<MeanImputer> { value: imputer }, $trace = opts);
  let t2 = ObservedScale(t1.x, TracePrivate<StandardScaler> { value: scaler }, $trace = opts);
  t2.x
}

function PredictLoadedLinear(x: float[][], features: string[], model_id: string, expected: string[], state: TracePrivate<RegressionPipeline>, observation: ObservationMode) -> NumArrayResult throws InferenceFailure {
  let err = LoadedFeatureError(features, expected);
  if (err != "") { throw InferenceFailure { stage: "input_validation", message: err }; }
  let p = state.value;
  let opts = ObservationOptions(observation);
  let t = PreparedPreprocess(x, p.imputer, p.scaler, opts);
  ObservedLinear(t, TracePrivate<LinearRegressionModel> { value: p.model }, $trace = opts)
}

function PredictLoadedProba(x: float[][], features: string[], model_id: string, expected: string[], state: TracePrivate<ClassificationPipeline>, observation: ObservationMode) -> ProbaResult throws InferenceFailure {
  let err = LoadedFeatureError(features, expected);
  if (err != "") { throw InferenceFailure { stage: "input_validation", message: err }; }
  let p = state.value;
  let opts = ObservationOptions(observation);
  let t = PreparedPreprocess(x, p.imputer, p.scaler, opts);
  let scores = ObservedDecisionScores(t, TracePrivate<LogisticRegressionModel> { value: p.model }, $trace = opts);
  ObservedSoftmax(scores, $trace = opts)
}

function PredictLoadedLabels(x: float[][], features: string[], model_id: string, expected: string[], state: TracePrivate<ClassificationPipeline>, observation: ObservationMode) -> IntArrayResult throws InferenceFailure {
  let err = LoadedFeatureError(features, expected);
  if (err != "") { throw InferenceFailure { stage: "input_validation", message: err }; }
  let p = state.value;
  let opts = ObservationOptions(observation);
  let t = PreparedPreprocess(x, p.imputer, p.scaler, opts);
  let scores = ObservedDecisionScores(t, TracePrivate<LogisticRegressionModel> { value: p.model }, $trace = opts);
  ObservedLabels(scores,p.model.classes, $trace = opts)
}
