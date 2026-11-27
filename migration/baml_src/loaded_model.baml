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

function PrepareArtifact(raw: string) -> FittedArtifact {
  let a = DecodeFittedArtifact(raw);
  let err = ArtifactError(a);
  if (err != "") {
    throw baml.errors.InvalidArgument { message: err };
  }
  a
}

function PrepareLinearModel(raw: string) -> LoadedLinearModel {
  let a = PrepareArtifact(raw);
  if (a.estimator_kind != "linear") {
    throw baml.errors.InvalidArgument { message: "artifact: expected linear estimator" };
  }
  let p = ArtifactRegression(a);
  let names = a.features;
  let id = a.model_id;
  LoadedLinearModel {
    predict_fn: (x: float[][], features: string[]) -> NumArrayResult {
      PredictLoadedLinear(x, features, id, names, p)
    }
  }
}

function PrepareLogisticModel(raw: string) -> LoadedLogisticModel {
  let a = PrepareArtifact(raw);
  if (a.estimator_kind == "linear") {
    throw baml.errors.InvalidArgument { message: "artifact: expected logistic estimator" };
  }
  let p = ArtifactClassification(a);
  let names = a.features;
  let id = a.model_id;
  LoadedLogisticModel {
    proba_fn: (x: float[][], features: string[]) -> ProbaResult {
      PredictLoadedProba(x, features, id, names, p)
    },
    labels_fn: (x: float[][], features: string[]) -> IntArrayResult {
      PredictLoadedLabels(x, features, id, names, p)
    }
  }
}

function LoadLinearModel(path: string) -> LoadedLinearModel {
  PrepareLinearModel(baml.fs.read(path))
}

function LoadLogisticModel(path: string) -> LoadedLogisticModel {
  PrepareLogisticModel(baml.fs.read(path))
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

function PreparedPreprocess(x: float[][], imputer: MeanImputer, scaler: StandardScaler) -> MatrixResult {
  let t1 = ImputerTransformCore(x, imputer, validated_state = true);
  if (!t1.ok) { return MatrixResult { ok: false, value: [], error: t1.error }; }
  let t2 = ScalerTransformCore(t1.x, scaler, true, validated_state = true);
  MatrixResult { ok: t2.ok, value: t2.x, error: t2.error }
}

function PredictLoadedLinear(x: float[][], features: string[], model_id: string, expected: string[], p: RegressionPipeline) -> NumArrayResult {
  let err = LoadedFeatureError(features, expected);
  if (err != "") { return NumArrayResult { ok: false, value: [], error: err }; }
  let t = PreparedPreprocess(x, p.imputer, p.scaler);
  if (!t.ok) { return NumArrayResult { ok: false, value: [], error: t.error }; }
  LinearRegressionPredictCore(t.value, p.model, true, validated_state = true)
}

function PredictLoadedProba(x: float[][], features: string[], model_id: string, expected: string[], p: ClassificationPipeline) -> ProbaResult {
  let err = LoadedFeatureError(features, expected);
  if (err != "") { return ProbaResult { ok: false, probs: [], error: err }; }
  let t = PreparedPreprocess(x, p.imputer, p.scaler);
  if (!t.ok) { return ProbaResult { ok: false, probs: [], error: t.error }; }
  LogRegPredictProbaCore(t.value, p.model, true, validated_state = true)
}

function PredictLoadedLabels(x: float[][], features: string[], model_id: string, expected: string[], p: ClassificationPipeline) -> IntArrayResult {
  let err = LoadedFeatureError(features, expected);
  if (err != "") { return IntArrayResult { ok: false, value: [], error: err }; }
  let t = PreparedPreprocess(x, p.imputer, p.scaler);
  if (!t.ok) { return IntArrayResult { ok: false, value: [], error: t.error }; }
  LabelsFromScores(t.value, p.model)
}
