// Data-only v1 contract. Training provenance is informational, not authentication.
class FittedArtifact {
  schema_version int
  source string
  source_version string
  model_id string
  features string[]
  imputer_kind string
  scaler_kind string
  estimator_kind string
  imputer_means float[]
  scaler_means float[]
  scaler_scales float[]
  weights float[][]
  intercepts float[]
  classes int[]
}

function DecodeFittedArtifact(raw: string) -> FittedArtifact {
  baml.json.decode<FittedArtifact>(raw)
}

function ReadFittedArtifact(path: string) -> FittedArtifact {
  DecodeFittedArtifact(baml.fs.read(path))
}

function ArtifactError(a: FittedArtifact) -> string {
  let n = a.features.length();
  if (a.schema_version != 1) { return "artifact: unsupported schema version"; }
  if (a.source != "scikit-learn" || a.source_version == "" || a.model_id == "") { return "artifact: invalid provenance"; }
  if (a.imputer_kind != "mean" || a.scaler_kind != "standard") { return "artifact: unsupported preprocessing"; }
  if (n == 0 || a.imputer_means.length() != n || a.scaler_means.length() != n || a.scaler_scales.length() != n) { return "artifact: invalid preprocessing dimensions"; }
  for (let j = 0; j < n; j += 1) {
    if (a.features[j] == "") { return "artifact: empty feature name"; }
    for (let k = 0; k < j; k += 1) {
      if (a.features[j] == a.features[k]) { return "artifact: duplicate feature name"; }
    }
    if (!IsFinite(a.imputer_means[j]) || !IsFinite(a.scaler_means[j]) || !IsFinite(a.scaler_scales[j]) || a.scaler_scales[j] <= 0.0) { return "artifact: invalid preprocessing parameters"; }
  }
  if (a.estimator_kind == "linear") {
    if (a.weights.length() != 1 || a.intercepts.length() != 1 || a.classes.length() != 0) { return "artifact: invalid linear dimensions"; }
  } else {
    if (a.estimator_kind == "logistic_multinomial") {
      if (a.classes.length() < 3 || a.weights.length() != a.classes.length() || a.intercepts.length() != a.classes.length()) { return "artifact: invalid multinomial dimensions"; }
    } else {
      if (a.estimator_kind == "logistic_binary") {
        if (a.classes.length() != 2 || a.weights.length() != 1 || a.intercepts.length() != 1) { return "artifact: invalid binary dimensions"; }
      } else { return "artifact: unsupported estimator"; }
    }
    for (let i = 1; i < a.classes.length(); i += 1) {
      if (a.classes[i] <= a.classes[i - 1]) { return "artifact: invalid class order"; }
    }
  }
  for (let i = 0; i < a.weights.length(); i += 1) {
    if (a.weights[i].length() != n || !IsFinite(a.intercepts[i])) { return "artifact: invalid model dimensions or intercept"; }
    for (let j = 0; j < n; j += 1) {
      if (!IsFinite(a.weights[i][j])) { return "artifact: non-finite weight"; }
    }
  }
  ""
}

function ArtifactFeatureError(a: FittedArtifact, features: string[]) -> string {
  if (features.length() != a.features.length()) { return "artifact: feature order mismatch"; }
  for (let j = 0; j < features.length(); j += 1) {
    if (features[j] != a.features[j]) { return "artifact: feature order mismatch"; }
  }
  ""
}

function ArtifactImputer(a: FittedArtifact) -> MeanImputer {
  MeanImputer { means: a.imputer_means, n_features: a.features.length() }
}
function ArtifactScaler(a: FittedArtifact) -> StandardScaler {
  StandardScaler { means: a.scaler_means, scales: a.scaler_scales, n_features: a.features.length() }
}
function ArtifactRegression(a: FittedArtifact) -> RegressionPipeline {
  RegressionPipeline { imputer: ArtifactImputer(a), scaler: ArtifactScaler(a), model: LinearRegressionModel { coefficients: a.weights[0], intercept: a.intercepts[0], n_features: a.features.length() } }
}
function ArtifactClassification(a: FittedArtifact) -> ClassificationPipeline {
  let weights = a.weights;
  let intercepts = a.intercepts;
  if (a.estimator_kind == "logistic_binary") {
    let zero: float[] = [];
    for (let j = 0; j < a.features.length(); j += 1) { zero.push(0.0); }
    weights = [zero,a.weights[0]];
    intercepts = [0.0,a.intercepts[0]];
  }
  ClassificationPipeline { imputer: ArtifactImputer(a), scaler: ArtifactScaler(a), model: LogisticRegressionModel { classes: a.classes, weights: weights, intercepts: intercepts, n_features: a.features.length(), n_iter: 0, converged: true } }
}

// Public entry points retain validation on mutable artifact values.
function PredictImportedLinear(x: float[][], features: string[], a: FittedArtifact) -> NumArrayResult {
  let err = ArtifactError(a);
  if (err == "") { err = ArtifactFeatureError(a,features); }
  if (err == "" && a.estimator_kind != "linear") { err = "artifact: expected linear estimator"; }
  if (err != "") { return NumArrayResult { ok: false, value: [], error: err }; }
  RegressionPipelinePredict(x,ArtifactRegression(a))
}
function PredictImportedProba(x: float[][], features: string[], a: FittedArtifact) -> ProbaResult {
  let err = ArtifactError(a);
  if (err == "") { err = ArtifactFeatureError(a,features); }
  if (err == "" && a.estimator_kind == "linear") { err = "artifact: expected logistic estimator"; }
  if (err != "") { return ProbaResult { ok: false, probs: [], error: err }; }
  ClassificationPipelinePredictProba(x,ArtifactClassification(a))
}
function PredictImportedLabels(x: float[][], features: string[], a: FittedArtifact) -> IntArrayResult {
  let err = ArtifactError(a);
  if (err == "") { err = ArtifactFeatureError(a,features); }
  if (err == "" && a.estimator_kind == "linear") { err = "artifact: expected logistic estimator"; }
  if (err != "") { return IntArrayResult { ok: false, value: [], error: err }; }
  // sklearn predicts from decision scores, not rounded probabilities.
  // This also preserves binary score == 0 -> first class.
  let p = ArtifactClassification(a);
  let t = ClassificationPipelinePreprocess(x,p);
  if (!t.ok) { return IntArrayResult { ok: false, value: [], error: t.error }; }
  let labels: int[] = [];
  for (let i = 0; i < t.value.length(); i += 1) {
    let scores = ModelRowLogits(t.value[i],p.model);
    let best = 0;
    for (let k = 0; k < scores.length(); k += 1) {
      if (!IsFinite(scores[k])) { return IntArrayResult { ok: false, value: [], error: "artifact: non-finite decision score" }; }
      if (scores[k] > scores[best]) { best = k; }
    }
    labels.push(a.classes[best]);
  }
  IntArrayResult { ok: true, value: labels, error: "" }
}
