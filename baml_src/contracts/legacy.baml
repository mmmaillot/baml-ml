// Frozen schema 1 & 2 artifact contracts: data-only JSON schemas, decoding and static validation.
// Version 2 is deliberately tree-only; the existing v1 schema remains unchanged.

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
  let err = ProvenanceError(a.source, a.source_version, a.model_id);
  if (err != "") { return err; }
  if (a.imputer_kind != "mean" || a.scaler_kind != "standard") { return "artifact: unsupported preprocessing"; }
  if (n == 0 || a.imputer_means.length() != n || a.scaler_means.length() != n || a.scaler_scales.length() != n) { return "artifact: invalid preprocessing dimensions"; }
  for (let j = 0; j < n; j += 1) {
    err = FeatureNameErrorAt(a.features, j);
    if (err != "") { return err; }
    if (!IsFinite(a.imputer_means[j]) || !ScalerParamValid(a.scaler_means[j], a.scaler_scales[j])) { return "artifact: invalid preprocessing parameters"; }
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
    if (!SortedUniqueClasses(a.classes)) { return "artifact: invalid class order"; }
  }
  FiniteMatrixParams(a.weights, a.intercepts, n)
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
  ClassificationPipeline { imputer: ArtifactImputer(a), scaler: ArtifactScaler(a), model: LogisticRegressionModel { classes: a.classes, weights: weights, intercepts: intercepts, n_features: a.features.length() } }
}

// Public entry points retain validation on mutable artifact values.

function PredictImportedLinear(x: float[][], features: string[], a: FittedArtifact) -> NumArrayResult {
  let err = ArtifactError(a);
  if (err == "") { err = LoadedFeatureError(features,a.features); }
  if (err == "" && a.estimator_kind != "linear") { err = "artifact: expected linear estimator"; }
  if (err != "") { return NumArrayResult { ok: false, value: [], error: err }; }
  RegressionPipelinePredict(x,ArtifactRegression(a))
}

function PredictImportedProba(x: float[][], features: string[], a: FittedArtifact) -> ProbaResult {
  let err = ArtifactError(a);
  if (err == "") { err = LoadedFeatureError(features,a.features); }
  if (err == "" && a.estimator_kind == "linear") { err = "artifact: expected logistic estimator"; }
  if (err != "") { return ProbaResult { ok: false, probs: [], error: err }; }
  ClassificationPipelinePredictProba(x,ArtifactClassification(a))
}

function PredictImportedLabels(x: float[][], features: string[], a: FittedArtifact) -> IntArrayResult {
  let err = ArtifactError(a);
  if (err == "") { err = LoadedFeatureError(features,a.features); }
  if (err == "" && a.estimator_kind == "linear") { err = "artifact: expected logistic estimator"; }
  if (err != "") { return IntArrayResult { ok: false, value: [], error: err }; }
  // sklearn predicts from decision scores, not rounded probabilities.
  // This also preserves binary score == 0 -> first class.
  let p = ArtifactClassification(a);
  let t = ClassificationPipelinePreprocess(x,p);
  if (!t.ok) { return IntArrayResult { ok: false, value: [], error: t.error }; }
  LabelsFromScores(t.value,p.model)
}

// Shared sklearn decision-score semantics; caller has validated model and input.

function LabelsFromScores(x: float[][], model: LogisticRegressionModel) -> IntArrayResult {
  let labels: int[] = [];
  for (let i = 0; i < x.length(); i += 1) {
    let scores = ModelRowLogits(x[i],model);
    let best = 0;
    for (let k = 0; k < scores.length(); k += 1) {
      if (!IsFinite(scores[k])) { return IntArrayResult { ok: false, value: [], error: "artifact: non-finite decision score" }; }
      if (scores[k] > scores[best]) { best = k; }
    }
    labels.push(model.classes[best]);
  }
  IntArrayResult { ok: true, value: labels, error: "" }
}

class ArtifactHeader {
  schema_version int
}

class FittedTreeArtifact {
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
  classes int[]
  nodes TreeNode[]
}

class PreparedTreePipeline {
  imputer MeanImputer
  scaler StandardScaler
  tree TreeClassifierState
}

function ReadArtifactVersion(raw: string) -> int {
  let h = baml.json.decode<ArtifactHeader>(raw);
  h.schema_version
}

function DecodeTreeArtifact(raw: string) -> FittedTreeArtifact {
  baml.json.decode<FittedTreeArtifact>(raw)
}

function TreeArtifactError(a: FittedTreeArtifact) -> string {
  let n = a.features.length();
  if (a.schema_version != 2 || a.estimator_kind != "decision_tree_classifier") { return "artifact: unsupported tree schema or estimator"; }
  let err = ProvenanceError(a.source, a.source_version, a.model_id);
  if (err != "") { return err; }
  if (a.imputer_kind != "mean" || a.scaler_kind != "standard") { return "artifact: unsupported preprocessing"; }
  if (n == 0 || a.imputer_means.length() != n || a.scaler_means.length() != n || a.scaler_scales.length() != n) { return "artifact: invalid preprocessing dimensions"; }
  for (let j = 0; j < n; j += 1) {
    err = FeatureNameErrorAt(a.features, j);
    if (err != "") { return err; }
    if (!IsFinite(a.imputer_means[j]) || !ScalerParamValid(a.scaler_means[j], a.scaler_scales[j])) { return "artifact: invalid preprocessing parameters"; }
  }
  TreeStateError(TreeClassifierState { n_features: n, classes: a.classes, nodes: a.nodes })
}

function PreparedTree(a: FittedTreeArtifact) -> PreparedTreePipeline {
  let n = a.features.length();
  PreparedTreePipeline {
    imputer: MeanImputer { means: a.imputer_means, n_features: n },
    scaler: StandardScaler { means: a.scaler_means, scales: a.scaler_scales, n_features: n },
    tree: TreeClassifierState { n_features: n, classes: a.classes, nodes: a.nodes }
  }
}
// Internal: caller owns the validated artifact and resolved policy.

