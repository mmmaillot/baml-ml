// Schema 3 arbitrary-length pipeline contract: JSON schema and decoding.
class BatchInputV3 {
  features string[]
  rows (float | string | null)[][]
}



class ColumnTransformerBranch {
  name string
  columns int[]
  mode string // "drop" | "passthrough" | "pipeline"
  steps PipelineStep[]
}



class ColumnTransformerStep {
  branches ColumnTransformerBranch[]
  remainder string // "drop" | "passthrough"
  remainder_columns int[] // resolved to explicit indices at export time
}



class PipelineStep {
  kind string // "simple_imputer" | "standard_scaler" | "one_hot_encoder" | "column_transformer"
  n_features_in int
  n_features_out int
  imputer SimpleImputerStep?
  scaler ScalerStep?
  one_hot OneHotEncoderStep?
  column_transformer ColumnTransformerStep?
}



class PipelineEstimator {
  kind string
  n_features_in int
  weights float[][]
  intercepts float[]
  classes int[]
  nodes TreeNode[]
  trees TreeClassifierState[]
}



class FittedPipelineArtifact {
  schema_version int
  source string
  source_version string
  model_id string
  features string[]
  input_kinds string[] // empty means all numeric; otherwise per-feature "numeric"|"categorical"
  steps PipelineStep[]
  estimator PipelineEstimator
}



function DecodePipelineArtifact(raw: string) -> FittedPipelineArtifact {
  baml.json.decode<FittedPipelineArtifact>(raw)
}



function InputKindAt(a: FittedPipelineArtifact, j: int) -> string {
  if (a.input_kinds.length() == a.features.length()) { return a.input_kinds[j]; }
  "numeric"
}



function NumericKinds(n: int) -> string[] {
  let kinds: string[] = [];
  for (let j = 0; j < n; j += 1) { kinds.push("numeric"); }
  kinds
}



function ValidatePipelineInput(cells: Cell[][], a: FittedPipelineArtifact) -> string {
  let n = a.features.length();
  if (cells.length() == 0) { return "matrix has no rows"; }
  let cols = cells[0].length();
  if (cols == 0) { return "matrix has zero columns"; }
  for (let i = 1; i < cells.length(); i += 1) {
    if (cells[i].length() != cols) { return "matrix is ragged: rows have different lengths"; }
  }
  if (cols != n) { return "feature count differs from artifact"; }
  for (let i = 0; i < cells.length(); i += 1) {
    for (let j = 0; j < cols; j += 1) {
      let c = cells[i][j];
      if (InputKindAt(a, j) == "categorical") {
        // Integer-coded categorical columns receive numbers; the consuming
        // step (string imputer or one-hot encoder) enforces the exact type.
        if (c.kind == "number" && !IsFinite(c.num)) { return "non-finite value in categorical column " + j.to_string(); }
      } else {
        if (c.kind == "string") { return "string value in numeric column " + j.to_string(); }
        if (c.kind == "number" && !IsFinite(c.num)) { return "non-finite value in numeric column " + j.to_string(); }
      }
    }
  }
  ""
}



function PipelineLogisticModel(e: PipelineEstimator) -> LogisticRegressionModel {
  let weights = e.weights;
  let intercepts = e.intercepts;
  if (e.kind == "logistic_binary") {
    let zero: float[] = [];
    for (let j = 0; j < e.n_features_in; j += 1) { zero.push(0.0); }
    weights = [zero, e.weights[0]];
    intercepts = [0.0, e.intercepts[0]];
  }
  LogisticRegressionModel { classes: e.classes, weights: weights, intercepts: intercepts, n_features: e.n_features_in }
}

// Terminal estimator dispatch. Classifiers fill labels and probs; linear
// fills values. Label-only classification skips the observed softmax span
// unless compute_probs is set (Adaptive signal collection).


