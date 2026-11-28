// Schema 3 arbitrary-length pipeline contract: JSON schema, decoding and static type-chain validation.
class BatchInputV3 {
  features string[]
  rows (float | string | null)[][]
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


function SortedUniqueClasses(classes: int[]) -> bool {
  for (let k = 1; k < classes.length(); k += 1) {
    if (classes[k] <= classes[k - 1]) { return false; }
  }
  true
}


function FiniteMatrixParams(weights: float[][], intercepts: float[], n: int) -> string {
  if (weights.length() != intercepts.length()) { return "artifact: invalid model dimensions"; }
  for (let i = 0; i < weights.length(); i += 1) {
    if (weights[i].length() != n || !IsFinite(intercepts[i])) { return "artifact: invalid model dimensions or intercept"; }
    for (let j = 0; j < n; j += 1) {
      if (!IsFinite(weights[i][j])) { return "artifact: non-finite weight"; }
    }
  }
  ""
}


function PipelineEstimatorError(e: PipelineEstimator, n: int) -> string {
  if (e.n_features_in != n) { return "artifact: estimator input dimension mismatch"; }
  if (e.kind == "linear") {
    if (e.weights.length() != 1 || e.intercepts.length() != 1 || e.classes.length() != 0 || e.nodes.length() != 0 || e.trees.length() != 0) { return "artifact: invalid linear dimensions"; }
    return FiniteMatrixParams(e.weights, e.intercepts, n);
  }
  if (e.kind == "logistic_binary") {
    if (e.classes.length() != 2 || !SortedUniqueClasses(e.classes) || e.weights.length() != 1 || e.intercepts.length() != 1 || e.nodes.length() != 0 || e.trees.length() != 0) { return "artifact: invalid binary dimensions"; }
    return FiniteMatrixParams(e.weights, e.intercepts, n);
  }
  if (e.kind == "logistic_multinomial") {
    if (e.classes.length() < 3 || !SortedUniqueClasses(e.classes) || e.weights.length() != e.classes.length() || e.intercepts.length() != e.classes.length() || e.nodes.length() != 0 || e.trees.length() != 0) { return "artifact: invalid multinomial dimensions"; }
    return FiniteMatrixParams(e.weights, e.intercepts, n);
  }
  if (e.kind == "decision_tree_classifier") {
    if (e.weights.length() != 0 || e.intercepts.length() != 0 || e.trees.length() != 0) { return "artifact: invalid tree dimensions"; }
    return TreeStateError(TreeClassifierState { n_features: n, classes: e.classes, nodes: e.nodes });
  }
  if (e.kind == "random_forest_classifier") {
    if (e.weights.length() != 0 || e.intercepts.length() != 0 || e.nodes.length() != 0) { return "artifact: invalid forest dimensions"; }
    return ForestStateError(ForestClassifierState { n_features: n, classes: e.classes, trees: e.trees });
  }
  "artifact: unsupported estimator"
}


function PipelineStepError(s: PipelineStep, n_in: int, kinds: string[]) -> string {
  if (s.n_features_in != n_in) { return "artifact: step input dimension mismatch"; }
  if (s.kind == "simple_imputer") {
    let imp = s.imputer;
    if (imp == null || s.scaler != null || s.one_hot != null || s.column_transformer != null) { return "artifact: malformed simple_imputer step"; }
    if (imp.strategy != "mean" && imp.strategy != "median" && imp.strategy != "most_frequent" && imp.strategy != "constant") { return "artifact: unsupported imputer strategy"; }
    if (imp.value_kind == "numeric") {
      if (imp.statistics.length() != n_in || imp.fill_strings.length() != 0) { return "artifact: invalid imputer dimensions"; }
      for (let j = 0; j < n_in; j += 1) {
        if (!IsFinite(imp.statistics[j])) { return "artifact: non-finite imputer statistic"; }
        if (kinds[j] != "numeric") { return "artifact: numeric imputer on categorical input"; }
      }
    } else if (imp.value_kind == "string") {
      if (imp.strategy == "mean" || imp.strategy == "median") { return "artifact: unsupported imputer strategy for string statistics"; }
      if (imp.statistics.length() != 0 || imp.fill_strings.length() != n_in) { return "artifact: invalid imputer dimensions"; }
      for (let j = 0; j < n_in; j += 1) {
        if (imp.fill_strings[j] == "") { return "artifact: empty imputer fill string"; }
        if (kinds[j] != "categorical") { return "artifact: string imputer on numeric input"; }
      }
    } else {
      return "artifact: unsupported imputer value kind";
    }
    if (s.n_features_out != n_in) { return "artifact: imputer must preserve feature count"; }
    return "";
  }
  if (s.kind == "one_hot_encoder") {
    let ohe = s.one_hot;
    if (ohe == null || s.imputer != null || s.scaler != null || s.column_transformer != null) { return "artifact: malformed one_hot_encoder step"; }
    if (ohe.columns.length() != n_in) { return "artifact: invalid one-hot dimensions"; }
    if (ohe.handle_unknown != "error" && ohe.handle_unknown != "ignore") { return "artifact: unsupported one-hot handle_unknown"; }
    if (ohe.drop != null && ohe.drop != "first" && ohe.drop != "if_binary") { return "artifact: unsupported one-hot drop"; }
    let out = 0;
    for (let j = 0; j < n_in; j += 1) {
      let err = OneHotColumnError(ohe.columns[j]);
      if (err != "") { return err; }
      if (kinds[j] != "categorical") { return "artifact: one-hot encoder on numeric input"; }
      let b = OneHotBlockSize(ohe.columns[j], ohe.drop);
      out = out + b;
    }
    // Individual blocks may be zero-width (dropped singleton category), but
    // the total encoder output must keep at least one column.
    if (out < 1) { return "artifact: one-hot output has zero width"; }
    if (s.n_features_out != out) { return "artifact: one-hot output dimension mismatch"; }
    return "";
  }
  if (s.kind == "column_transformer") {
    let ct = s.column_transformer;
    if (ct == null || s.imputer != null || s.scaler != null || s.one_hot != null) { return "artifact: malformed column_transformer step"; }
    if (ct.remainder != "drop" && ct.remainder != "passthrough") { return "artifact: unsupported column transformer remainder"; }
    let used: bool[] = [];
    for (let j = 0; j < n_in; j += 1) { used.push(false); }
    let out = 0;
    for (let b = 0; b < ct.branches.length(); b += 1) {
      let branch = ct.branches[b];
      if (branch.name == "") { return "artifact: empty branch name"; }
      for (let b2 = 0; b2 < b; b2 += 1) {
        if (ct.branches[b2].name == branch.name) { return "artifact: duplicate branch name"; }
      }
      if (branch.mode != "drop" && branch.mode != "passthrough" && branch.mode != "pipeline") { return "artifact: unsupported branch mode"; }
      if (branch.columns.length() == 0) { return "artifact: branch has no columns"; }
      let sub_kinds: string[] = [];
      for (let c = 0; c < branch.columns.length(); c += 1) {
        let col = branch.columns[c];
        if (col < 0 || col >= n_in || used[col]) { return "artifact: invalid or overlapping branch column"; }
        used[col] = true;
        sub_kinds.push(kinds[col]);
      }
      if (branch.mode == "drop") {
        if (branch.steps.length() != 0) { return "artifact: drop branch with steps"; }
      } else if (branch.mode == "passthrough") {
        if (branch.steps.length() != 0) { return "artifact: passthrough branch with steps"; }
        for (let j = 0; j < sub_kinds.length(); j += 1) {
          if (sub_kinds[j] != "numeric") { return "artifact: passthrough branch on categorical input"; }
        }
        out = out + branch.columns.length();
      } else {
        if (branch.steps.length() == 0) { return "artifact: pipeline branch without steps"; }
        let err = BranchStepsError(branch.steps, branch.columns.length(), sub_kinds);
        if (err != "") { return err; }
        out = out + branch.steps[branch.steps.length() - 1].n_features_out;
      }
    }
    // The remainder must be exactly the ordered (ascending) complement of the
    // branch columns: no missing, extra, duplicated or out-of-range column.
    let complement: int[] = [];
    for (let j = 0; j < n_in; j += 1) {
      if (!used[j]) { complement.push(j); }
    }
    if (ct.remainder_columns.length() != complement.length()) { return "artifact: remainder columns are not the branch complement"; }
    for (let c = 0; c < complement.length(); c += 1) {
      if (ct.remainder_columns[c] != complement[c]) { return "artifact: remainder columns are not the ordered branch complement"; }
      if (ct.remainder == "passthrough" && kinds[complement[c]] != "numeric") { return "artifact: passthrough remainder on categorical input"; }
    }
    if (ct.remainder == "passthrough") { out = out + ct.remainder_columns.length(); }
    if (s.n_features_out != out) { return "artifact: column transformer output dimension mismatch"; }
    return "";
  }
  if (s.kind == "standard_scaler") {
    let sc = s.scaler;
    if (sc == null || s.imputer != null || s.one_hot != null || s.column_transformer != null) { return "artifact: malformed standard_scaler step"; }
    if (sc.means.length() != n_in || sc.scales.length() != n_in) { return "artifact: invalid scaler dimensions"; }
    for (let j = 0; j < n_in; j += 1) {
      if (!IsFinite(sc.means[j]) || !IsFinite(sc.scales[j]) || sc.scales[j] <= 0.0) { return "artifact: invalid scaler parameters"; }
      if (kinds[j] != "numeric") { return "artifact: scaler on categorical input"; }
    }
    if (s.n_features_out != n_in) { return "artifact: scaler must preserve feature count"; }
    return "";
  }
  "artifact: unsupported pipeline step kind"
}


function PipelineArtifactError(a: FittedPipelineArtifact) -> string {
  if (a.schema_version != 3) { return "artifact: unsupported pipeline schema version"; }
  if (a.source != "scikit-learn" || a.source_version == "" || a.model_id == "") { return "artifact: invalid provenance"; }
  let n = a.features.length();
  if (n == 0) { return "artifact: empty feature list"; }
  for (let j = 0; j < n; j += 1) {
    if (a.features[j] == "") { return "artifact: empty feature name"; }
    for (let k = 0; k < j; k += 1) {
      if (a.features[j] == a.features[k]) { return "artifact: duplicate feature name"; }
    }
  }
  if (a.input_kinds.length() != 0 && a.input_kinds.length() != n) { return "artifact: input kind count mismatch"; }
  let kinds: string[] = [];
  for (let j = 0; j < n; j += 1) {
    let kind = InputKindAt(a, j);
    if (kind != "numeric" && kind != "categorical") { return "artifact: unsupported input kind"; }
    kinds.push(kind);
  }
  let expected = n;
  for (let i = 0; i < a.steps.length(); i += 1) {
    let s = a.steps[i];
    let err = PipelineStepError(s, expected, kinds);
    if (err != "") { return err; }
    expected = s.n_features_out;
    if (s.kind == "one_hot_encoder" || s.kind == "column_transformer") {
      kinds = NumericKinds(expected);
    } else if (s.kind != "simple_imputer" && s.kind != "standard_scaler") {
      return "artifact: unsupported pipeline step kind";
    }
    // simple_imputer and standard_scaler preserve the column kinds.
  }
  // The terminal estimator consumes a dense numeric matrix.
  for (let j = 0; j < kinds.length(); j += 1) {
    if (kinds[j] != "numeric") { return "artifact: terminal estimator input is not numeric"; }
  }
  PipelineEstimatorError(a.estimator, expected)
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

