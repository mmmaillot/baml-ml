// Schema v3: arbitrary-length sequential pipelines (0..N typed preprocessing
// steps followed by 1 supported estimator), data-only. The v1 and v2 artifact,
// loader, BatchOutput and ShadowOutput contracts are unchanged.

class Cell {
  kind string // "number" | "string" | "null"
  num float
  str string
}

class BatchInputV3 {
  features string[]
  rows (float | string | null)[][]
}

class SimpleImputerStep {
  strategy string // "mean" | "median" | "most_frequent" | "constant"
  value_kind string // "numeric" | "string"
  statistics float[]
  fill_strings string[]
}

class ScalerStep {
  means float[]
  scales float[]
}

// One ordered fitted category list per input column. Integer categories are
// stored as float64 (exact for |c| < 2^53, enforced at export/validation).
class OneHotColumn {
  kind string // "string" | "integer"
  strings string[]
  ints float[]
}

class OneHotEncoderStep {
  columns OneHotColumn[]
  handle_unknown string // "error" | "ignore"
  drop string? // null (sklearn None) | "first" | "if_binary"
}

// One named branch of a column transformer: explicit validated input column
// indices plus "drop", "passthrough" (numeric columns only) or a nested
// sequential sub-pipeline of supported preprocessors (no nested transformer).
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

class CellMatrixResult {
  ok bool
  cells Cell[][]
  error string
}

class PipelineExecution {
  scaled float[][]? // output of the most recent standard_scaler step (Adaptive extreme_zscore)
}

class PipelineRun {
  values float[]
  labels int[]
  probs float[][]
}

class PipelineResult {
  ok bool
  values float[]
  labels int[]
  probs float[][]
  error string
}

class LoadedPipelineModel {
  estimator_kind string
  classes int[]
  predict_fn: (Cell[][], string[]) -> PipelineResult throws never
  proba_fn: (Cell[][], string[]) -> PipelineResult throws never
  function predict(self, cells: Cell[][], features: string[]) -> PipelineResult {
    let f = self.predict_fn;
    f(cells, features)
  }
  function predict_proba(self, cells: Cell[][], features: string[]) -> PipelineResult {
    let f = self.proba_fn;
    f(cells, features)
  }
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
    if (e.weights.length() != 1 || e.intercepts.length() != 1 || e.classes.length() != 0 || e.nodes.length() != 0) { return "artifact: invalid linear dimensions"; }
    return FiniteMatrixParams(e.weights, e.intercepts, n);
  }
  if (e.kind == "logistic_binary") {
    if (e.classes.length() != 2 || !SortedUniqueClasses(e.classes) || e.weights.length() != 1 || e.intercepts.length() != 1 || e.nodes.length() != 0) { return "artifact: invalid binary dimensions"; }
    return FiniteMatrixParams(e.weights, e.intercepts, n);
  }
  if (e.kind == "logistic_multinomial") {
    if (e.classes.length() < 3 || !SortedUniqueClasses(e.classes) || e.weights.length() != e.classes.length() || e.intercepts.length() != e.classes.length() || e.nodes.length() != 0) { return "artifact: invalid multinomial dimensions"; }
    return FiniteMatrixParams(e.weights, e.intercepts, n);
  }
  if (e.kind == "decision_tree_classifier") {
    if (e.weights.length() != 0 || e.intercepts.length() != 0) { return "artifact: invalid tree dimensions"; }
    return TreeStateError(TreeClassifierState { n_features: n, classes: e.classes, nodes: e.nodes });
  }
  "artifact: unsupported estimator"
}

function OneHotBlockSize(col: OneHotColumn, drop: string?) -> int {
  let n = if (col.kind == "string") { col.strings.length() } else { col.ints.length() };
  let adj = if (drop == "first" || (drop == "if_binary" && n == 2)) { 1 } else { 0 };
  n - adj
}

function OneHotColumnError(col: OneHotColumn) -> string {
  let n = if (col.kind == "string") { col.strings.length() } else { col.ints.length() };
  if (col.kind != "string" && col.kind != "integer") { return "artifact: unsupported one-hot category kind"; }
  if (n == 0) { return "artifact: empty one-hot categories"; }
  if (col.kind == "string") {
    if (col.ints.length() != 0) { return "artifact: malformed one-hot string categories"; }
    for (let i = 0; i < n; i += 1) {
      for (let k = 0; k < i; k += 1) {
        if (col.strings[i] == col.strings[k]) { return "artifact: duplicate one-hot category"; }
      }
    }
  } else {
    if (col.strings.length() != 0) { return "artifact: malformed one-hot integer categories"; }
    for (let i = 0; i < n; i += 1) {
      if (!IsFinite(col.ints[i]) || col.ints[i] != col.ints[i].floor() || Abs(col.ints[i]) >= 9007199254740992.0) { return "artifact: invalid one-hot integer category"; }
      for (let k = 0; k < i; k += 1) {
        if (col.ints[i] == col.ints[k]) { return "artifact: duplicate one-hot category"; }
      }
    }
  }
  ""
}

// Validate a nested sequential sub-pipeline (inside a column transformer
// branch). Nested column transformers are rejected: unsupported, not approximated.
function BranchStepsError(steps: PipelineStep[], n_in: int, kinds: string[]) -> string {
  let expected = n_in;
  let sub_kinds = kinds;
  for (let k = 0; k < steps.length(); k += 1) {
    let s = steps[k];
    if (s.kind == "column_transformer") { return "artifact: nested column transformer"; }
    let err = PipelineStepError(s, expected, sub_kinds);
    if (err != "") { return err; }
    expected = s.n_features_out;
    if (s.kind == "one_hot_encoder") {
      sub_kinds = NumericKinds(expected);
    }
  }
  for (let j = 0; j < sub_kinds.length(); j += 1) {
    if (sub_kinds[j] != "numeric") { return "artifact: branch pipeline output is not numeric"; }
  }
  ""
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
      if (b < 1) { return "artifact: one-hot block has zero width"; }
      out = out + b;
    }
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
    for (let c = 0; c < ct.remainder_columns.length(); c += 1) {
      let col = ct.remainder_columns[c];
      if (col < 0 || col >= n_in || used[col]) { return "artifact: invalid or overlapping remainder column"; }
      used[col] = true;
      if (ct.remainder == "passthrough" && kinds[col] != "numeric") { return "artifact: passthrough remainder on categorical input"; }
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
  PipelineEstimatorError(a.estimator, expected)
}

function PreparePipelineArtifact(raw: string) -> FittedPipelineArtifact {
  let a = DecodePipelineArtifact(raw);
  let err = PipelineArtifactError(a);
  if (err != "") {
    throw baml.errors.InvalidArgument { message: err };
  }
  a
}

function CellOf(v: float | string | null) -> Cell {
  if (v == null) { return Cell { kind: "null", num: 0.0, str: "" }; }
  if (v is string) { return Cell { kind: "string", num: 0.0, str: v }; }
  Cell { kind: "number", num: v, str: "" }
}

function BatchCells(input: BatchInputV3) -> Cell[][] {
  let out: Cell[][] = [];
  for (let i = 0; i < input.rows.length(); i += 1) {
    let row: Cell[] = [];
    for (let j = 0; j < input.rows[i].length(); j += 1) {
      row.push(CellOf(input.rows[i][j]));
    }
    out.push(row);
  }
  out
}

function ShadowCells(rows: (float | string | null)[][]) -> Cell[][] {
  let out: Cell[][] = [];
  for (let i = 0; i < rows.length(); i += 1) {
    let row: Cell[] = [];
    for (let j = 0; j < rows[i].length(); j += 1) {
      row.push(CellOf(rows[i][j]));
    }
    out.push(row);
  }
  out
}

// Programmatic float matrices use NaN for missing values, like the v1/v2 API.
function NumericCells(x: float[][]) -> Cell[][] {
  let out: Cell[][] = [];
  for (let i = 0; i < x.length(); i += 1) {
    let row: Cell[] = [];
    for (let j = 0; j < x[i].length(); j += 1) {
      if (IsNaN(x[i][j])) {
        row.push(Cell { kind: "null", num: 0.0, str: "" });
      } else {
        row.push(Cell { kind: "number", num: x[i][j], str: "" });
      }
    }
    out.push(row);
  }
  out
}

// Internal: numeric-only consumers (v1/v2 selective path, scaler output).
// String cells are rejected; null cells become NaN.
function CellsToFloatMatrix(cells: Cell[][]) -> float[][] throws InferenceFailure {
  let out: float[][] = [];
  for (let i = 0; i < cells.length(); i += 1) {
    let row: float[] = [];
    for (let j = 0; j < cells[i].length(); j += 1) {
      let c = cells[i][j];
      if (c.kind == "string") {
        throw InferenceFailure { stage: "input_validation", message: "batch: string value for a numeric-only model" };
      }
      if (c.kind == "null") {
        row.push(float.nan());
      } else {
        row.push(c.num);
      }
    }
    out.push(row);
  }
  out
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

function CheckCellMatrixDims(cells: Cell[][], n: int) -> string {
  if (cells.length() == 0) { return "matrix has no rows"; }
  let cols = cells[0].length();
  if (cols == 0) { return "matrix has zero columns"; }
  for (let i = 1; i < cells.length(); i += 1) {
    if (cells[i].length() != cols) { return "matrix is ragged: rows have different lengths"; }
  }
  if (cols != n) { return "feature count differs from fitted step"; }
  ""
}

// Internal: validated step state. Numeric imputation reuses the checked
// float64 core (null cells are NaN, string cells are rejected); string
// imputation replaces null cells with the fitted fill string.
function ImputerStepCore(cells: Cell[][], imp: SimpleImputerStep) -> CellMatrixResult {
  let n = if (imp.value_kind == "numeric") { imp.statistics.length() } else { imp.fill_strings.length() };
  let err = CheckCellMatrixDims(cells, n);
  let out: Cell[][] = [];
  if (err == "" && imp.value_kind == "string") {
    for (let i = 0; i < cells.length() && err == ""; i += 1) {
      let row: Cell[] = [];
      for (let j = 0; j < n; j += 1) {
        let c = cells[i][j];
        if (c.kind == "number") {
          err = "imputer_transform: numeric value in string imputer";
        } else if (c.kind == "null") {
          row.push(Cell { kind: "string", num: 0.0, str: imp.fill_strings[j] });
        } else {
          row.push(c);
        }
      }
      if (err == "") { out.push(row); }
    }
  }
  let x: float[][] = [];
  if (err == "" && imp.value_kind == "numeric") {
    for (let i = 0; i < cells.length(); i += 1) {
      let row: float[] = [];
      for (let j = 0; j < n; j += 1) {
        let c = cells[i][j];
        if (c.kind == "string") {
          err = "imputer_transform: string value in numeric imputer";
        } else if (c.kind == "null") {
          row.push(float.nan());
        } else {
          row.push(c.num);
        }
      }
      if (err != "") { break; }
      x.push(row);
    }
  }
  let r = ImputerTransformCore(x, MeanImputer { means: imp.statistics, n_features: n }, validated_state = true);
  if (err == "" && imp.value_kind == "numeric") { err = r.error; }
  if (err == "" && imp.value_kind == "numeric") {
    out = [];
    for (let i = 0; i < r.x.length(); i += 1) {
      let row: Cell[] = [];
      for (let j = 0; j < n; j += 1) {
        row.push(Cell { kind: "number", num: r.x[i][j], str: "" });
      }
      out.push(row);
    }
  }
  if (err == "") {
    CellMatrixResult { ok: true, cells: out, error: "" }
  } else {
    CellMatrixResult { ok: false, cells: [], error: err }
  }
}

// Internal: validated step state. Scaler output is finite by construction.
function ScalerStepCore(cells: Cell[][], sc: ScalerStep) -> CellMatrixResult {
  let n = sc.means.length();
  let err = CheckCellMatrixDims(cells, n);
  let x: float[][] = [];
  if (err == "") {
    for (let i = 0; i < cells.length(); i += 1) {
      let row: float[] = [];
      for (let j = 0; j < n; j += 1) {
        let c = cells[i][j];
        if (c.kind == "string") {
          err = "scaler_transform: string value in numeric input";
        } else if (c.kind == "null") {
          row.push(float.nan());
        } else {
          row.push(c.num);
        }
      }
      if (err != "") { break; }
      x.push(row);
    }
  }
  let r = ScalerTransformCore(x, StandardScaler { means: sc.means, scales: sc.scales, n_features: n }, false, validated_state = true);
  if (err == "") { err = r.error; }
  if (err == "") {
    let out: Cell[][] = [];
    for (let i = 0; i < r.x.length(); i += 1) {
      let row: Cell[] = [];
      for (let j = 0; j < n; j += 1) {
        row.push(Cell { kind: "number", num: r.x[i][j], str: "" });
      }
      out.push(row);
    }
    CellMatrixResult { ok: true, cells: out, error: "" }
  } else {
    CellMatrixResult { ok: false, cells: [], error: err }
  }
}

// Internal: validated step state. Deterministic dense float64 expansion in
// fitted category order. A dropped first category encodes as an all-zero
// block; unseen categories follow handle_unknown (error rejects, ignore
// emits all zeros), matching scikit-learn 1.8.0.
function OneHotStepCore(cells: Cell[][], ohe: OneHotEncoderStep) -> CellMatrixResult {
  let n = ohe.columns.length();
  let err = CheckCellMatrixDims(cells, n);
  let out: Cell[][] = [];
  if (err == "") {
    for (let i = 0; i < cells.length() && err == ""; i += 1) {
      let row: Cell[] = [];
      for (let j = 0; j < n && err == ""; j += 1) {
        let col = ohe.columns[j];
        let block = OneHotBlockSize(col, ohe.drop);
        let idx = -1;
        if (col.kind == "string" && cells[i][j].kind == "string") {
          for (let k = 0; k < col.strings.length(); k += 1) {
            if (cells[i][j].str == col.strings[k]) { idx = k; }
          }
        } else if (col.kind == "integer" && cells[i][j].kind == "number") {
          for (let k = 0; k < col.ints.length(); k += 1) {
            if (cells[i][j].num == col.ints[k]) { idx = k; }
          }
        }
        if (idx < 0 && ohe.handle_unknown == "error") {
          err = "one_hot_encoding: unseen category in column " + j.to_string();
        } else {
          // if_binary drops only when the original category count is two.
          let cat_count = if (col.kind == "string") { col.strings.length() } else { col.ints.length() };
          let drop_adj = if (ohe.drop == "first" || (ohe.drop == "if_binary" && cat_count == 2)) { 1 } else { 0 };
          for (let k = 0; k < block; k += 1) {
            let on = idx >= drop_adj && (idx - drop_adj) == k;
            row.push(Cell { kind: "number", num: if (on) { 1.0 } else { 0.0 }, str: "" });
          }
        }
      }
      if (err == "") { out.push(row); }
    }
  }
  if (err == "") {
    CellMatrixResult { ok: true, cells: out, error: "" }
  } else {
    CellMatrixResult { ok: false, cells: [], error: err }
  }
}

function StepOptions(opts: trace.Options, index: int, kind: string) -> trace.Options {
  opts.context(metadata = {
    "baml_ml.step_index": index,
    "baml_ml.step_kind": kind
  })
}

function ObservedImputeStep(cells: Cell[][], state: TracePrivate<SimpleImputerStep>) -> Cell[][] throws InferenceFailure {
  let r = ImputerStepCore(cells, state.value);
  if (!r.ok) { throw InferenceFailure { stage: "imputation", message: r.error }; }
  r.cells
}

function ObservedScaleStep(cells: Cell[][], state: TracePrivate<ScalerStep>) -> Cell[][] throws InferenceFailure {
  let r = ScalerStepCore(cells, state.value);
  if (!r.ok) { throw InferenceFailure { stage: "standardization", message: r.error }; }
  r.cells
}

function ObservedOneHotEncode(cells: Cell[][], state: TracePrivate<OneHotEncoderStep>) -> Cell[][] throws InferenceFailure {
  let r = OneHotStepCore(cells, state.value);
  if (!r.ok) { throw InferenceFailure { stage: "one_hot_encoding", message: r.error }; }
  r.cells
}

// Internal: execute one validated step. Used by top-level pipelines and by
// column transformer branches; the caller supplies the step index metadata.
function RunOneStep(cur: Cell[][], s: PipelineStep, index: int, opts: trace.Options, exec: PipelineExecution) -> Cell[][] throws InferenceFailure {
  if (s.kind == "simple_imputer") {
    let imp = s.imputer;
    if (imp == null) { throw InferenceFailure { stage: "imputation", message: "artifact: malformed simple_imputer step" }; }
    return ObservedImputeStep(cur, TracePrivate<SimpleImputerStep> { value: imp }, $trace = StepOptions(opts, index, s.kind));
  }
  if (s.kind == "standard_scaler") {
    let sc = s.scaler;
    if (sc == null) { throw InferenceFailure { stage: "standardization", message: "artifact: malformed standard_scaler step" }; }
    let out = ObservedScaleStep(cur, TracePrivate<ScalerStep> { value: sc }, $trace = StepOptions(opts, index, s.kind));
    exec.scaled = CellsToFloatMatrix(out);
    return out;
  }
  if (s.kind == "one_hot_encoder") {
    let ohe = s.one_hot;
    if (ohe == null) { throw InferenceFailure { stage: "one_hot_encoding", message: "artifact: malformed one_hot_encoder step" }; }
    return ObservedOneHotEncode(cur, TracePrivate<OneHotEncoderStep> { value: ohe }, $trace = StepOptions(opts, index, s.kind));
  }
  if (s.kind == "column_transformer") {
    if (s.column_transformer == null) { throw InferenceFailure { stage: "column_transform", message: "artifact: malformed column_transformer step" }; }
    return ObservedColumnTransformer(cur, TracePrivate<PipelineStep> { value: s }, index, opts, exec, $trace = StepOptions(opts, index, s.kind));
  }
  throw InferenceFailure { stage: "preprocessing", message: "artifact: unsupported pipeline step kind" };
}

function RunPipelineSteps(cells: Cell[][], a: FittedPipelineArtifact, opts: trace.Options, exec: PipelineExecution) -> Cell[][] throws InferenceFailure {
  let cur = cells;
  for (let i = 0; i < a.steps.length(); i += 1) {
    cur = RunOneStep(cur, a.steps[i], i, opts, exec);
  }
  cur
}

function BranchOptions(opts: trace.Options, index: int, name: string) -> trace.Options {
  opts.context(metadata = {
    "baml_ml.step_index": index,
    "baml_ml.branch_name": name
  })
}

// Per-branch child span with redacted fitted branch state.
function ObservedColumnTransformerBranch(sub: Cell[][], state: TracePrivate<ColumnTransformerBranch>, index: int, opts: trace.Options, exec: PipelineExecution) -> Cell[][] throws InferenceFailure {
  let branch = state.value;
  if (branch.mode == "passthrough") { return sub; }
  if (branch.mode == "drop") { return []; }
  let cur = sub;
  for (let k = 0; k < branch.steps.length(); k += 1) {
    cur = RunOneStep(cur, branch.steps[k], index, opts, exec);
  }
  cur
}

// Internal: extract the selected columns of one row.
function SubRow(row: Cell[], columns: int[]) -> Cell[] {
  let out: Cell[] = [];
  for (let c = 0; c < columns.length(); c += 1) {
    out.push(row[columns[c]]);
  }
  out
}

// Internal: validated step state. Selects explicit column subsets, runs
// each branch (drop/passthrough/sub-pipeline) and concatenates the branch
// outputs horizontally in branch order, then the resolved remainder columns.
function ColumnTransformerStepCore(cells: Cell[][], step: PipelineStep, opts: trace.Options, index: int, exec: PipelineExecution) -> CellMatrixResult {
  let ct = step.column_transformer;
  if (ct == null) { return CellMatrixResult { ok: false, cells: [], error: "artifact: malformed column_transformer step" }; }
  let err = CheckCellMatrixDims(cells, step.n_features_in);
  let out: Cell[][] = [];
  if (err == "") {
    for (let i = 0; i < cells.length() && err == ""; i += 1) {
      let row: Cell[] = [];
      for (let b = 0; b < ct.branches.length(); b += 1) {
        let branch = ct.branches[b];
        let sub: Cell[][] = [SubRow(cells[i], branch.columns)];
        let part = ObservedColumnTransformerBranch(sub, TracePrivate<ColumnTransformerBranch> { value: branch }, index, opts, exec, $trace = BranchOptions(opts, index, branch.name));
        if (part.length() > 0) {
          for (let j = 0; j < part[0].length(); j += 1) {
            row.push(part[0][j]);
          }
        }
      }
      if (ct.remainder == "passthrough") {
        for (let c = 0; c < ct.remainder_columns.length(); c += 1) {
          row.push(cells[i][ct.remainder_columns[c]]);
        }
      }
      out.push(row);
    }
  }
  // Column count of every output row must equal the validated n_features_out.
  if (err == "" && out.length() > 0 && out[0].length() != step.n_features_out) {
    err = "column_transform: branch output dimension mismatch";
  }
  if (err == "") {
    CellMatrixResult { ok: true, cells: out, error: "" }
  } else {
    CellMatrixResult { ok: false, cells: [], error: err }
  }
}

function ObservedColumnTransformer(cells: Cell[][], state: TracePrivate<PipelineStep>, index: int, opts: trace.Options, exec: PipelineExecution) -> Cell[][] throws InferenceFailure {
  let r = ColumnTransformerStepCore(cells, state.value, opts, index, exec);
  if (!r.ok) { throw InferenceFailure { stage: "column_transform", message: r.error }; }
  r.cells
}

// The estimator consumes a dense finite float64 matrix; missing or
// non-numeric values that survived preprocessing are rejected here.
function PipelineEstimatorMatrix(cells: Cell[][]) -> float[][] throws InferenceFailure {
  let out: float[][] = [];
  for (let i = 0; i < cells.length(); i += 1) {
    let row: float[] = [];
    for (let j = 0; j < cells[i].length(); j += 1) {
      let c = cells[i][j];
      if (c.kind == "string") {
        throw InferenceFailure { stage: "input_validation", message: "pipeline: non-numeric feature reached the estimator" };
      }
      if (c.kind == "null") {
        throw InferenceFailure { stage: "input_validation", message: "pipeline: missing value reached the estimator without imputation" };
      }
      if (!IsFinite(c.num)) {
        throw InferenceFailure { stage: "input_validation", message: "pipeline: non-finite feature reached the estimator" };
      }
      row.push(c.num);
    }
    out.push(row);
  }
  out
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
function RunPipelineEstimator(x: float[][], a: FittedPipelineArtifact, operation: string, full: bool, compute_probs: bool, opts: trace.Options) -> PipelineRun throws InferenceFailure {
  let e = a.estimator;
  if (e.kind == "linear") {
    if (operation == "predict-proba") {
      throw InferenceFailure { stage: "input_validation", message: "predict-proba requires a classifier estimator" };
    }
    let r = ObservedLinear(x, TracePrivate<LinearRegressionModel> { value: LinearRegressionModel { coefficients: e.weights[0], intercept: e.intercepts[0], n_features: e.n_features_in } }, $trace = opts);
    return PipelineRun { values: r.value, labels: [], probs: [] };
  }
  if (e.kind == "logistic_binary" || e.kind == "logistic_multinomial") {
    let model = PipelineLogisticModel(e);
    let scores = ObservedDecisionScores(x, TracePrivate<LogisticRegressionModel> { value: model }, $trace = opts);
    if (operation == "predict-proba") {
      let p = ObservedSoftmax(scores, $trace = opts);
      return PipelineRun { values: [], labels: ClassLabels(scores, e.classes), probs: p.probs };
    }
    let l = ObservedLabels(scores, e.classes, $trace = opts);
    let probs: float[][] = [];
    if (compute_probs) {
      for (let i = 0; i < scores.length(); i += 1) { probs.push(SoftmaxRow(scores[i])); }
    }
    return PipelineRun { values: [], labels: l.value, probs: probs };
  }
  if (e.kind == "decision_tree_classifier") {
    let r = ObservedTreeTraversal(x, TracePrivate<TreeClassifierState> { value: TreeClassifierState { n_features: e.n_features_in, classes: e.classes, nodes: e.nodes } }, full, $trace = opts);
    return PipelineRun { values: [], labels: r.labels, probs: r.probs };
  }
  throw InferenceFailure { stage: "prediction", message: "artifact: unsupported estimator" };
}

function PipelineModelLoaded(artifact: FittedPipelineArtifact) -> void {}

function PredictLoadedPipeline(cells: Cell[][], features: string[], model_id: string, expected: string[], state: TracePrivate<FittedPipelineArtifact>, observation: ObservationMode, operation: string) -> PipelineResult throws InferenceFailure {
  let err = LoadedFeatureError(features, expected);
  if (err != "") { throw InferenceFailure { stage: "input_validation", message: err }; }
  let a = state.value;
  err = ValidatePipelineInput(cells, a);
  if (err != "") { throw InferenceFailure { stage: "input_validation", message: err }; }
  let opts = ObservationOptions(observation);
  let exec = PipelineExecution { scaled: null };
  let out = RunPipelineSteps(cells, a, opts, exec);
  let x = PipelineEstimatorMatrix(out);
  let run = RunPipelineEstimator(x, a, operation, observation == ObservationMode.Full, false, opts);
  PipelineResult { ok: true, values: run.values, labels: run.labels, probs: run.probs, error: "" }
}

// Internal: caller owns the validated artifact and resolved policy.
function BuildLoadedPipeline(a: FittedPipelineArtifact, observation: ObservationMode, adaptive: AdaptivePolicy?) -> LoadedPipelineModel {
  let names = a.features;
  let id = a.model_id;
  let model_opts = ModelObservation(observation, id, a.schema_version, a.source_version);
  PipelineModelLoaded(a, $trace = model_opts);
  if (adaptive != null) { return AdaptivePipeline(SelectivePipeline(a), adaptive); }
  let state = TracePrivate<FittedPipelineArtifact> { value: a };
  LoadedPipelineModel {
    estimator_kind: a.estimator.kind,
    classes: a.estimator.classes,
    predict_fn: (cells: Cell[][], features: string[]) -> PipelineResult {
      PredictLoadedPipeline(cells, features, id, names, state, observation, "predict", $trace = model_opts) catch (e) {
        let f: InferenceFailure => PipelineResult { ok: false, values: [], labels: [], probs: [], error: f.message }
      }
    },
    proba_fn: (cells: Cell[][], features: string[]) -> PipelineResult {
      PredictLoadedPipeline(cells, features, id, names, state, observation, "predict-proba", $trace = model_opts) catch (e) {
        let f: InferenceFailure => PipelineResult { ok: false, values: [], labels: [], probs: [], error: f.message }
      }
    }
  }
}

function PreparePipelineModel(raw: string, observation: ObservationMode = ObservationMode.Semantic, policy: AdaptivePolicy? = null) -> LoadedPipelineModel {
  let adaptive = ResolveAdaptivePolicy(observation, policy);
  let opts = ObservationOptions(observation);
  let a = PreparePipelineArtifact(raw, $trace = if (observation == ObservationMode.Full) { trace.span(inputs = false, output = false) } else { opts });
  BuildLoadedPipeline(a, observation, adaptive)
}

function LoadPipelineModel(path: string, observation: ObservationMode = ObservationMode.Semantic, policy: AdaptivePolicy? = null) -> LoadedPipelineModel {
  PreparePipelineModel(baml.fs.read(path), observation = observation, policy = policy)
}

function CollectPipelineSignals(signals: AdaptiveRowSignals[], cells: Cell[][], exec: PipelineExecution, run: PipelineRun, is_classifier: bool) -> void {
  for (let i = 0; i < cells.length(); i += 1) {
    let has_nan = false;
    for (let j = 0; j < cells[i].length(); j += 1) {
      if (cells[i][j].kind == "null") { has_nan = true; }
    }
    let max_abs_z = 0.0;
    let scaled = exec.scaled;
    if (scaled != null && i < scaled.length()) { max_abs_z = MaxAbsoluteValue(scaled[i]); }
    if (is_classifier) {
      signals.push(AdaptiveRowSignals { has_nan: has_nan, max_abs_z: max_abs_z, probabilities: run.probs[i], label: run.labels[i], prediction: 0.0 });
    } else {
      signals.push(AdaptiveRowSignals { has_nan: has_nan, max_abs_z: max_abs_z, probabilities: [], label: null, prediction: run.values[i] });
    }
  }
}

function SelectivePipeline(a: FittedPipelineArtifact) -> SelectiveModel {
    SelectiveModel {
        model_id: a.model_id,
        schema_version: a.schema_version,
        source_version: a.source_version,
        estimator_kind: a.estimator.kind,
        features: a.features,
        classes: a.estimator.classes,
        guard: SnapshotGuard { captured: false },
        snapshot_fn: (opts: trace.Options) -> void {
            PipelineModelLoaded(a, $trace = opts);
        },
        run_fn: (
            cells: Cell[][],
            operation: string,
            mode: ObservationMode,
            signals: AdaptiveRowSignals[]?,
        ) -> SelectivePrediction {
            let err = ValidatePipelineInput(cells, a);
            if (err != "") {
                throw InferenceFailure { stage: "input_validation", message: err };
            }
            let opts = ObservationOptions(mode);
            let exec = PipelineExecution { scaled: null };
            let out = RunPipelineSteps(cells, a, opts, exec);
            let x = PipelineEstimatorMatrix(out);
            let is_classifier = a.estimator.kind != "linear";
            let run = RunPipelineEstimator(x, a, operation, mode == ObservationMode.Full, signals != null, opts);
            if (signals != null) {
                CollectPipelineSignals(signals, cells, exec, run, is_classifier);
            }
            SelectivePrediction {
                values: run.values,
                labels: run.labels,
                probabilities: run.probs,
            }
        },
    }
}

function RunPipelineBatch(raw_model: string, raw_input: string, operation: string, mode: ObservationMode, adaptive: AdaptivePolicy?) -> BatchOutput {
  let opts = ObservationOptions(mode);
  let a = PreparePipelineArtifact(raw_model, $trace = if (mode == ObservationMode.Full) { trace.span(inputs = false, output = false) } else { opts });
  let input = baml.json.decode<BatchInputV3>(raw_input);
  let cells = BatchCells(input);
  let m = BuildLoadedPipeline(a, mode, adaptive);
  if (operation == "predict-proba") {
    let r = m.predict_proba(cells, input.features);
    if (!r.ok) { throw baml.errors.InvalidArgument { message: r.error }; }
    return BatchOutput { schema_version: 1, model_id: a.model_id, estimator_kind: a.estimator.kind, operation: operation, classes: a.estimator.classes, predictions: r.probs };
  }
  let r = m.predict(cells, input.features);
  if (!r.ok) { throw baml.errors.InvalidArgument { message: r.error }; }
  if (a.estimator.kind == "linear") {
    return BatchOutput { schema_version: 1, model_id: a.model_id, estimator_kind: a.estimator.kind, operation: operation, classes: [], predictions: r.values };
  }
  BatchOutput { schema_version: 1, model_id: a.model_id, estimator_kind: a.estimator.kind, operation: operation, classes: a.estimator.classes, predictions: r.labels }
}
