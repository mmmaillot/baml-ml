// Frozen BTEL contract span wrappers for schema 3 pipelines and forests.
// @contract(btel-v1)
function ObservedForestTraversal(x: float[][], state: TracePrivate<ForestClassifierState>, capture_leaves: bool) -> ForestBatchResult throws InferenceFailure {
  ForestPredictCore(x, state.value, capture_leaves, $trace = trace.timing())
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


// @contract(btel-v1)
function PreparePipelineArtifact(raw: string) -> FittedPipelineArtifact {
  let a = DecodePipelineArtifact(raw);
  let err = PipelineArtifactError(a);
  if (err != "") {
    throw baml.errors.InvalidArgument { message: err };
  }
  a
}


function StepOptions(opts: trace.Options, index: int, kind: string) -> trace.Options {
  opts.context(metadata = {
    "baml_ml.step_index": index,
    "baml_ml.step_kind": kind
  })
}


// @contract(btel-v1)
function ObservedImputeStep(cells: Cell[][], state: TracePrivate<SimpleImputerStep>) -> Cell[][] throws InferenceFailure {
  let r = ImputerStepCore(cells, state.value);
  if (!r.ok) { throw InferenceFailure { stage: "imputation", message: r.error }; }
  r.cells
}


// @contract(btel-v1)
function ObservedScaleStep(cells: Cell[][], state: TracePrivate<ScalerStep>) -> Cell[][] throws InferenceFailure {
  let r = ScalerStepCore(cells, state.value);
  if (!r.ok) { throw InferenceFailure { stage: "standardization", message: r.error }; }
  r.cells
}


// @contract(btel-v1)
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
    MergeScalerAbsMax(exec, CellsToFloatMatrix(out));
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

// @contract(btel-v1)
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

function ColumnTransformerStepCore(cells: Cell[][], step: PipelineStep, opts: trace.Options, index: int, exec: PipelineExecution) -> CellMatrixResult {
  let ct = step.column_transformer;
  if (ct == null) { return CellMatrixResult { ok: false, cells: [], error: "artifact: malformed column_transformer step" }; }
  let err = CheckCellMatrixDims(cells, step.n_features_in);
  // Each branch transforms the whole selected sub-matrix once; outputs are
  // then concatenated row by row in branch order.
  let branch_outs: Cell[][][] = [];
  if (err == "") {
    for (let b = 0; b < ct.branches.length() && err == ""; b += 1) {
      let branch = ct.branches[b];
      let sub: Cell[][] = [];
      for (let i = 0; i < cells.length(); i += 1) {
        sub.push(SubRow(cells[i], branch.columns));
      }
      branch_outs.push(ObservedColumnTransformerBranch(sub, TracePrivate<ColumnTransformerBranch> { value: branch }, index, opts, exec, $trace = BranchOptions(opts, index, branch.name)));
    }
  }
  let out: Cell[][] = [];
  if (err == "") {
    for (let i = 0; i < cells.length(); i += 1) {
      let row: Cell[] = [];
      for (let b = 0; b < branch_outs.length(); b += 1) {
        let part = branch_outs[b];
        if (part.length() > i) {
          for (let j = 0; j < part[i].length(); j += 1) {
            row.push(part[i][j]);
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


// @contract(btel-v1)
function ObservedColumnTransformer(cells: Cell[][], state: TracePrivate<PipelineStep>, index: int, opts: trace.Options, exec: PipelineExecution) -> Cell[][] throws InferenceFailure {
  let r = ColumnTransformerStepCore(cells, state.value, opts, index, exec);
  if (!r.ok) { throw InferenceFailure { stage: "column_transform", message: r.error }; }
  r.cells
}

// The estimator consumes a dense finite float64 matrix; missing or
// non-numeric values that survived preprocessing are rejected here.

function RunPipelineEstimator(x: float[][], a: FittedPipelineArtifact, operation: Operation, full: bool, compute_probs: bool, opts: trace.Options) -> PipelineRun throws InferenceFailure {
  let e = a.estimator;
  if (e.kind == "linear") {
    if (operation == Operation.PredictProba) {
      throw InferenceFailure { stage: "input_validation", message: "predict-proba requires a classifier estimator" };
    }
    let r = ObservedLinear(x, TracePrivate<LinearRegressionModel> { value: LinearRegressionModel { coefficients: e.weights[0], intercept: e.intercepts[0], n_features: e.n_features_in } }, $trace = opts);
    return PipelineRun { values: r.value, labels: [], probs: [] };
  }
  if (e.kind == "logistic_binary" || e.kind == "logistic_multinomial") {
    let model = PipelineLogisticModel(e);
    let scores = ObservedDecisionScores(x, TracePrivate<LogisticRegressionModel> { value: model }, $trace = opts);
    if (operation == Operation.PredictProba) {
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
  if (e.kind == "random_forest_classifier") {
    let r = ObservedForestTraversal(x, TracePrivate<ForestClassifierState> { value: ForestClassifierState { n_features: e.n_features_in, classes: e.classes, trees: e.trees } }, full, $trace = opts);
    return PipelineRun { values: [], labels: r.labels, probs: r.probs };
  }
  throw InferenceFailure { stage: "prediction", message: "artifact: unsupported estimator" };
}


// @contract(btel-v1)
function PipelineModelLoaded(artifact: FittedPipelineArtifact) -> void {}


// @contract(btel-v1)
function PredictLoadedPipeline(cells: Cell[][], features: string[], model_id: string, expected: string[], state: TracePrivate<FittedPipelineArtifact>, observation: ObservationMode, operation: Operation) -> PipelineResult throws InferenceFailure {
  let err = LoadedFeatureError(features, expected);
  if (err != "") { throw InferenceFailure { stage: "input_validation", message: err }; }
  let a = state.value;
  err = ValidatePipelineInput(cells, a);
  if (err != "") { throw InferenceFailure { stage: "input_validation", message: err }; }
  let opts = ObservationOptions(observation);
  let exec = PipelineExecution { z_abs_max: null };
  let out = RunPipelineSteps(cells, a, opts, exec);
  let x = PipelineEstimatorMatrix(out);
  let run = RunPipelineEstimator(x, a, operation, observation == ObservationMode.Full, false, opts);
  PipelineResult { ok: true, values: run.values, labels: run.labels, probs: run.probs, error: "" }
}

// Internal: caller owns the validated artifact and resolved policy.

