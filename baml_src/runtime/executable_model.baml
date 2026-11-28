// Single execution engine shared by direct loaded-model inference (Timing,
// Semantic, Full), Adaptive and shadow. The factory owns one ExecutableModel
// per loaded handle; its run core executes preprocessing + terminal estimator
// and optionally collects row signals. Direct contract spans delegate to this
// core with signals = null, so no extra span is inserted in non-adaptive
// modes and the BTEL parent-child hierarchy is unchanged.

class ExecutableModel {
    model_id: string,
    schema_version: int,
    source_version: string,
    estimator_kind: string,
    features: string[],
    classes: int[],
    guard: SnapshotGuard,
    snapshot_fn: (trace.Options) -> void throws never,
    run_fn: (Cell[][], Operation, ObservationMode, AdaptiveRowSignals[]?) -> ExecutionResult
        throws InferenceFailure,
    function run(
        self,
        cells: Cell[][],
        operation: Operation,
        mode: ObservationMode,
        signals: AdaptiveRowSignals[]? = null,
    ) -> ExecutionResult throws InferenceFailure {
        let f = self.run_fn;
        f(cells, operation, mode, signals)
    }
}


class ExecutionResult {
    values: float[],
    labels: int[],
    probabilities: float[][],
    leaves: int[],
    paths: int[][],
    function for_operation(
        self,
        operation: Operation,
        is_classifier: bool,
    ) -> float[] | int[] | float[][] {
        if (operation == Operation.PredictProba) {
            return self.probabilities;
        }
        if (is_classifier) {
            return self.labels;
        }
        self.values
    }
}


function CollectRegressionSignals(
    signals: AdaptiveRowSignals[],
    x: float[][],
    scaled: float[][],
    values: float[],
) -> void {
    for (let i = 0; i < x.length(); i += 1) {
        signals.push(AdaptiveRowSignals {
            has_nan: HasMissingInput(x[i]),
            max_abs_z: MaxAbsoluteValue(scaled[i]),
            probabilities: [],
            label: null,
            prediction: values[i],
        });
    }
}


function CollectClassificationSignals(
    signals: AdaptiveRowSignals[],
    x: float[][],
    scaled: float[][],
    probabilities: float[][],
    labels: int[],
) -> void {
    for (let i = 0; i < x.length(); i += 1) {
        signals.push(AdaptiveRowSignals {
            has_nan: HasMissingInput(x[i]),
            max_abs_z: MaxAbsoluteValue(scaled[i]),
            probabilities: probabilities[i],
            label: labels[i],
            prediction: 0.0,
        });
    }
}


function CollectPipelineSignals(signals: AdaptiveRowSignals[], cells: Cell[][], exec: PipelineExecution, run: PipelineRun, is_classifier: bool) -> void {
  for (let i = 0; i < cells.length(); i += 1) {
    let has_nan = false;
    for (let j = 0; j < cells[i].length(); j += 1) {
      if (cells[i][j].kind == "null") { has_nan = true; }
    }
    let max_abs_z = 0.0;
    let z = exec.z_abs_max;
    if (z != null && i < z.length()) { max_abs_z = z[i]; }
    if (is_classifier) {
      signals.push(AdaptiveRowSignals { has_nan: has_nan, max_abs_z: max_abs_z, probabilities: run.probs[i], label: run.labels[i], prediction: 0.0 });
    } else {
      signals.push(AdaptiveRowSignals { has_nan: has_nan, max_abs_z: max_abs_z, probabilities: [], label: null, prediction: run.values[i] });
    }
  }
}


function ExecutablePipeline(a: FittedPipelineArtifact) -> ExecutableModel {
    ExecutableModel {
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
            operation: Operation,
            mode: ObservationMode,
            signals: AdaptiveRowSignals[]?,
        ) -> ExecutionResult {
            let err = ValidatePipelineInput(cells, a);
            if (err != "") {
                throw InferenceFailure { stage: "input_validation", message: err };
            }
            let opts = ObservationOptions(mode);
            let exec = PipelineExecution { z_abs_max: null };
            let out = RunPipelineSteps(cells, a, opts, exec);
            let x = PipelineEstimatorMatrix(out);
            let is_classifier = a.estimator.kind != "linear";
            let run = RunPipelineEstimator(x, a, operation, mode == ObservationMode.Full, signals != null, opts);
            if (signals != null) {
                CollectPipelineSignals(signals, cells, exec, run, is_classifier);
            }
            ExecutionResult {
                values: run.values,
                labels: run.labels,
                probabilities: run.probs,
                leaves: [],
                paths: [],
            }
        },
    }
}


function ExecutableLinear(a: FittedArtifact, p: RegressionPipeline) -> ExecutableModel {
    ExecutableModel {
        model_id: a.model_id,
        schema_version: a.schema_version,
        source_version: a.source_version,
        estimator_kind: a.estimator_kind,
        features: a.features,
        classes: a.classes,
        guard: SnapshotGuard { captured: false },
        snapshot_fn: (opts: trace.Options) -> void {
            ModelLoaded(a, $trace = opts);
        },
        run_fn: (
            cells: Cell[][],
            operation: Operation,
            mode: ObservationMode,
            signals: AdaptiveRowSignals[]?,
        ) -> ExecutionResult {
            let opts = ObservationOptions(mode);
            let x = CellsToFloatMatrix(cells);
            let scaled = PreparedPreprocess(x, p.imputer, p.scaler, opts);
            let r = ObservedLinear(
                scaled,
                TracePrivate<LinearRegressionModel> { value: p.model },
                $trace = opts,
            );
            if (signals != null) {
                CollectRegressionSignals(signals, x, scaled, r.value);
            }
            ExecutionResult {
                values: r.value,
                labels: [],
                probabilities: [],
                leaves: [],
                paths: [],
            }
        },
    }
}


function ExecutableLogistic(a: FittedArtifact, p: ClassificationPipeline) -> ExecutableModel {
    ExecutableModel {
        model_id: a.model_id,
        schema_version: a.schema_version,
        source_version: a.source_version,
        estimator_kind: a.estimator_kind,
        features: a.features,
        classes: a.classes,
        guard: SnapshotGuard { captured: false },
        snapshot_fn: (opts: trace.Options) -> void {
            ModelLoaded(a, $trace = opts);
        },
        run_fn: (
            cells: Cell[][],
            operation: Operation,
            mode: ObservationMode,
            signals: AdaptiveRowSignals[]?,
        ) -> ExecutionResult {
            let opts = ObservationOptions(mode);
            let x = CellsToFloatMatrix(cells);
            let scaled = PreparedPreprocess(x, p.imputer, p.scaler, opts);
            let scores = ObservedDecisionScores(
                scaled,
                TracePrivate<LogisticRegressionModel> { value: p.model },
                $trace = opts,
            );
            if (operation == Operation.PredictProba) {
                let result = ObservedSoftmax(scores, $trace = opts);
                if (signals != null) {
                    CollectClassificationSignals(
                        signals,
                        x,
                        scaled,
                        result.probs,
                        ClassLabels(scores, a.classes),
                    );
                }
                return ExecutionResult {
                    values: [],
                    labels: [],
                    probabilities: result.probs,
                    leaves: [],
                    paths: [],
                };
            }
            let result = ObservedLabels(scores, a.classes, $trace = opts);
            if (signals != null) {
                let probabilities: float[][] = [];
                for (let score in scores) {
                    probabilities.push(SoftmaxRow(score));
                }
                CollectClassificationSignals(signals, x, scaled, probabilities, result.value);
            }
            ExecutionResult {
                values: [],
                labels: result.value,
                probabilities: [],
                leaves: [],
                paths: [],
            }
        },
    }
}


function ExecutableTree(a: FittedTreeArtifact, p: PreparedTreePipeline) -> ExecutableModel {
    ExecutableModel {
        model_id: a.model_id,
        schema_version: a.schema_version,
        source_version: a.source_version,
        estimator_kind: a.estimator_kind,
        features: a.features,
        classes: a.classes,
        guard: SnapshotGuard { captured: false },
        snapshot_fn: (opts: trace.Options) -> void {
            TreeModelLoaded(a, $trace = opts);
        },
        run_fn: (
            cells: Cell[][],
            operation: Operation,
            mode: ObservationMode,
            signals: AdaptiveRowSignals[]?,
        ) -> ExecutionResult {
            let opts = ObservationOptions(mode);
            let x = CellsToFloatMatrix(cells);
            let scaled = PreparedPreprocess(x, p.imputer, p.scaler, opts);
            let r = ObservedTreeTraversal(
                scaled,
                TracePrivate<TreeClassifierState> { value: p.tree },
                mode == ObservationMode.Full,
                $trace = opts,
            );
            if (signals != null) {
                CollectClassificationSignals(signals, x, scaled, r.probs, r.labels);
            }
            ExecutionResult {
                values: [],
                labels: r.labels,
                probabilities: r.probs,
                leaves: r.leaves,
                paths: r.paths,
            }
        },
    }
}
