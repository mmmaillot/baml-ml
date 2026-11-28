// SelectiveModel builders per estimator family.
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
            operation: Operation,
            mode: ObservationMode,
            signals: AdaptiveRowSignals[]?,
        ) -> SelectivePrediction {
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
            SelectivePrediction {
                values: run.values,
                labels: run.labels,
                probabilities: run.probs,
            }
        },
    }
}


function SelectiveLinear(a: FittedArtifact, p: RegressionPipeline) -> SelectiveModel {
    SelectiveModel {
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
        ) -> SelectivePrediction {
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
            SelectivePrediction {
                values: r.value,
                labels: [],
                probabilities: [],
            }
        },
    }
}


function SelectiveLogistic(a: FittedArtifact, p: ClassificationPipeline) -> SelectiveModel {
    SelectiveModel {
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
        ) -> SelectivePrediction {
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
                return SelectivePrediction {
                    values: [],
                    labels: [],
                    probabilities: result.probs,
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
            SelectivePrediction {
                values: [],
                labels: result.value,
                probabilities: [],
            }
        },
    }
}


function SelectiveTree(a: FittedTreeArtifact, p: PreparedTreePipeline) -> SelectiveModel {
    SelectiveModel {
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
        ) -> SelectivePrediction {
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
            SelectivePrediction {
                values: [],
                labels: r.labels,
                probabilities: r.probs,
            }
        },
    }
}


