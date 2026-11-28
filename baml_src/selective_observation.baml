// Internal execution shared by Adaptive and shadow. The loaded-model closures
// and shadow orchestrator own the validated artifact and all fitted state.
class SelectiveModel {
    model_id: string,
    schema_version: int,
    source_version: string,
    estimator_kind: string,
    features: string[],
    classes: int[],
    guard: SnapshotGuard,
    snapshot_fn: (trace.Options) -> void throws never,
    run_fn: (Cell[][], string, ObservationMode, AdaptiveRowSignals[]?) -> SelectivePrediction
        throws InferenceFailure,
}

class SelectivePrediction {
    values: float[],
    labels: int[],
    probabilities: float[][],
    function for_operation(
        self,
        operation: string,
        is_classifier: bool,
    ) -> float[] | int[] | float[][] {
        if (operation == "predict-proba") {
            return self.probabilities;
        }
        if (is_classifier) {
            return self.labels;
        }
        self.values
    }
}

function SelectiveContext(
    model: SelectiveModel,
    opts: trace.Options,
    role: string,
) -> trace.Options {
    let with_model = opts.context(metadata = {
        "baml_ml.contract_version": 1,
        "baml_ml.model_id": model.model_id,
        "baml_ml.schema_version": model.schema_version,
        "baml_ml.source_version": model.source_version,
    });
    if (role == "") {
        return with_model;
    }
    with_model.context(metadata = { "baml_ml.shadow_role": role })
}

function RecordSelectiveModelLoad(model: SelectiveModel, opts: trace.Options) -> void {
    let record = model.snapshot_fn;
    record(opts);
}

function CaptureLazySnapshot(model: SelectiveModel) -> void {
    if (model.guard.claim()) {
        RecordSelectiveModelLoad(
            model,
            trace.span().context(metadata = {
                "baml_ml.lazy_snapshot": true,
            }),
        );
    }
}

function RunSelectiveModel(
    model: SelectiveModel,
    cells: Cell[][],
    operation: string,
    mode: ObservationMode,
    signals: AdaptiveRowSignals[]? = null,
) -> SelectivePrediction throws InferenceFailure {
    let run = model.run_fn;
    run(cells, operation, mode, signals)
}

function EscalatedRow(model: SelectiveModel, row: Cell[], operation: string) -> void
    throws InferenceFailure
{
    RunSelectiveModel(model, [row], operation, ObservationMode.Full);
}

function EscalationOptions(row: Escalation) -> trace.Options {
    trace.span(inputs = false, output = false).context(
        metadata = {
            "baml_ml.row_index": row.row_index,
            "baml_ml.trigger_reasons": row.trigger_reasons,
            "baml_ml.escalated": true,
        },
    )
}

function AdaptiveSummary() -> void {}

function RunAdaptiveCells(
    model: SelectiveModel,
    cells: Cell[][],
    features: string[],
    operation: string,
    policy: AdaptivePolicy,
) -> SelectivePrediction throws InferenceFailure {
    let error = LoadedFeatureError(features, model.features);
    if (error != "") {
        throw InferenceFailure {
            stage: "input_validation",
            message: error,
        };
    }
    let signals: AdaptiveRowSignals[] = [];
    let result = RunSelectiveModel(
        model,
        cells,
        operation,
        ObservationMode.Semantic,
        signals = signals,
        $trace = trace.empty_span(),
    );
    let evaluation = EvaluateTriggers(policy, signals);
    if (evaluation.triggered_rows > 0) {
        AdaptiveSummary(
            $trace = trace.empty_span().context(metadata = {
                "baml_ml.triggered_rows": evaluation.triggered_rows,
                "baml_ml.escalated_rows": evaluation.escalated_rows,
                "baml_ml.capped": evaluation.capped,
                "baml_ml.triggers.missing_input": evaluation.triggers.missing_input,
                "baml_ml.triggers.extreme_zscore": evaluation.triggers.extreme_zscore,
                "baml_ml.triggers.low_margin": evaluation.triggers.low_margin,
                "baml_ml.triggers.label_disagreement": evaluation.triggers.label_disagreement,
                "baml_ml.triggers.score_delta": evaluation.triggers.score_delta,
            }),
        );
    }
    if (evaluation.escalated_rows > 0) {
        CaptureLazySnapshot(model);
        for (let row in evaluation.rows) {
            EscalatedRow(
                model,
                cells[row.row_index],
                operation,
                $trace = EscalationOptions(row),
            );
        }
    }
    result
}

function RunAdaptive(
    model: SelectiveModel,
    x: float[][],
    features: string[],
    operation: string,
    policy: AdaptivePolicy,
) -> SelectivePrediction throws InferenceFailure {
    RunAdaptiveCells(model, NumericCells(x), features, operation, policy)
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
            operation: string,
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
            operation: string,
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
            if (operation == "predict-proba") {
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
            operation: string,
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

function AdaptiveLinear(model: SelectiveModel, policy: AdaptivePolicy) -> LoadedLinearModel {
    let opts = SelectiveContext(model, trace.empty_span(), "");
    LoadedLinearModel {
        predict_fn: (x: float[][], features: string[]) -> NumArrayResult {
            let r = RunAdaptive(model, x, features, "predict", policy, $trace = opts) catch (e) {
                let f: InferenceFailure => {
                    return NumArrayResult {
                        ok: false,
                        value: [],
                        error: f.message,
                    };
                },
            };
            NumArrayResult {
                ok: true,
                value: r.values,
                error: "",
            }
        },
    }
}

function AdaptivePipeline(model: SelectiveModel, policy: AdaptivePolicy) -> LoadedPipelineModel {
    let opts = SelectiveContext(model, trace.empty_span(), "");
    LoadedPipelineModel {
        estimator_kind: model.estimator_kind,
        classes: model.classes,
        predict_fn: (cells: Cell[][], features: string[]) -> PipelineResult {
            let r = RunAdaptiveCells(model, cells, features, "predict", policy, $trace = opts) catch (e) {
                let f: InferenceFailure => {
                    return PipelineResult {
                        ok: false,
                        values: [],
                        labels: [],
                        probs: [],
                        error: f.message,
                    };
                },
            };
            PipelineResult {
                ok: true,
                values: r.values,
                labels: r.labels,
                probs: [],
                error: "",
            }
        },
        proba_fn: (cells: Cell[][], features: string[]) -> PipelineResult {
            let r = RunAdaptiveCells(
                model,
                cells,
                features,
                "predict-proba",
                policy,
                $trace = opts,
            ) catch (e) {
                let f: InferenceFailure => {
                    return PipelineResult {
                        ok: false,
                        values: [],
                        labels: [],
                        probs: [],
                        error: f.message,
                    };
                },
            };
            PipelineResult {
                ok: true,
                values: [],
                labels: [],
                probs: r.probabilities,
                error: "",
            }
        },
    }
}

function AdaptiveClassifier(model: SelectiveModel, policy: AdaptivePolicy) -> LoadedLogisticModel {
    let opts = SelectiveContext(model, trace.empty_span(), "");
    LoadedLogisticModel {
        labels_fn: (x: float[][], features: string[]) -> IntArrayResult {
            let r = RunAdaptive(model, x, features, "predict", policy, $trace = opts) catch (e) {
                let f: InferenceFailure => {
                    return IntArrayResult {
                        ok: false,
                        value: [],
                        error: f.message,
                    };
                },
            };
            IntArrayResult {
                ok: true,
                value: r.labels,
                error: "",
            }
        },
        proba_fn: (x: float[][], features: string[]) -> ProbaResult {
            let r = RunAdaptive(
                model,
                x,
                features,
                "predict-proba",
                policy,
                $trace = opts,
            ) catch (e) {
                let f: InferenceFailure => {
                    return ProbaResult {
                        ok: false,
                        probs: [],
                        error: f.message,
                    };
                },
            };
            ProbaResult {
                ok: true,
                probs: r.probabilities,
                error: "",
            }
        },
    }
}
