// Selective execution: semantic pass, trigger-driven Full replays, summaries.
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


// @contract(btel-v1)
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


// @contract(btel-v1)
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


// @contract(btel-v1)
function AdaptiveSummary() -> void {}


// @contract(btel-v1)
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


// @contract(btel-v1)
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


