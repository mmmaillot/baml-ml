// Selective execution: semantic pass, trigger-driven Full replays, summaries.
// Internal orchestration shared by Adaptive and shadow; the ExecutableModel
// engine (executable_model.baml) owns the validated artifact and all fitted
// state and performs the actual execution.


function SelectiveContext(
    model: ExecutableModel,
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


function RecordSelectiveModelLoad(model: ExecutableModel, opts: trace.Options) -> void {
    let record = model.snapshot_fn;
    record(opts);
}


function CaptureLazySnapshot(model: ExecutableModel) -> void {
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
    model: ExecutableModel,
    cells: Cell[][],
    operation: Operation,
    mode: ObservationMode,
    signals: AdaptiveRowSignals[]? = null,
) -> ExecutionResult throws InferenceFailure {
    model.run(cells, operation, mode, signals = signals)
}


// @contract(btel-v1)
function EscalatedRow(model: ExecutableModel, row: Cell[], operation: Operation) -> void
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
    model: ExecutableModel,
    cells: Cell[][],
    features: string[],
    operation: Operation,
    policy: AdaptivePolicy,
) -> ExecutionResult throws InferenceFailure {
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
    model: ExecutableModel,
    x: float[][],
    features: string[],
    operation: Operation,
    policy: AdaptivePolicy,
) -> ExecutionResult throws InferenceFailure {
    RunAdaptiveCells(model, NumericCells(x), features, operation, policy)
}
