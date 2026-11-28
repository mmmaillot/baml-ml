// Champion/challenger shadow orchestration.
class ShadowInput {
    features: string[],
    rows: (float | string | null)[][],
    classes: int[]?,
}


class ShadowSummary {
    total_rows: int,
    triggered_rows: int,
    escalated_rows: int,
    capped: bool,
    label_disagreements: int,
    max_score_delta: float,
    triggers: TriggerCounts,
    challenger_ok: bool,
    challenger_error: InferenceFailure?,
}


class ShadowOutput {
    schema_version: int,
    operation: string,
    estimator_kind: string,
    champion_model_id: string,
    challenger_model_id: string,
    classes: int[],
    predictions: float[] | int[] | float[][],
    summary: ShadowSummary,
}


function PrepareShadowModel(raw: string) -> SelectiveModel {
    let version = ReadArtifactVersion(raw, $trace = trace.timing());
    if (version == 2) {
        let tree = PrepareTreeArtifact(raw, $trace = trace.empty_span());
        return SelectiveTree(tree, PreparedTree(tree));
    }
    if (version == 3) {
        return SelectivePipeline(PreparePipelineArtifact(raw, $trace = trace.empty_span()));
    }
    let a = PrepareArtifact(raw, null, $trace = trace.empty_span());
    if (a.estimator_kind == "linear") {
        return SelectiveLinear(a, ArtifactRegression(a));
    }
    SelectiveLogistic(a, ArtifactClassification(a))
}


function CheckShadowCompatibility(
    champion: SelectiveModel,
    challenger: SelectiveModel,
    input: ShadowInput,
    operation: string,
) -> void {
    if (
        champion.schema_version != challenger.schema_version
            || champion.estimator_kind != challenger.estimator_kind
    ) {
        throw baml.errors.InvalidArgument {
            message: "shadow: models must have the same schema version and estimator kind",
        };
    }
    if (operation == "predict-proba" && champion.estimator_kind == "linear") {
        throw baml.errors.InvalidArgument {
            message: "shadow: predict-proba requires a classifier",
        };
    }
    if (
        LoadedFeatureError(champion.features, challenger.features) != ""
            || LoadedFeatureError(input.features, champion.features) != ""
    ) {
        throw baml.errors.InvalidArgument {
            message: "shadow: model and batch feature order must match",
        };
    }
    let input_classes = input.classes;
    if (
        champion.classes != challenger.classes
            || (input_classes != null && champion.classes != input_classes)
    ) {
        throw baml.errors.InvalidArgument {
            message: "shadow: ordered classes must match",
        };
    }
}


function ShadowResult(
    champion: SelectiveModel,
    challenger: SelectiveModel,
    operation: string,
    result: SelectivePrediction,
    summary: ShadowSummary,
) -> ShadowOutput {
    ShadowOutput {
        schema_version: 1,
        operation: operation,
        estimator_kind: champion.estimator_kind,
        champion_model_id: champion.model_id,
        challenger_model_id: challenger.model_id,
        classes: champion.classes,
        predictions: result.for_operation(operation, champion.estimator_kind != "linear"),
        summary: summary,
    }
}


// @contract(btel-v1)
function ShadowChallengerFailed(failure: InferenceFailure) -> void throws InferenceFailure {
    throw failure;
}


// @contract(btel-v1)
function ShadowEscalatedRow(
    champion: SelectiveModel,
    challenger: SelectiveModel,
    row: Cell[],
    operation: string,
) -> void throws InferenceFailure {
    RunSelectiveModel(
        champion,
        [row],
        operation,
        ObservationMode.Full,
        $trace = SelectiveContext(
            champion,
            trace.span(inputs = false, output = false),
            "champion",
        ),
    );
    RunSelectiveModel(
        challenger,
        [row],
        operation,
        ObservationMode.Full,
        $trace = SelectiveContext(
            challenger,
            trace.span(inputs = false, output = false),
            "challenger",
        ),
    );
}


function CaptureShadowSnapshots(champion: SelectiveModel, challenger: SelectiveModel) -> void {
    CaptureLazySnapshot(
        champion,
        $trace = SelectiveContext(champion, trace.context(), "champion"),
    );
    CaptureLazySnapshot(
        challenger,
        $trace = SelectiveContext(challenger, trace.context(), "challenger"),
    );
}


// @contract(btel-v1)
function ShadowBatch(
    champion: SelectiveModel,
    challenger: SelectiveModel,
    rows: Cell[][],
    operation: string,
    policy: AdaptivePolicy,
) -> ShadowOutput {
    RecordSelectiveModelLoad(
        champion,
        SelectiveContext(champion, trace.empty_span(), "champion"),
    );
    RecordSelectiveModelLoad(
        challenger,
        SelectiveContext(challenger, trace.empty_span(), "challenger"),
    );
    let champion_signals: AdaptiveRowSignals[] = [];
    let challenger_signals: AdaptiveRowSignals[] = [];
    let result = RunSelectiveModel(
        champion,
        rows,
        operation,
        ObservationMode.Semantic,
        signals = champion_signals,
        $trace = SelectiveContext(champion, trace.empty_span(), "champion"),
    );
    RunSelectiveModel(
        challenger,
        rows,
        operation,
        ObservationMode.Semantic,
        signals = challenger_signals,
        $trace = SelectiveContext(challenger, trace.empty_span(), "challenger"),
    ) catch (e) {
        let failure: InferenceFailure => {
            CaptureShadowSnapshots(champion, challenger);
            ShadowChallengerFailed(
                failure,
                $trace = SelectiveContext(challenger, trace.span(), "challenger").context(
                    metadata = { "baml_ml.challenger_isolated": true },
                ),
            ) catch (recorded) {
                let expected: InferenceFailure => {},
            };
            return ShadowResult(champion, challenger, operation, result, ShadowSummary {
                total_rows: rows.length(),
                triggered_rows: 0,
                escalated_rows: 0,
                capped: false,
                label_disagreements: 0,
                max_score_delta: 0.0,
                triggers: EmptyTriggerCounts(),
                challenger_ok: false,
                challenger_error: failure,
            });
        },
    };
    let evaluation = EvaluateTriggers(policy, champion_signals, challenger = challenger_signals);
    if (evaluation.escalated_rows > 0) {
        CaptureShadowSnapshots(champion, challenger);
        for (let row in evaluation.rows) {
            ShadowEscalatedRow(
                champion,
                challenger,
                rows[row.row_index],
                operation,
                $trace = EscalationOptions(row),
            );
        }
    }
    ShadowResult(champion, challenger, operation, result, ShadowSummary {
        total_rows: rows.length(),
        triggered_rows: evaluation.triggered_rows,
        escalated_rows: evaluation.escalated_rows,
        capped: evaluation.capped,
        label_disagreements: evaluation.label_disagreements,
        max_score_delta: evaluation.max_score_delta,
        triggers: evaluation.triggers,
        challenger_ok: true,
        challenger_error: null,
    })
}
