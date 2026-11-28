// Standalone CLI entry points (predict, shadow) and batch dispatch.
// Standalone JSON batch contract. The native pack host generates the CLI.

class BatchInput {
  features string[]
  rows: (float?)[][]
}


class BatchOutput {
  schema_version int
  model_id string
  estimator_kind string
  operation string
  classes int[]
  predictions: float[] | int[] | float[][]
}


function DecodeBatchInput(raw: string) -> BatchInput {
  baml.json.decode<BatchInput>(raw)
}


function BatchRows(input: BatchInput) -> float[][] {
  let out: float[][] = [];
  for (let i = 0; i < input.rows.length(); i += 1) {
    let row: float[] = [];
    for (let j = 0; j < input.rows[i].length(); j += 1) {
      let v = input.rows[i][j];
      row.push(v ?? float.nan());
    }
    out.push(row);
  }
  out
}


function ParseObservationMode(value: string) -> ObservationMode {
  if (value == "semantic") { return ObservationMode.Semantic; }
  if (value == "full") { return ObservationMode.Full; }
  if (value == "adaptive") { return ObservationMode.Adaptive; }
  if (value == "timing") { return ObservationMode.Timing; }
  throw baml.errors.InvalidArgument { message: "observation must be semantic, full, timing or adaptive; set BAML_TELEMETRY=off to disable recording" };
}


// @public-api
function RunBatch(raw_model: string, raw_input: string, operation: string, mode: ObservationMode, policy: AdaptivePolicy? = null) -> BatchOutput {
  let op = ParseOperation(operation);
  let adaptive = ResolveAdaptivePolicy(mode, policy);
  let opts = ObservationOptions(mode);
  // Read only the schema tag for dispatch; fitted state is decoded and validated once.
  let version = ReadArtifactVersion(raw_model, $trace = trace.timing());
  if (version == 2) { return RunTreeBatch(raw_model,raw_input,op,mode,adaptive, $trace = trace.timing()); }
  if (version == 3) { return RunPipelineBatch(raw_model,raw_input,op,mode,adaptive, $trace = trace.timing()); }
  // Existing v1 decoding and validation are unchanged.
  let a = PrepareArtifact(raw_model, null, $trace = if (mode == ObservationMode.Full) { trace.span(inputs = false, output = false) } else { opts });
  let input = DecodeBatchInput(raw_input);
  let rows = BatchRows(input);
  if (a.estimator_kind == "linear") {
    if (op != Operation.Predict) {
      throw baml.errors.InvalidArgument { message: "predict-proba requires logistic regression" };
    }
    let m = BuildLoadedLinear(a,mode,adaptive);
    let r = m.predict(rows,input.features);
    if (!r.ok) { throw baml.errors.InvalidArgument { message: r.error }; }
    return BatchOutput { schema_version: 1, model_id: a.model_id, estimator_kind: a.estimator_kind, operation: OperationName(op), classes: [], predictions: r.value };
  }
  let m = BuildLoadedLogistic(a,mode,adaptive);
  if (op == Operation.PredictProba) {
    let r = m.predict_proba(rows,input.features);
    if (!r.ok) { throw baml.errors.InvalidArgument { message: r.error }; }
    return BatchOutput { schema_version: 1, model_id: a.model_id, estimator_kind: a.estimator_kind, operation: OperationName(op), classes: a.classes, predictions: r.probs };
  }
  let r = m.predict(rows,input.features);
  if (!r.ok) { throw baml.errors.InvalidArgument { message: r.error }; }
  BatchOutput { schema_version: 1, model_id: a.model_id, estimator_kind: a.estimator_kind, operation: OperationName(op), classes: a.classes, predictions: r.value }
}

/// Run a fitted sklearn pipeline on a JSON batch. Use --output-format json.
/// Output is written by the native pack host to stdout; redirect to a file.

// @public-api
function predict(model: string, input: string, operation: string = "predict", observation: string = "semantic", policy: string = "") -> BatchOutput {
  let mode = ParseObservationMode(observation);
  if (policy != "" && mode != ObservationMode.Adaptive) {
    throw AdaptivePolicyError { message: "policy requires adaptive observation" };
  }
  let overrides = if (policy == "") { null } else { ReadAdaptivePolicy(policy) };
  RunBatch(baml.fs.read(model), baml.fs.read(input), operation, mode, policy = overrides)
}

/// Compare same-family fitted pipelines. Predictions on stdout come from the champion.

// @public-api
function shadow(champion: string, challenger: string, input: string, operation: string = "predict", policy: string = "") -> ShadowOutput {
  RunShadow(baml.fs.read(champion), baml.fs.read(challenger), baml.fs.read(input), operation, ReadAdaptivePolicy(policy))
}

// @public-api
function RunShadow(
    raw_champion: string,
    raw_challenger: string,
    raw_input: string,
    operation: string,
    policy: AdaptivePolicy,
) -> ShadowOutput {
    let op = ParseOperation(operation);
    ValidateAdaptivePolicy(policy);
    let champion = PrepareShadowModel(raw_champion);
    let challenger = PrepareShadowModel(raw_challenger);
    let input = baml.json.decode<ShadowInput>(raw_input);
    CheckShadowCompatibility(champion, challenger, input, op);
    let rows = ShadowCells(input.rows);
    ShadowBatch(
        champion,
        challenger,
        rows,
        op,
        policy,
        $trace = trace.empty_span().context(metadata = {
            "baml_ml.champion_model_id": champion.model_id,
            "baml_ml.challenger_model_id": challenger.model_id,
            "baml_ml.estimator_kind": champion.estimator_kind,
            "baml_ml.operation": OperationName(op),
            "baml_ml.contract_version": 1,
        }),
    )
}


function RunTreeBatch(raw_model: string, raw_input: string, operation: Operation, mode: ObservationMode, adaptive: AdaptivePolicy?) -> BatchOutput {
  let opts = ObservationOptions(mode);
  let a = PrepareTreeArtifact(raw_model, $trace = if (mode == ObservationMode.Full) { trace.span(inputs = false, output = false) } else { opts });
  let input = DecodeBatchInput(raw_input);
  let rows = BatchRows(input);
  let m = BuildLoadedTree(a,mode,adaptive);
  if (operation == Operation.PredictProba) {
    let r = m.predict_proba(rows,input.features);
    if (!r.ok) { throw baml.errors.InvalidArgument { message: r.error }; }
    return BatchOutput { schema_version: 1, model_id: a.model_id, estimator_kind: a.estimator_kind, operation: OperationName(operation), classes: a.classes, predictions: r.probs };
  }
  let r = m.predict(rows,input.features);
  if (!r.ok) { throw baml.errors.InvalidArgument { message: r.error }; }
  BatchOutput { schema_version: 1, model_id: a.model_id, estimator_kind: a.estimator_kind, operation: OperationName(operation), classes: a.classes, predictions: r.value }
}


function RunPipelineBatch(raw_model: string, raw_input: string, operation: Operation, mode: ObservationMode, adaptive: AdaptivePolicy?) -> BatchOutput {
  let opts = ObservationOptions(mode);
  let a = PreparePipelineArtifact(raw_model, $trace = if (mode == ObservationMode.Full) { trace.span(inputs = false, output = false) } else { opts });
  let input = baml.json.decode<TabularBatchInput>(raw_input);
  let cells = BatchCells(input);
  let m = BuildLoadedPipeline(a, mode, adaptive);
  if (operation == Operation.PredictProba) {
    let r = m.predict_proba(cells, input.features);
    if (!r.ok) { throw baml.errors.InvalidArgument { message: r.error }; }
    return BatchOutput { schema_version: 1, model_id: a.model_id, estimator_kind: a.estimator.kind, operation: OperationName(operation), classes: a.estimator.classes, predictions: r.probs };
  }
  let r = m.predict(cells, input.features);
  if (!r.ok) { throw baml.errors.InvalidArgument { message: r.error }; }
  if (a.estimator.kind == "linear") {
    return BatchOutput { schema_version: 1, model_id: a.model_id, estimator_kind: a.estimator.kind, operation: OperationName(operation), classes: [], predictions: r.values };
  }
  BatchOutput { schema_version: 1, model_id: a.model_id, estimator_kind: a.estimator.kind, operation: OperationName(operation), classes: a.estimator.classes, predictions: r.labels }
}

