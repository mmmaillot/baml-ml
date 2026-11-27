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
  if (value == "timing") { return ObservationMode.Timing; }
  throw baml.errors.InvalidArgument { message: "observation must be semantic, full or timing; set BAML_TELEMETRY=off to disable recording" };
}

function RunBatch(raw_model: string, raw_input: string, operation: string, mode: ObservationMode) -> BatchOutput {
  if (operation != "predict" && operation != "predict-proba") {
    throw baml.errors.InvalidArgument { message: "operation must be predict or predict-proba" };
  }
  let opts = ObservationOptions(mode);
  // One decode and validation, even when dispatching dynamically by estimator.
  let a = PrepareArtifact(raw_model, null, $trace = if (mode == ObservationMode.Full) { trace.span(inputs = false, output = false) } else { opts });
  let input = DecodeBatchInput(raw_input);
  let rows = BatchRows(input);
  if (a.estimator_kind == "linear") {
    if (operation != "predict") {
      throw baml.errors.InvalidArgument { message: "predict-proba requires logistic regression" };
    }
    let m = BuildLoadedLinear(a,mode);
    let r = m.predict(rows,input.features);
    if (!r.ok) { throw baml.errors.InvalidArgument { message: r.error }; }
    return BatchOutput { schema_version: 1, model_id: a.model_id, estimator_kind: a.estimator_kind, operation: operation, classes: [], predictions: r.value };
  }
  let m = BuildLoadedLogistic(a,mode);
  if (operation == "predict-proba") {
    let r = m.predict_proba(rows,input.features);
    if (!r.ok) { throw baml.errors.InvalidArgument { message: r.error }; }
    return BatchOutput { schema_version: 1, model_id: a.model_id, estimator_kind: a.estimator_kind, operation: operation, classes: a.classes, predictions: r.probs };
  }
  let r = m.predict(rows,input.features);
  if (!r.ok) { throw baml.errors.InvalidArgument { message: r.error }; }
  BatchOutput { schema_version: 1, model_id: a.model_id, estimator_kind: a.estimator_kind, operation: operation, classes: a.classes, predictions: r.value }
}

/// Run a fitted sklearn pipeline on a JSON batch. Use --output-format json.
/// Output is written by the native pack host to stdout; redirect to a file.
function predict(model: string, input: string, operation: string = "predict", observation: string = "semantic") -> BatchOutput {
  let mode = ParseObservationMode(observation);
  RunBatch(baml.fs.read(model), baml.fs.read(input), operation, mode)
}
