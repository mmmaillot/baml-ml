// Observation policy types and native trace option builders.
// Native observation policy, fixed for the lifetime of a loaded model.
// No exporter, collector or manual span bookkeeping.

enum ObservationMode {
  Timing,
  Semantic,
  Full,
  Adaptive,
}


// Redaction protects the captured state in the semantic call payloads, not
// arbitrary copies made by a caller or by additional tracing instrumentation.

class TracePrivate<T> {
  value: T @trace.redact
}


function ObservationOptions(mode: ObservationMode) -> trace.Options {
  if (mode == ObservationMode.Full) { return trace.span(); }
  if (mode == ObservationMode.Semantic || mode == ObservationMode.Adaptive) { return trace.empty_span(); }
  trace.timing()
}


function ModelObservation(mode: ObservationMode, model_id: string, schema_version: int, source_version: string) -> trace.Options {
  ObservationOptions(mode).context(metadata = {
    "baml_ml.contract_version": 1,
    "baml_ml.model_id": model_id,
    "baml_ml.schema_version": schema_version,
    "baml_ml.source_version": source_version
  })
}

// Called only after successful validation and conversion. Full mode records
// the artifact once; no parameter snapshot is returned to the caller.

