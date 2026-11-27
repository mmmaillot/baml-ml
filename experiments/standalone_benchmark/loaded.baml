// Benchmark-only pack target. Use the same loaded-model API as predict.
class LoadedBenchmark {
  read_ns bigint
  prepare_ns bigint
  decode_ns bigint
  inference_ns bigint
  predictions float[][]
}

function loaded(model: string, input: string, calls: int, warmups: int) -> LoadedBenchmark {
  let reading = baml.time.Instant.now();
  let raw_model = baml.fs.read(model);
  let raw_input = baml.fs.read(input);
  let read_ns = reading.elapsed().to_nanoseconds();

  let preparing = baml.time.Instant.now();
  let m = PrepareLogisticModel(raw_model, observation = ObservationMode.Timing);
  let prepare_ns = preparing.elapsed().to_nanoseconds();

  let decoding = baml.time.Instant.now();
  let batch = DecodeBatchInput(raw_input);
  let rows = BatchRows(batch);
  let decode_ns = decoding.elapsed().to_nanoseconds();

  for (let i = 0; i < warmups; i += 1) {
    let r = m.predict_proba(rows, batch.features);
    if (!r.ok) { throw baml.errors.InvalidArgument { message: r.error }; }
  }

  let result: float[][] = [];
  let predicting = baml.time.Instant.now();
  for (let i = 0; i < calls; i += 1) {
    let r = m.predict_proba(rows, batch.features);
    if (!r.ok) { throw baml.errors.InvalidArgument { message: r.error }; }
    result = r.probs;
  }
  let inference_ns = predicting.elapsed().to_nanoseconds();
  LoadedBenchmark { read_ns: read_ns, prepare_ns: prepare_ns, decode_ns: decode_ns, inference_ns: inference_ns, predictions: result }
}
