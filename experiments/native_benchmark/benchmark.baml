// Only native timing in the benchmark harness, not in the library.
// Instant.now is a wall clock, not a monotonic timer. Reject negative samples.
class NativeBenchResult {
  ok bool
  calls int
  rows int
  elapsed_ns bigint
  checksum float
}

function BenchPredict(n: int, batch_size: int) -> NativeBenchResult {
  let model = IrisFrozenPipeline();
  let x: float[][] = [];
  for (let j = 0; j < batch_size; j += 1) {
    if (j % 2 == 0) { x.push([5.1, float.nan(), 1.4, 0.2]); }
    else { x.push([6.7, 3.1, 4.7, 1.5]); }
  }
  let ok = true;
  for (let w = 0; w < 10; w += 1) {
    let warm = ClassificationPipelinePredictProba(x, model);
    if (!warm.ok) { ok = false; }
  }
  let checksum = 0.0;
  let start = baml.time.Instant.now();
  for (let i = 0; i < n; i += 1) {
    let r = ClassificationPipelinePredictProba(x, model);
    if (!r.ok) { ok = false; }
    else {
      for (let j = 0; j < r.probs.length(); j += 1) {
        checksum = checksum + r.probs[j][0] + 2.0 * r.probs[j][1] + 3.0 * r.probs[j][2];
      }
    }
  }
  let elapsed = start.elapsed().to_nanoseconds();
  NativeBenchResult { ok: ok, calls: n, rows: n * batch_size, elapsed_ns: elapsed, checksum: checksum }
}

function BenchFit() -> NativeBenchResult {
  let x = NativeTrainX();
  let y = NativeTrainY();
  let warm = ClassificationPipelineFit(x,y,1.0,5000,0.00000001);
  let ok = warm.ok;
  let checksum = 0.0;
  let start = baml.time.Instant.now();
  for (let i = 0; i < 3; i += 1) {
    let fit = ClassificationPipelineFit(x,y,1.0,5000,0.00000001);
    if (!fit.ok) { ok = false; }
    else {
      for (let k = 0; k < fit.pipeline.model.weights.length(); k += 1) {
        for (let j = 0; j < fit.pipeline.model.weights[k].length(); j += 1) {
          checksum = checksum + Abs(fit.pipeline.model.weights[k][j]);
        }
        checksum = checksum + Abs(fit.pipeline.model.intercepts[k]);
      }
    }
  }
  let elapsed = start.elapsed().to_nanoseconds();
  NativeBenchResult { ok: ok, calls: 3, rows: 360, elapsed_ns: elapsed, checksum: checksum }
}

function BenchZero() -> NativeBenchResult { BenchPredict(0, 1) }
function Bench100() -> NativeBenchResult { BenchPredict(100, 1) }
function Bench1000() -> NativeBenchResult { BenchPredict(1000, 1) }
function Bench5000() -> NativeBenchResult { BenchPredict(5000, 1) }
function BenchBatch() -> NativeBenchResult { BenchPredict(100, 30) }
