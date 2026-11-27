class NativeBenchResult {
  ok bool
  calls int
  rows int
  elapsed_ns bigint
  checksum float
}

function ScaleModel(d: int) -> ClassificationPipeline {
  let means: float[] = [];
  let scales: float[] = [];
  let weights: float[][] = [[],[],[]];
  for (let j = 0; j < d; j += 1) {
    means.push(0.0);
    scales.push(1.0);
    for (let k = 0; k < 3; k += 1) {
      weights[k].push((float((j+k)%7)-3.0)/float(d));
    }
  }
  ClassificationPipeline {
    imputer: MeanImputer { means: means, n_features: d },
    scaler: StandardScaler { means: means, scales: scales, n_features: d },
    model: LogisticRegressionModel { classes: [0,1,2], weights: weights, intercepts: [0.1,0.0,-0.1], n_features: d, n_iter: 0, converged: true }
  }
}

function ScaleRows(d: int, n: int, offset: int) -> float[][] {
  let x: float[][] = [];
  for (let i = 0; i < n; i += 1) {
    let row: float[] = [];
    for (let j = 0; j < d; j += 1) {
      row.push(float(((i+offset)*17+j*13)%101)/50.0-1.0);
    }
    // Ensure distinct rows even beyond the periodic component.
    row[0] = float(i+offset)/10000.0;
    x.push(row);
  }
  x
}

function ScalePredict(d: int, batch: int, n: int, distinct: bool) -> NativeBenchResult {
  let model = ScaleModel(d);
  let inputs: float[][][] = [];
  let count = 1;
  if (distinct) { count = n+10; }
  for (let i = 0; i < count; i += 1) {
    inputs.push(ScaleRows(d,batch,i*batch));
  }
  let ok = true;
  for (let w = 0; w < 10; w += 1) {
    let warm = ClassificationPipelinePredictProba(inputs[w%count],model);
    if (!warm.ok) { ok = false; }
  }
  let checksum = 0.0;
  let start = baml.time.Instant.now();
  for (let i = 0; i < n; i += 1) {
    let r = ClassificationPipelinePredictProba(inputs[(i+10)%count],model);
    if (!r.ok) { ok = false; }
    else {
      for (let j = 0; j < r.probs.length(); j += 1) {
        checksum = checksum + r.probs[j][0]+2.0*r.probs[j][1]+3.0*r.probs[j][2];
      }
    }
  }
  let elapsed = start.elapsed().to_nanoseconds();
  NativeBenchResult { ok: ok, calls: n, rows: n*batch, elapsed_ns: elapsed, checksum: checksum }
}

function ScaleFit(n: int) -> NativeBenchResult {
  let x = ScaleRows(4,n,0);
  let y: int[] = [];
  for (let i = 0; i < n; i += 1) { y.push(i%3); }
  let warm = ClassificationPipelineFit(x,y,1.0,100,0.000001);
  let ok = warm.ok;
  let checksum = 0.0;
  let start = baml.time.Instant.now();
  let fit = ClassificationPipelineFit(x,y,1.0,100,0.000001);
  if (!fit.ok) { ok = false; }
  else {
    for (let k = 0; k < fit.pipeline.model.weights.length(); k += 1) {
      for (let j = 0; j < fit.pipeline.model.weights[k].length(); j += 1) {
        checksum = checksum + Abs(fit.pipeline.model.weights[k][j]);
      }
      checksum = checksum + Abs(fit.pipeline.model.intercepts[k]);
    }
  }
  let elapsed = start.elapsed().to_nanoseconds();
  NativeBenchResult { ok: ok, calls: 1, rows: n, elapsed_ns: elapsed, checksum: checksum }
}
function D4() -> NativeBenchResult { ScalePredict(4,1,1000,false) }
function D32() -> NativeBenchResult { ScalePredict(32,1,1000,false) }
function D128() -> NativeBenchResult { ScalePredict(128,1,1000,false) }
function D512() -> NativeBenchResult { ScalePredict(512,1,1000,false) }
function B32() -> NativeBenchResult { ScalePredict(32,32,100,false) }
function B256() -> NativeBenchResult { ScalePredict(32,256,100,false) }
function B1024() -> NativeBenchResult { ScalePredict(32,1024,100,false) }
function Distinct100() -> NativeBenchResult { ScalePredict(32,1,100,true) }
function Repeat100() -> NativeBenchResult { ScalePredict(32,1,100,false) }
function Distinct1000() -> NativeBenchResult { ScalePredict(32,1,1000,true) }
function Fit150() -> NativeBenchResult { ScaleFit(150) }
function Fit1000() -> NativeBenchResult { ScaleFit(1000) }
function Fit10000() -> NativeBenchResult { ScaleFit(10000) }
