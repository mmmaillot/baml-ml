// Minimal Pipeline: SimpleImputer (mean) -> StandardScaler -> LinearRegression.
// Regression variant only; the classification variant (predict_proba) lands
// with issue #6 (LogisticRegression). Fitted state is an explicit Pipeline
// value holding the three fitted step states. Preprocessing and prediction are
// separate named BAML functions so later tickets can observe their execution.

class Pipeline {
  imputer MeanImputer
  scaler StandardScaler
  model LinearRegressionModel
}

class PipelineFitResult {
  ok bool
  pipeline Pipeline
  error string
}

function EmptyPipeline() -> Pipeline {
  Pipeline { imputer: EmptyImputer(), scaler: EmptyScaler(), model: EmptyLinRegModel() }
}

// Fit every step sequentially on the training data only: impute, then scale
// the imputed matrix, then fit the regressor on the scaled matrix.
function PipelineFit(x: float[][], y: float[]) -> PipelineFitResult {
  let f1 = ImputerFit(x);
  let err = f1.error;
  let imp = EmptyImputer();
  if (err == "") {
    imp = f1.imputer;
  }
  let t1 = ImputerTransform(x, imp);
  if (err == "") {
    err = t1.error;
  }
  let x1: float[][] = [];
  if (err == "") {
    x1 = t1.x;
  }
  let f2 = ScalerFit(x1);
  if (err == "") {
    err = f2.error;
  }
  let sc = EmptyScaler();
  if (err == "") {
    sc = f2.scaler;
  }
  let t2 = ScalerTransform(x1, sc);
  if (err == "") {
    err = t2.error;
  }
  let x2: float[][] = [];
  if (err == "") {
    x2 = t2.x;
  }
  let f3 = LinearRegressionFit(x2, y);
  if (err == "") {
    err = f3.error;
  }
  let model = EmptyLinRegModel();
  if (err == "") {
    model = f3.model;
  }
  if (err == "") {
    PipelineFitResult { ok: true, pipeline: Pipeline { imputer: imp, scaler: sc, model: model }, error: "" }
  } else {
    PipelineFitResult { ok: false, pipeline: EmptyPipeline(), error: err }
  }
}

// Named preprocessing entry point: impute then scale with the stored states.
function PipelinePreprocess(x: float[][], pipeline: Pipeline) -> MatrixResult {
  let t1 = ImputerTransform(x, pipeline.imputer);
  let err = t1.error;
  let x1: float[][] = [];
  if (err == "") {
    x1 = t1.x;
  }
  let t2 = ScalerTransform(x1, pipeline.scaler);
  if (err == "") {
    err = t2.error;
  }
  let out: float[][] = [];
  if (err == "") {
    out = t2.x;
  }
  if (err == "") {
    MatrixResult { ok: true, value: out, error: "" }
  } else {
    MatrixResult { ok: false, value: [], error: err }
  }
}

// Named prediction entry point: preprocess, then predict with the stored model.
function PipelinePredict(x: float[][], pipeline: Pipeline) -> NumArrayResult {
  let p = PipelinePreprocess(x, pipeline);
  let err = p.error;
  let r = LinearRegressionPredict(p.value, pipeline.model);
  if (err == "") {
    err = r.error;
  }
  let preds: float[] = [];
  if (err == "") {
    preds = r.value;
  }
  if (err == "") {
    NumArrayResult { ok: true, value: preds, error: "" }
  } else {
    NumArrayResult { ok: false, value: [], error: err }
  }
}
