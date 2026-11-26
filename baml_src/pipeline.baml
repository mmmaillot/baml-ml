// Minimal Pipeline API (issue #7):
// SimpleImputer (mean) -> StandardScaler -> LinearRegression or
// LogisticRegression.
// Named `RegressionPipeline*` / `ClassificationPipeline*` explicitly because
// BAML 0.226.3 beta has no function overloading or union narrowing.
//
// Fitted state is an explicit pipeline value holding the three fitted step
// states. Preprocessing and prediction are separate named BAML functions so
// later tickets can observe their execution, and downstream steps are skipped
// when an upstream step fails so traces are not polluted by bogus calls on
// empty inputs.

class RegressionPipeline {
  imputer MeanImputer
  scaler StandardScaler
  model LinearRegressionModel
}

class RegressionPipelineFitResult {
  ok bool
  pipeline RegressionPipeline
  error string
}

function EmptyRegressionPipeline() -> RegressionPipeline {
  RegressionPipeline { imputer: EmptyImputer(), scaler: EmptyScaler(), model: EmptyLinRegModel() }
}

// Shared impute-then-scale helper usable by both pipeline variants.
function PipelinePreprocessSteps(x: float[][], imputer: MeanImputer, scaler: StandardScaler) -> MatrixResult {
  let err = "";
  if (imputer.n_features <= 0 || scaler.n_features != imputer.n_features) {
    err = "pipeline_preprocess: invalid fitted preprocessing state";
  }
  let out: float[][] = [];
  if (err == "") {
    let t1 = ImputerTransform(x, imputer);
    err = t1.error;
    if (err == "") {
      let t2 = ScalerTransform(t1.x, scaler);
      err = t2.error;
      if (err == "") {
        out = t2.x;
      }
    }
  }
  if (err == "") {
    MatrixResult { ok: true, value: out, error: "" }
  } else {
    MatrixResult { ok: false, value: [], error: err }
  }
}

// Fit every step sequentially on the training data only: impute, then scale
// the imputed matrix, then fit the regressor on the scaled matrix.
function RegressionPipelineFit(x: float[][], y: float[]) -> RegressionPipelineFitResult {
  let f1 = ImputerFit(x);
  let err = f1.error;
  let imp = EmptyImputer();
  let sc = EmptyScaler();
  let model = EmptyLinRegModel();
  if (err == "") {
    imp = f1.imputer;
    let t1 = ImputerTransform(x, imp);
    err = t1.error;
    if (err == "") {
      let f2 = ScalerFit(t1.x);
      err = f2.error;
      if (err == "") {
        sc = f2.scaler;
        let t2 = ScalerTransform(t1.x, sc);
        err = t2.error;
        if (err == "") {
          let f3 = LinearRegressionFit(t2.x, y);
          err = f3.error;
          if (err == "") {
            model = f3.model;
          }
        }
      }
    }
  }
  if (err == "") {
    RegressionPipelineFitResult { ok: true, pipeline: RegressionPipeline { imputer: imp, scaler: sc, model: model }, error: "" }
  } else {
    RegressionPipelineFitResult { ok: false, pipeline: EmptyRegressionPipeline(), error: err }
  }
}

// Named preprocessing entry point: impute then scale with the stored states.
function RegressionPipelinePreprocess(x: float[][], pipeline: RegressionPipeline) -> MatrixResult {
  let err = "";
  if (pipeline.imputer.n_features <= 0 || pipeline.scaler.n_features != pipeline.imputer.n_features || pipeline.model.n_features != pipeline.imputer.n_features) {
    err = "pipeline_preprocess: invalid fitted pipeline state";
  }
  let out: float[][] = [];
  if (err == "") {
    let p = PipelinePreprocessSteps(x, pipeline.imputer, pipeline.scaler);
    err = p.error;
    if (err == "") {
      out = p.value;
    }
  }
  if (err == "") {
    MatrixResult { ok: true, value: out, error: "" }
  } else {
    MatrixResult { ok: false, value: [], error: err }
  }
}

// Named prediction entry point: preprocess, then predict with the stored model.
function RegressionPipelinePredict(x: float[][], pipeline: RegressionPipeline) -> NumArrayResult {
  let p = RegressionPipelinePreprocess(x, pipeline);
  let err = p.error;
  let preds: float[] = [];
  if (err == "") {
    let r = LinearRegressionPredict(p.value, pipeline.model);
    err = r.error;
    if (err == "") {
      preds = r.value;
    }
  }
  if (err == "") {
    NumArrayResult { ok: true, value: preds, error: "" }
  } else {
    NumArrayResult { ok: false, value: [], error: err }
  }
}

// Classification half of the Pipeline (issue #7): SimpleImputer (mean) ->
// StandardScaler -> LogisticRegression, exposing predict and predict_proba
// entry points.

class ClassificationPipeline {
  imputer MeanImputer
  scaler StandardScaler
  model LogisticRegressionModel
}

class ClassificationPipelineFitResult {
  ok bool
  pipeline ClassificationPipeline
  error string
}

function EmptyClassificationPipeline() -> ClassificationPipeline {
  ClassificationPipeline { imputer: EmptyImputer(), scaler: EmptyScaler(), model: EmptyLogRegModel() }
}

// Fit every step sequentially on the training data only: impute, then scale
// the imputed matrix, then fit the classifier on the scaled matrix. The
// classifier hyperparameters (c, max_iter, tol) are passed through to
// LogRegFit (beta has no default parameter values).
function ClassificationPipelineFit(x: float[][], y: int[], c: float, max_iter: int, tol: float) -> ClassificationPipelineFitResult {
  let f1 = ImputerFit(x);
  let err = f1.error;
  let imp = EmptyImputer();
  let sc = EmptyScaler();
  let model = EmptyLogRegModel();
  if (err == "") {
    imp = f1.imputer;
    let t1 = ImputerTransform(x, imp);
    err = t1.error;
    if (err == "") {
      let f2 = ScalerFit(t1.x);
      err = f2.error;
      if (err == "") {
        sc = f2.scaler;
        let t2 = ScalerTransform(t1.x, sc);
        err = t2.error;
        if (err == "") {
          let f3 = LogRegFit(t2.x, y, c, max_iter, tol);
          err = f3.error;
          if (err == "") {
            model = f3.model;
          }
        }
      }
    }
  }
  if (err == "") {
    ClassificationPipelineFitResult { ok: true, pipeline: ClassificationPipeline { imputer: imp, scaler: sc, model: model }, error: "" }
  } else {
    ClassificationPipelineFitResult { ok: false, pipeline: EmptyClassificationPipeline(), error: err }
  }
}

// Named preprocessing entry point: impute then scale with the stored states.
function ClassificationPipelinePreprocess(x: float[][], pipeline: ClassificationPipeline) -> MatrixResult {
  let err = "";
  if (pipeline.imputer.n_features <= 0 || pipeline.scaler.n_features != pipeline.imputer.n_features || pipeline.model.n_features != pipeline.imputer.n_features) {
    err = "pipeline_preprocess: invalid fitted pipeline state";
  }
  let out: float[][] = [];
  if (err == "") {
    let p = PipelinePreprocessSteps(x, pipeline.imputer, pipeline.scaler);
    err = p.error;
    if (err == "") {
      out = p.value;
    }
  }
  if (err == "") {
    MatrixResult { ok: true, value: out, error: "" }
  } else {
    MatrixResult { ok: false, value: [], error: err }
  }
}

// Named predict_proba entry point: preprocess, then compute class probabilities
// with the stored model.
function ClassificationPipelinePredictProba(x: float[][], pipeline: ClassificationPipeline) -> ProbaResult {
  let p = ClassificationPipelinePreprocess(x, pipeline);
  let err = p.error;
  let probs: float[][] = [];
  if (err == "") {
    let r = LogRegPredictProba(p.value, pipeline.model);
    err = r.error;
    if (err == "") {
      probs = r.probs;
    }
  }
  if (err == "") {
    ProbaResult { ok: true, probs: probs, error: "" }
  } else {
    ProbaResult { ok: false, probs: [], error: err }
  }
}

// Named prediction entry point: preprocess, then predict class labels with the
// stored model.
function ClassificationPipelinePredict(x: float[][], pipeline: ClassificationPipeline) -> IntArrayResult {
  let p = ClassificationPipelinePreprocess(x, pipeline);
  let err = p.error;
  let preds: int[] = [];
  if (err == "") {
    let r = LogRegPredict(p.value, pipeline.model);
    err = r.error;
    if (err == "") {
      preds = r.value;
    }
  }
  if (err == "") {
    IntArrayResult { ok: true, value: preds, error: "" }
  } else {
    IntArrayResult { ok: false, value: [], error: err }
  }
}
