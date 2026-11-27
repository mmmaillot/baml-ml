// Inference-only implementation; training code is preserved at pre-inference-cleanup.

class RegressionPipeline {
  imputer MeanImputer
  scaler StandardScaler
  model LinearRegressionModel
}

function EmptyRegressionPipeline() -> RegressionPipeline {
  RegressionPipeline { imputer: EmptyImputer(), scaler: EmptyScaler(), model: EmptyLinRegModel() }
}

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
      // Successful imputation preserves shape and guarantees finite values.
      let t2 = ScalerTransformCore(t1.x, scaler, true);
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

class ClassificationPipeline {
  imputer MeanImputer
  scaler StandardScaler
  model LogisticRegressionModel
}

function EmptyClassificationPipeline() -> ClassificationPipeline {
  ClassificationPipeline { imputer: EmptyImputer(), scaler: EmptyScaler(), model: EmptyLogRegModel() }
}

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

function ClassificationPipelinePredictProba(x: float[][], pipeline: ClassificationPipeline) -> ProbaResult {
  let p = ClassificationPipelinePreprocess(x, pipeline);
  let err = p.error;
  let probs: float[][] = [];
  if (err == "") {
    // Successful scaling checks every computed value for finiteness.
    let r = LogRegPredictProbaCore(p.value, pipeline.model, true);
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

function ClassificationPipelinePredict(x: float[][], pipeline: ClassificationPipeline) -> IntArrayResult {
  let p = ClassificationPipelinePreprocess(x, pipeline);
  let err = p.error;
  let preds: int[] = [];
  if (err == "") {
    let r = LogRegPredictCore(p.value, pipeline.model, true);
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
