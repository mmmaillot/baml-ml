function ValidationPipeline() -> ClassificationPipeline {
  ClassificationPipeline {
    imputer: MeanImputer { means: [2.0], n_features: 1 },
    scaler: StandardScaler { means: [1.0], scales: [2.0], n_features: 1 },
    model: LogisticRegressionModel { classes: [0,1], weights: [[1.0],[-1.0]], intercepts: [0.0,0.0], n_features: 1 }
  }
}

function ValidationReference(x: float[][], p: ClassificationPipeline) -> ProbaResult {
  let a = ImputerTransform(x,p.imputer);
  if (!a.ok) { ProbaResult { ok: false, probs: [], error: a.error } }
  else {
    let b = ScalerTransform(a.x,p.scaler);
    if (!b.ok) { ProbaResult { ok: false, probs: [], error: b.error } }
    else { LogRegPredictProba(b.x,p.model) }
  }
}

test "trusted_prediction_matches_checked_composition" {
  let p = ValidationPipeline();
  let x = [[float.nan()],[3.0],[-1.0]];
  let a = ClassificationPipelinePredictProba(x,p);
  let b = ValidationReference(x,p);
  assert.is_true(a.ok)
  assert.equal(a.probs,b.probs)
  assert.equal(ClassificationPipelinePredict(x,p).value,[0,0,1])
  assert.equal(ClassificationPipelinePreprocess(x,p).value,[[0.5],[1.0],[-1.0]])
}

test "trusted_prediction_rejects_bad_inputs" {
  let p = ValidationPipeline();
  assert.equal(ClassificationPipelinePredictProba([],p).error,"matrix has no rows")
  assert.equal(ClassificationPipelinePredictProba([[]],p).error,"matrix has zero columns")
  assert.equal(ClassificationPipelinePredictProba([[1.0],[1.0,2.0]],p).error,"matrix is ragged: rows have different lengths")
  assert.equal(ClassificationPipelinePredictProba([[1.0,2.0]],p).error,"imputer_transform: feature count differs from fitted imputer")
  assert.equal(ClassificationPipelinePredictProba([[float.inf()]],p).error,"imputer_transform: input contains infinite values")
  assert.equal(ClassificationPipelinePredict([[float.inf()]],p).error,"imputer_transform: input contains infinite values")
}

test "public_transforms_still_check_nonfinite_inputs" {
  let p = ValidationPipeline();
  assert.equal(ScalerTransform([[float.nan()]],p.scaler).error,"matrix contains NaN or infinite values")
  assert.equal(LogRegPredictProba([[float.inf()]],p.model).error,"matrix contains NaN or infinite values")
  assert.equal(LogRegPredict([[float.nan()]],p.model).error,"matrix contains NaN or infinite values")
}

test "trusted_prediction_checks_scaler_and_model_parameters" {
  let p = ValidationPipeline();
  let bad_scale = ClassificationPipeline { imputer: p.imputer, scaler: StandardScaler { means: [1.0], scales: [0.0], n_features: 1 }, model: p.model };
  assert.equal(ClassificationPipelinePredictProba([[1.0]],bad_scale).error,"scaler_transform: fitted scaler contains invalid scales")
  let bad_model = ClassificationPipeline { imputer: p.imputer, scaler: p.scaler, model: LogisticRegressionModel { classes: [0,1], weights: [[float.nan()],[-1.0]], intercepts: [0.0,0.0], n_features: 1 } };
  assert.equal(ClassificationPipelinePredictProba([[1.0]],bad_model).error,"logreg_predict_proba: fitted model contains non-finite weights")
  assert.equal(ClassificationPipelinePredictProba([[float.inf()]],bad_scale).error,"imputer_transform: input contains infinite values")
  assert.equal(ClassificationPipelinePredictProba([[1.0]],EmptyClassificationPipeline()).error,"pipeline_preprocess: invalid fitted pipeline state")
}

test "trusted_prediction_checks_arithmetic_overflow" {
  let p = ValidationPipeline();
  let overflow = ClassificationPipeline { imputer: p.imputer, scaler: StandardScaler { means: [-1e308], scales: [1.0], n_features: 1 }, model: p.model };
  assert.equal(ClassificationPipelinePredictProba([[1e308]],overflow).error,"scaler_transform: standardized value is not finite")
  let huge_model = ClassificationPipeline { imputer: p.imputer, scaler: p.scaler, model: LogisticRegressionModel { classes: [0,1], weights: [[1e308],[-1e308]], intercepts: [0.0,0.0], n_features: 1 } };
  let a = ClassificationPipelinePredictProba([[1e308]],huge_model);
  let b = ValidationReference([[1e308]],huge_model);
  assert.equal(a.ok,b.ok)
  assert.equal(a.error,b.error)
  assert.is_true(!a.ok)
}
