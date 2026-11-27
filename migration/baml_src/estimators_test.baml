// Fitted states are supplied directly; training is external.
test "imputer_transform_contract" {
  let imp = MeanImputer { means: [2.0,8.0], n_features: 2 };
  let pred = ImputerTransform([[float.nan(),100.0]],imp);
  assert.is_true(pred.ok)
  assert.equal(pred.x,[[2.0,100.0]])
  assert.is_true(!ImputerTransform([[1.0]],imp).ok)
  assert.is_true(!ImputerTransform([[float.inf(),1.0]],imp).ok)
  let bad = MeanImputer { means: [float.nan(),8.0], n_features: 2 };
  assert.is_true(!ImputerTransform([[1.0,2.0]],bad).ok)
}
test "scaler_transform_contract" {
  let s = StandardScaler { means: [2.0,5.0], scales: [1.0,1.0], n_features: 2 };
  let p = ScalerTransform([[1.0,5.0],[3.0,5.0]],s);
  assert.is_true(p.ok)
  assert.equal(p.x,[[-1.0,0.0],[1.0,0.0]])
  assert.is_true(!ScalerTransform([[float.nan(),5.0]],s).ok)
  assert.is_true(!ScalerTransform([[1.0]],s).ok)
}
test "linear_regression_predict_contract" {
  let m = LinearRegressionModel { coefficients: [2.0], intercept: 1.0, n_features: 1 };
  let p = LinearRegressionPredict([[4.0]],m);
  assert.is_true(p.ok)
  assert.is_true(ApproxRelative(p.value[0],9.0,1e-12))
  assert.is_true(!LinearRegressionPredict([[float.inf()]],m).ok)
  assert.is_true(!LinearRegressionPredict([[1.0,2.0]],m).ok)
  let bad = LinearRegressionModel { coefficients: [float.nan()], intercept: 1.0, n_features: 1 };
  assert.is_true(!LinearRegressionPredict([[4.0]],bad).ok)
  let huge = LinearRegressionModel { coefficients: [1e308], intercept: 0.0, n_features: 1 };
  assert.is_true(!LinearRegressionPredict([[1e308]],huge).ok)
}
test "logreg_softmax_extreme_scores" {
  assert.equal(SoftmaxRow([1000.0,1000.0,-1000.0]),[0.5,0.5,0.0])
}
test "pipeline_invalid_state" {
  assert.is_true(!RegressionPipelinePredict([[1.0]],EmptyRegressionPipeline()).ok)
  assert.is_true(!ClassificationPipelinePredict([[1.0]],EmptyClassificationPipeline()).ok)
}
