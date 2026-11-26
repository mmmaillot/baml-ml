// Focused migration smoke/contract tests. Full historical suite is #27.
test "imputer_training_only" {
  let fit = ImputerFit([[1.0, float.nan()], [3.0, 8.0]]);
  assert.is_true(fit.ok)
  assert.equal(fit.imputer.means, [2.0, 8.0])
  let pred = ImputerTransform([[float.nan(), 100.0]], fit.imputer);
  assert.is_true(pred.ok)
  assert.equal(pred.x, [[2.0, 100.0]])
  assert.is_true(!ImputerFit([[float.nan()], [float.nan()]]).ok)
  assert.is_true(!ImputerFit([[float.inf()]]).ok)
  assert.is_true(!ImputerTransform([[1.0]], fit.imputer).ok)
}
test "scaler_fit_transform_contract" {
  let fit = ScalerFitTransform([[1.0, 5.0], [3.0, 5.0]]);
  assert.is_true(fit.ok)
  assert.equal(fit.scaler.means, [2.0, 5.0])
  assert.equal(fit.scaler.scales, [1.0, 1.0])
  assert.equal(fit.x, [[-1.0, 0.0], [1.0, 0.0]])
  assert.is_true(!ScalerFit([[float.nan()]]).ok)
  assert.is_true(!ScalerTransform([[1.0]], fit.scaler).ok)
}
test "linear_regression_fit_predict" {
  let fit = LinearRegressionFit([[0.0], [1.0], [2.0], [3.0]], [1.0, 3.0, 5.0, 7.0]);
  assert.is_true(fit.ok)
  let pred = LinearRegressionPredict([[4.0]], fit.model);
  assert.is_true(pred.ok)
  assert.is_true(ApproxRelative(pred.value[0], 9.0, 1e-12))
  assert.is_true(!LinearRegressionFit([[1.0], [1.0]], [1.0, 2.0]).ok)
  assert.is_true(!LinearRegressionFit([[1.0], [2.0]], [1.0]).ok)
  assert.is_true(!LinearRegressionPredict([[float.inf()]], fit.model).ok)
}
test "logreg_softmax_and_errors" {
  let probs = SoftmaxRow([1000.0, 1000.0, -1000.0]);
  assert.equal(probs, [0.5, 0.5, 0.0])
  assert.equal(UniqueSortedClasses([7, -2, 7, 3]), [-2, 3, 7])
  assert.is_true(!LogRegFit([[1.0], [2.0]], [0, 0], 1.0, 5000, 1e-8).ok)
  assert.is_true(!LogRegFit([[1.0], [2.0]], [0, 1], float.nan(), 5000, 1e-8).ok)
  assert.is_true(!LogRegFit([[float.nan()], [2.0]], [0, 1], 1.0, 5000, 1e-8).ok)
}
test "pipeline_validation" {
  assert.is_true(!RegressionPipelineFit([[float.nan()], [float.nan()]], [1.0, 2.0]).ok)
  assert.is_true(!ClassificationPipelineFit([[1.0], [2.0]], [0], 1.0, 5000, 1e-8).ok)
  assert.is_true(!ClassificationPipelineFit([[1.0], [2.0]], [0, 1], 1.0, 0, 1e-8).ok)
  assert.is_true(!RegressionPipelinePredict([[1.0]], EmptyRegressionPipeline()).ok)
  assert.is_true(!ClassificationPipelinePredict([[1.0]], EmptyClassificationPipeline()).ok)
}
