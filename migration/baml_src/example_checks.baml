// These entry points execute the entire pipeline in BAML. The Python fixture
// generator supplies data and independent sklearn reference values only.
class ExampleResult {
  ok bool
  error string
  max_prediction_diff float
  max_imputer_mean_diff float
  max_scaler_mean_diff float
  labels_match bool
}

function ExampleFailure(error: string) -> ExampleResult {
  ExampleResult { ok: false, error: error, max_prediction_diff: 0.0, max_imputer_mean_diff: 0.0, max_scaler_mean_diff: 0.0, labels_match: false }
}

// Explicit checks prevent NaN, empty output, or malformed dimensions from
// silently passing a max-error test.
function MaxReferenceDiff(actual: float[], expected: float[]) -> float {
  if (actual.length() != expected.length() || actual.length() == 0) {
    return float.inf();
  }
  let max_diff = 0.0;
  for (let i = 0; i < actual.length(); i += 1) {
    if (!IsFinite(actual[i]) || !IsFinite(expected[i])) {
      return float.inf();
    }
    let diff = Abs(actual[i] - expected[i]);
    if (diff > max_diff) {
      max_diff = diff;
    }
  }
  max_diff
}

function CheckIrisExample(x_train: float[][], y_train: int[], x_test: float[][], expected: float[][], labels: int[], imputer_means: float[], scaler_means: float[]) -> ExampleResult {
  let fit = ClassificationPipelineFit(x_train, y_train, 1.0, 5000, 1e-8);
  if (!fit.ok) { return ExampleFailure(fit.error); }
  let p = ClassificationPipelinePredictProba(x_test, fit.pipeline);
  if (!p.ok) { return ExampleFailure(p.error); }
  let pred = ClassificationPipelinePredict(x_test, fit.pipeline);
  if (!pred.ok) { return ExampleFailure(pred.error); }
  if (p.probs.length() != expected.length() || p.probs.length() == 0) {
    return ExampleFailure("probability row count mismatch");
  }
  let diff = 0.0;
  for (let i = 0; i < expected.length(); i += 1) {
    let d = MaxReferenceDiff(p.probs[i], expected[i]);
    if (d > diff) { diff = d; }
  }
  let labels_match = pred.value.length() == labels.length();
  if (labels_match) {
    for (let i = 0; i < labels.length(); i += 1) {
      if (pred.value[i] != labels[i]) { labels_match = false; }
    }
  }
  ExampleResult {
    ok: true, error: "", max_prediction_diff: diff,
    max_imputer_mean_diff: MaxReferenceDiff(fit.pipeline.imputer.means, imputer_means),
    max_scaler_mean_diff: MaxReferenceDiff(fit.pipeline.scaler.means, scaler_means),
    labels_match: labels_match
  }
}

function CheckDiabetesExample(x_train: float[][], y_train: float[], x_test: float[][], expected: float[], imputer_means: float[], scaler_means: float[]) -> ExampleResult {
  let fit = RegressionPipelineFit(x_train, y_train);
  if (!fit.ok) { return ExampleFailure(fit.error); }
  let pred = RegressionPipelinePredict(x_test, fit.pipeline);
  if (!pred.ok) { return ExampleFailure(pred.error); }
  ExampleResult {
    ok: true, error: "",
    max_prediction_diff: MaxReferenceDiff(pred.value, expected),
    max_imputer_mean_diff: MaxReferenceDiff(fit.pipeline.imputer.means, imputer_means),
    max_scaler_mean_diff: MaxReferenceDiff(fit.pipeline.scaler.means, scaler_means),
    labels_match: true
  }
}
