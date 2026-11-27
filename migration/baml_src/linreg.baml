// Inference-only implementation; training code is preserved at pre-inference-cleanup.

class LinearRegressionModel {
  coefficients float[]
  intercept float
  n_features int
}

function EmptyLinRegModel() -> LinearRegressionModel {
  LinearRegressionModel { coefficients: [], intercept: 0.0, n_features: 0 }
}

function LinearRegressionPredict(x: float[][], model: LinearRegressionModel) -> NumArrayResult {
  let err = "";
  if (model.n_features <= 0 || model.coefficients.length() != model.n_features) {
    err = "linreg_predict: invalid fitted model state";
  }
  if (err == "" && !IsFinite(model.intercept)) {
    err = "linreg_predict: fitted model contains non-finite parameters";
  }
  if (err == "") {
    for (let j = 0; j < model.n_features && err == ""; j += 1) {
      if (!IsFinite(model.coefficients[j])) {
        err = "linreg_predict: fitted model contains non-finite parameters";
      }
    }
  }
  let out: float[] = [];
  if (err == "") {
    let d = CheckFiniteMatrix(x);
    err = d.error;
    if (err == "" && d.cols != model.n_features) {
      err = "linreg_predict: feature count differs from fitted model";
    }
    if (err == "") {
      let non_finite_pred = false;
      for (let i = 0; i < x.length(); i += 1) {
        let pred = model.intercept + DotUnchecked(model.coefficients, x[i]);
        if (!IsFinite(pred)) {
          non_finite_pred = true;
        }
        out.push(pred);
      }
      if (non_finite_pred) {
        err = "linreg_predict: prediction is not finite";
      }
    }
  }
  if (err == "") {
    NumArrayResult { ok: true, value: out, error: "" }
  } else {
    NumArrayResult { ok: false, value: [], error: err }
  }
}
