// Inference-only implementation; training code is preserved at pre-inference-cleanup.

class LogisticRegressionModel {
  classes int[]
  weights float[][]
  intercepts float[]
  n_features int
}

class ProbaResult {
  ok bool
  probs float[][]
  error string
}

function EmptyLogRegModel() -> LogisticRegressionModel {
  LogisticRegressionModel { classes: [], weights: [], intercepts: [], n_features: 0 }
}

// Stable softmax: subtract the largest logit before exponentiation.
function SoftmaxRow(logits: float[]) -> float[] {
  let m = logits[0];
  for (let i = 1; i < logits.length(); i += 1) {
    if (logits[i] > m) {
      m = logits[i];
    }
  }
  let exps: float[] = [];
  let s = 0.0;
  for (let i = 0; i < logits.length(); i += 1) {
    let e = Exp(logits[i] - m);
    exps.push(e);
    s = s + e;
  }
  let out: float[] = [];
  for (let i = 0; i < logits.length(); i += 1) {
    out.push(exps[i] / s);
  }
  out
}

function ModelRowLogits(row: float[], model: LogisticRegressionModel) -> float[] {
  let out: float[] = [];
  for (let k = 0; k < model.classes.length(); k += 1) {
    out.push(model.intercepts[k] + DotUnchecked(model.weights[k], row));
  }
  out
}

function LogRegPredictProba(x: float[][], model: LogisticRegressionModel) -> ProbaResult {
  LogRegPredictProbaCore(x, model, false)
}

// Internal only: finite_input is true after successful preprocessing.
function LogRegPredictProbaCore(x: float[][], model: LogisticRegressionModel, finite_input: bool, validated_state: bool = false) -> ProbaResult {
  let err = "";
  if (!validated_state && (model.n_features <= 0 || model.classes.length() < 2 || model.weights.length() != model.classes.length() || model.intercepts.length() != model.classes.length())) {
    err = "logreg_predict_proba: invalid fitted model state";
  }
  if (!validated_state && err == "") {
    for (let k = 0; k < model.weights.length() && err == ""; k += 1) {
      if (model.weights[k].length() != model.n_features) {
        err = "logreg_predict_proba: invalid fitted model state";
      }
      for (let j = 0; j < model.weights[k].length() && err == ""; j += 1) {
        if (!IsFinite(model.weights[k][j])) {
          err = "logreg_predict_proba: fitted model contains non-finite weights";
        }
      }
      if (err == "" && !IsFinite(model.intercepts[k])) {
        err = "logreg_predict_proba: fitted model contains non-finite intercepts";
      }
    }
  }
  let d = if (finite_input) { CheckMatrix(x) } else { CheckFiniteMatrix(x) };
  if (err == "" && d.error != "") {
    err = d.error;
  }
  if (err == "" && d.cols != model.n_features) {
    err = "logreg_predict_proba: feature count differs from fitted model";
  }
  let out: float[][] = [];
  if (err == "") {
    let non_finite = false;
    for (let i = 0; i < x.length(); i += 1) {
      let row = SoftmaxRow(ModelRowLogits(x[i], model));
      for (let k = 0; k < row.length(); k += 1) {
        if (!IsFinite(row[k])) {
          non_finite = true;
        }
      }
      out.push(row);
    }
    if (non_finite) {
      err = "logreg_predict_proba: probability is not finite";
    }
  }
  if (err == "") {
    ProbaResult { ok: true, probs: out, error: "" }
  } else {
    ProbaResult { ok: false, probs: [], error: err }
  }
}

function LogRegPredict(x: float[][], model: LogisticRegressionModel) -> IntArrayResult {
  LogRegPredictCore(x, model, false)
}

function LogRegPredictCore(x: float[][], model: LogisticRegressionModel, finite_input: bool) -> IntArrayResult {
  let pr = LogRegPredictProbaCore(x, model, finite_input);
  let err = pr.error;
  let out: int[] = [];
  if (err == "") {
    for (let i = 0; i < pr.probs.length(); i += 1) {
      let best = 0;
      for (let k = 1; k < model.classes.length(); k += 1) {
        if (pr.probs[i][k] > pr.probs[i][best]) {
          best = k;
        }
      }
      out.push(model.classes[best]);
    }
  }
  if (err == "") {
    IntArrayResult { ok: true, value: out, error: "" }
  } else {
    IntArrayResult { ok: false, value: [], error: err }
  }
}
