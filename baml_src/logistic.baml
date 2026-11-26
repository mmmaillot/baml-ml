// Multiclass (multinomial) logistic regression with L2 regularization.
// All math runs in BAML (Exp/Log/Sqrt are implemented in primitives.baml).
//
// Objective (issue #6): sum of negative log-likelihoods + ||W||^2 / (2*C),
// intercepts excluded from the penalty. We optimize the consistently scaled
// equivalent mean-NLL + ||W||^2 / (2*C*n) (same argmin), where n = row count.
// C = 1.0 is the default usage (pass c = 1.0 from Python; beta has no default
// parameter values).
//
// Optimizer: gradient descent with Armijo backtracking line search,
// max_iter iterations, convergence when the gradient inf-norm <= tol.
// Non-convergence is reported as an error (ok = false), never silently.
//
// Class labels: int[], encoded as sorted unique classes (deterministic class
// order); predict returns the class label at the argmax (ties -> lowest class).

class LogisticRegressionModel {
  classes int[]
  weights float[][]
  intercepts float[]
  n_features int
  n_iter int
  converged bool
}

class LogRegParams {
  weights float[][]
  intercepts float[]
}

class LogRegFitResult {
  ok bool
  model LogisticRegressionModel
  error string
}

class ProbaResult {
  ok bool
  probs float[][]
  error string
}

function EmptyLogRegModel() -> LogisticRegressionModel {
  LogisticRegressionModel { classes: [], weights: [], intercepts: [], n_features: 0, n_iter: 0, converged: false }
}

// Sorted unique labels (insertion sort; class counts are tiny).
function UniqueSortedClasses(y: int[]) -> int[] {
  let out: int[] = [];
  for (let i = 0; i < y.length(); i += 1) {
    let v = y[i];
    let found = false;
    for (let j = 0; j < out.length(); j += 1) {
      if (out[j] == v) {
        found = true;
      }
    }
    if (!found) {
      out.push(v);
    }
  }
  for (let i = 1; i < out.length(); i += 1) {
    let key = out[i];
    let j = i - 1;
    while (j >= 0 && out[j] > key) {
      out[j + 1] = out[j];
      j = j - 1;
    }
    out[j + 1] = key;
  }
  out
}

function ClassIndex(classes: int[], label: int) -> int {
  let idx = 0;
  for (let k = 0; k < classes.length(); k += 1) {
    if (classes[k] == label) {
      idx = k;
    }
  }
  idx
}

function ZeroParams(k: int, p: int) -> LogRegParams {
  let w: float[][] = [];
  for (let i = 0; i < k; i += 1) {
    let row: float[] = [];
    for (let j = 0; j < p; j += 1) {
      row.push(0.0);
    }
    w.push(row);
  }
  let b: float[] = [];
  for (let i = 0; i < k; i += 1) {
    b.push(0.0);
  }
  LogRegParams { weights: w, intercepts: b }
}

// Stable softmax: subtract the max logit before Exp. The denominator is >= 1
// (the max term contributes Exp(0) = 1), so extreme logits stay finite and row
// probabilities sum to 1.
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

// Per-row logits from theta: intercepts[k] + dot(weights[k], x_row).
function ParamsRowLogits(row: float[], theta: LogRegParams) -> float[] {
  let out: float[] = [];
  for (let k = 0; k < theta.intercepts.length(); k += 1) {
    out.push(theta.intercepts[k] + DotUnchecked(theta.weights[k], row));
  }
  out
}

// Stable log-sum-exp: max + Log(sum(Exp(logits - max))); the Log argument is
// >= 1, so it never hits the x <= 0 branch.
function LogSumExp(logits: float[]) -> float {
  let m = logits[0];
  for (let i = 1; i < logits.length(); i += 1) {
    if (logits[i] > m) {
      m = logits[i];
    }
  }
  let s = 0.0;
  for (let i = 0; i < logits.length(); i += 1) {
    s = s + Exp(logits[i] - m);
  }
  m + Log(s)
}

// Internal (exposed on the client like other *Unchecked helpers): mean NLL +
// ||W||^2 / (2*C*n), the 1/n-scaled equivalent of the issue's objective.
function LogRegLoss(x: float[][], y: int[], classes: int[], theta: LogRegParams, c: float) -> float {
  let n = 0.0;
  for (let i = 0; i < x.length(); i += 1) {
    n = n + 1.0;
  }
  let loss = 0.0;
  for (let i = 0; i < x.length(); i += 1) {
    let logits = ParamsRowLogits(x[i], theta);
    let ti = ClassIndex(classes, y[i]);
    loss = loss + LogSumExp(logits) - logits[ti];
  }
  loss = loss / n;
  let pen = 0.0;
  for (let k = 0; k < theta.weights.length(); k += 1) {
    for (let j = 0; j < theta.weights[k].length(); j += 1) {
      pen = pen + theta.weights[k][j] * theta.weights[k][j];
    }
  }
  loss + pen / (2.0 * c * n)
}

// Internal: gradient of LogRegLoss; same shape as theta. Intercepts are not
// penalized.
function LogRegGradient(x: float[][], y: int[], classes: int[], theta: LogRegParams, c: float) -> LogRegParams {
  let n = 0.0;
  for (let i = 0; i < x.length(); i += 1) {
    n = n + 1.0;
  }
  let k = theta.intercepts.length();
  let p = 0;
  if (k > 0) {
    p = theta.weights[0].length();
  }
  let gw = ZeroParams(k, p).weights;
  let gb: float[] = [];
  for (let i = 0; i < k; i += 1) {
    gb.push(0.0);
  }
  for (let i = 0; i < x.length(); i += 1) {
    let logits = ParamsRowLogits(x[i], theta);
    let probs = SoftmaxRow(logits);
    let ti = ClassIndex(classes, y[i]);
    for (let kk = 0; kk < k; kk += 1) {
      let coeff = probs[kk];
      if (kk == ti) {
        coeff = coeff - 1.0;
      }
      gb[kk] = gb[kk] + coeff;
      for (let j = 0; j < p; j += 1) {
        gw[kk][j] = gw[kk][j] + coeff * x[i][j];
      }
    }
  }
  for (let kk = 0; kk < k; kk += 1) {
    gb[kk] = gb[kk] / n;
    for (let j = 0; j < p; j += 1) {
      gw[kk][j] = gw[kk][j] / n + theta.weights[kk][j] / (c * n);
    }
  }
  LogRegParams { weights: gw, intercepts: gb }
}

function GradInfNorm(g: LogRegParams) -> float {
  let m = 0.0;
  for (let k = 0; k < g.intercepts.length(); k += 1) {
    let a = Abs(g.intercepts[k]);
    if (a > m) {
      m = a;
    }
  }
  for (let k = 0; k < g.weights.length(); k += 1) {
    for (let j = 0; j < g.weights[k].length(); j += 1) {
      let a = Abs(g.weights[k][j]);
      if (a > m) {
        m = a;
      }
    }
  }
  m
}

function ParamsSubScale(theta: LogRegParams, grad: LogRegParams, t: float) -> LogRegParams {
  let w: float[][] = [];
  for (let k = 0; k < theta.weights.length(); k += 1) {
    let row: float[] = [];
    for (let j = 0; j < theta.weights[k].length(); j += 1) {
      row.push(theta.weights[k][j] - t * grad.weights[k][j]);
    }
    w.push(row);
  }
  let b: float[] = [];
  for (let k = 0; k < theta.intercepts.length(); k += 1) {
    b.push(theta.intercepts[k] - t * grad.intercepts[k]);
  }
  LogRegParams { weights: w, intercepts: b }
}

class LogRegOptResult {
  theta LogRegParams
  iters int
  converged bool
}

// Gradient descent with Armijo backtracking line search. Kept at function top
// level (no enclosing if-block) because the beta compiler miscompiles deeply
// nested statement blocks.
function LogRegOptimize(x: float[][], y: int[], classes: int[], c: float, max_iter: int, tol: float) -> LogRegOptResult {
  let k = classes.length();
  let p = x[0].length();
  let theta = ZeroParams(k, p);
  let loss = LogRegLoss(x, y, classes, theta, c);
  let converged = false;
  let done = false;
  let it_done = 0;
  let t = 1.0;
  for (let it = 0; it < max_iter && !done; it += 1) {
    let grad = LogRegGradient(x, y, classes, theta, c);
    let gnorm = GradInfNorm(grad);
    if (gnorm <= tol) {
      converged = true;
      done = true;
    }
    if (!done) {
      let accepted = false;
      let cand = theta;
      let closs = loss;
      for (let bt = 0; bt < 50 && !accepted; bt += 1) {
        let c2 = ParamsSubScale(theta, grad, t);
        let l2 = LogRegLoss(x, y, classes, c2, c);
        if (IsFinite(l2) && l2 <= loss - 0.0001 * t * gnorm * gnorm) {
          cand = c2;
          closs = l2;
          accepted = true;
        }
        if (!accepted) {
          t = t * 0.5;
        }
      }
      if (!accepted) {
        done = true;
      }
      if (accepted) {
        theta = cand;
        loss = closs;
        t = t * 2.0;
        it_done = it + 1;
      }
    }
  }
  LogRegOptResult { theta: theta, iters: it_done, converged: converged }
}

function LogRegFit(x: float[][], y: int[], c: float, max_iter: int, tol: float) -> LogRegFitResult {
  let d = CheckFiniteMatrix(x);
  let err = d.error;
  if (err == "" && y.length() != d.rows) {
    err = "logreg_fit: y length differs from row count of x";
  }
  if (err == "" && (!IsFinite(c) || c <= 0.0)) {
    err = "logreg_fit: c must be finite and positive";
  }
  if (err == "" && (!IsFinite(tol) || tol <= 0.0)) {
    err = "logreg_fit: tol must be finite and positive";
  }
  if (err == "" && max_iter < 1) {
    err = "logreg_fit: max_iter must be >= 1";
  }
  let classes: int[] = [];
  if (err == "") {
    classes = UniqueSortedClasses(y);
    if (classes.length() < 2) {
      err = "logreg_fit: need at least two distinct classes";
    }
  }
  let model = EmptyLogRegModel();
  if (err == "") {
    let opt = LogRegOptimize(x, y, classes, c, max_iter, tol);
    if (!opt.converged) {
      err = "logreg_fit: did not converge within max_iter (gradient inf-norm above tol)";
    }
    if (err == "") {
      model = LogisticRegressionModel { classes: classes, weights: opt.theta.weights, intercepts: opt.theta.intercepts, n_features: d.cols, n_iter: opt.iters, converged: opt.converged };
    }
  }
  if (err == "") {
    LogRegFitResult { ok: true, model: model, error: "" }
  } else {
    LogRegFitResult { ok: false, model: EmptyLogRegModel(), error: err }
  }
}

// logits from a fitted model row.
function ModelRowLogits(row: float[], model: LogisticRegressionModel) -> float[] {
  let out: float[] = [];
  for (let k = 0; k < model.classes.length(); k += 1) {
    out.push(model.intercepts[k] + DotUnchecked(model.weights[k], row));
  }
  out
}

// Predict class probabilities per row (stable softmax).
function LogRegPredictProba(x: float[][], model: LogisticRegressionModel) -> ProbaResult {
  let err = "";
  if (model.n_features <= 0 || model.classes.length() < 2 || model.weights.length() != model.classes.length() || model.intercepts.length() != model.classes.length()) {
    err = "logreg_predict_proba: invalid fitted model state";
  }
  if (err == "") {
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
  let d = CheckFiniteMatrix(x);
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

// Predict the class label per row: argmax of the probabilities over the stored
// class order (ties resolve to the lowest class).
function LogRegPredict(x: float[][], model: LogisticRegressionModel) -> IntArrayResult {
  let pr = LogRegPredictProba(x, model);
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
