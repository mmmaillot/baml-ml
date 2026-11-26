// LinearRegression: dense single-target ordinary least squares with an
// intercept. Method: QR decomposition of the design matrix A = [1 | X] by
// modified Gram-Schmidt, then back substitution on R beta = Q^T y.
// Rank tolerance: column j is deficient when its post-orthogonalization norm
// is <= 1e-10 (= 0.0000000001, beta has no exponent literals) times its
// original norm; rank-deficient input is rejected explicitly instead of
// returning an invalid solution.

class LinearRegressionModel {
  coefficients float[]
  intercept float
  n_features int
}

class LinRegFitResult {
  ok bool
  model LinearRegressionModel
  error string
}

function EmptyLinRegModel() -> LinearRegressionModel {
  LinearRegressionModel { coefficients: [], intercept: 0.0, n_features: 0 }
}

function LinearRegressionFit(x: float[][], y: float[]) -> LinRegFitResult {
  let d = CheckFiniteMatrix(x);
  let err = d.error;
  let y_bad = false;
  if (y.length() != x.length()) {
    y_bad = true;
  }
  for (let i = 0; i < y.length() && !y_bad; i += 1) {
    if (!IsFinite(y[i])) {
      y_bad = true;
    }
  }
  if (y_bad) {
    err = "linreg_fit: y must be finite with one value per row of x";
  }
  let model = EmptyLinRegModel();
  if (err == "") {
    let n = x.length();
    let m = d.cols + 1;
    // A = [1 | X], n rows x m columns, row-major.
    let a: float[][] = [];
    for (let i = 0; i < n; i += 1) {
      let row: float[] = [1.0];
      for (let j = 0; j < d.cols; j += 1) {
        row.push(x[i][j]);
      }
      a.push(row);
    }
    // Modified Gram-Schmidt: q holds orthonormal columns, rcols[j][i] = R[i][j].
    let q: float[][] = [];
    let rcols: float[][] = [];
    let deficient = false;
    for (let j = 0; j < m && !deficient; j += 1) {
      let v: float[] = [];
      for (let i = 0; i < n; i += 1) {
        v.push(a[i][j]);
      }
      let orig = VecNorm(v);
      let rcol: float[] = [];
      for (let k = 0; k < m; k += 1) {
        rcol.push(0.0);
      }
      let qn = q.length();
      for (let i = 0; i < qn; i += 1) {
        let rij = DotUnchecked(q[i], v);
        rcol[i] = rij;
        for (let t = 0; t < n; t += 1) {
          v[t] = v[t] - rij * q[i][t];
        }
      }
      let rjj = VecNorm(v);
      rcol[j] = rjj;
      if (rjj <= 0.0000000001 * orig) {
        deficient = true;
      }
      if (rjj > 0.0000000001 * orig) {
        let qj: float[] = [];
        for (let t = 0; t < n; t += 1) {
          qj.push(v[t] / rjj);
        }
        q.push(qj);
      }
      rcols.push(rcol);
    }
    if (deficient) {
      err = "linreg_fit: rank-deficient design matrix";
    }
    if (err == "") {
      // b = Q^T y, then back substitution R beta = b (R stored by columns).
      let b: float[] = [];
      for (let j = 0; j < m; j += 1) {
        b.push(DotUnchecked(q[j], y));
      }
      let beta: float[] = [];
      for (let j = 0; j < m; j += 1) {
        beta.push(0.0);
      }
      for (let j = m - 1; j >= 0; j -= 1) {
        let s = b[j];
        for (let k = j + 1; k < m; k += 1) {
          s = s - rcols[k][j] * beta[k];
        }
        beta[j] = s / rcols[j][j];
      }
      let beta_bad = false;
      for (let j = 0; j < m && !beta_bad; j += 1) {
        if (!IsFinite(beta[j])) {
          beta_bad = true;
        }
      }
      if (beta_bad) {
        err = "linreg_fit: non-finite least-squares solution";
      }
      if (err == "") {
        let coefs: float[] = [];
        for (let j = 1; j < m; j += 1) {
          coefs.push(beta[j]);
        }
        model = LinearRegressionModel { coefficients: coefs, intercept: beta[0], n_features: d.cols };
      }
    }
  }
  if (err == "") {
    LinRegFitResult { ok: true, model: model, error: "" }
  } else {
    LinRegFitResult { ok: false, model: EmptyLinRegModel(), error: err }
  }
}

// Predict one value per row: intercept + coefficients . row.
function LinearRegressionPredict(x: float[][], model: LinearRegressionModel) -> NumArrayResult {
  let err = "";
  if (model.n_features <= 0 || model.coefficients.length() != model.n_features) {
    err = "linreg_predict: invalid fitted model state";
  }
  let d = CheckFiniteMatrix(x);
  if (err == "" && d.error != "") {
    err = d.error;
  }
  if (err == "" && d.cols != model.n_features) {
    err = "linreg_predict: feature count differs from fitted model";
  }
  let out: float[] = [];
  if (err == "") {
    for (let i = 0; i < x.length(); i += 1) {
      out.push(model.intercept + DotUnchecked(model.coefficients, x[i]));
    }
  }
  if (err == "") {
    NumArrayResult { ok: true, value: out, error: "" }
  } else {
    NumArrayResult { ok: false, value: [], error: err }
  }
}
