// StandardScaler: dense standardization (center per feature, divide by unit
// population variance, ddof=0). Input must be finite and already imputed
// (no NaN, no ±Inf). scale = sqrt(population variance) computed in BAML via
// Newton's method (Sqrt); zero-variance columns use scale = 1.

class StandardScaler {
  means float[]
  scales float[]
  n_features int
}

class ScalerFitResult {
  ok bool
  scaler StandardScaler
  error string
}

class ScalerTransformResult {
  ok bool
  x float[][]
  error string
}

class ScalerFitTransformResult {
  ok bool
  scaler StandardScaler
  x float[][]
  error string
}

function EmptyScaler() -> StandardScaler {
  StandardScaler { means: [], scales: [], n_features: 0 }
}

function ScalerFit(x: float[][]) -> ScalerFitResult {
  let d = CheckFiniteMatrix(x);
  let err = d.error;
  let means: float[] = [];
  let variances: float[] = [];
  if (err == "") {
    means = ColumnMeansUnchecked(x, d.cols);
    for (let j = 0; j < means.length() && err == ""; j += 1) {
      if (!IsFinite(means[j])) {
        err = "scaler_fit: computed column mean is not finite";
      }
    }
  }
  if (err == "") {
    variances = ColumnVariancesFromMeansUnchecked(x, means, d.cols);
    for (let j = 0; j < variances.length() && err == ""; j += 1) {
      if (!IsFinite(variances[j])) {
        err = "scaler_fit: computed non-finite variance (input magnitude too large)";
      }
    }
  }
  let scales: float[] = [];
  if (err == "") {
    for (let j = 0; j < d.cols; j += 1) {
      let s = Sqrt(variances[j]);
      if (s == 0.0) {
        s = 1.0;
      }
      scales.push(s);
    }
  }
  if (err == "") {
    ScalerFitResult { ok: true, scaler: StandardScaler { means: means, scales: scales, n_features: d.cols }, error: "" }
  } else {
    ScalerFitResult { ok: false, scaler: EmptyScaler(), error: err }
  }
}

// Standardize using the stored training statistics. Input must be finite and
// have the same feature count as the fitted scaler; malformed scaler states
// (including non-positive or non-finite scales) and non-finite outputs are
// rejected without dividing by zero.
function ScalerTransform(x: float[][], scaler: StandardScaler) -> ScalerTransformResult {
  let err = "";
  if (scaler.n_features <= 0 || scaler.means.length() != scaler.n_features || scaler.scales.length() != scaler.n_features) {
    err = "scaler_transform: invalid fitted scaler state";
  }
  if (err == "") {
    for (let j = 0; j < scaler.n_features && err == ""; j += 1) {
      if (!IsFinite(scaler.means[j])) {
        err = "scaler_transform: fitted scaler contains non-finite means";
      }
      if (err == "" && (!IsFinite(scaler.scales[j]) || scaler.scales[j] <= 0.0)) {
        err = "scaler_transform: fitted scaler contains invalid scales";
      }
    }
  }
  let out: float[][] = [];
  if (err == "") {
    let d = CheckFiniteMatrix(x);
    err = d.error;
    if (err == "" && d.cols != scaler.n_features) {
      err = "scaler_transform: feature count differs from fitted scaler";
    }
    if (err == "") {
      let non_finite_out = false;
      for (let i = 0; i < x.length(); i += 1) {
        let row: float[] = [];
        for (let j = 0; j < d.cols; j += 1) {
          let z = (x[i][j] - scaler.means[j]) / scaler.scales[j];
          if (!IsFinite(z)) {
            non_finite_out = true;
          }
          row.push(z);
        }
        out.push(row);
      }
      if (non_finite_out) {
        err = "scaler_transform: standardized value is not finite";
      }
    }
  }
  if (err == "") {
    ScalerTransformResult { ok: true, x: out, error: "" }
  } else {
    ScalerTransformResult { ok: false, x: [], error: err }
  }
}

function ScalerFitTransform(x: float[][]) -> ScalerFitTransformResult {
  let f = ScalerFit(x);
  let err = f.error;
  let sc = EmptyScaler();
  let out: float[][] = [];
  if (err == "") {
    sc = f.scaler;
    let t = ScalerTransform(x, sc);
    err = t.error;
    if (err == "") {
      out = t.x;
    }
  }
  if (err == "") {
    ScalerFitTransformResult { ok: true, scaler: sc, x: out, error: "" }
  } else {
    ScalerFitTransformResult { ok: false, scaler: EmptyScaler(), x: [], error: err }
  }
}
