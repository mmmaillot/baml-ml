// StandardScaler: dense standardization (center per feature, divide by unit
// population variance, ddof=0). Input must be finite and already imputed
// (no NaN, no ±Inf). Variances come from Chan, Golub & LeVeque's (1983)
// corrected two-pass formula, and `scale = Sqrt(variance)` is computed in BAML
// via native float.sqrt() through the compatibility wrapper.
//
// Near-constant threshold (issue #14): following Chan, Golub & LeVeque's
// two-pass rounding error bound (also used by sklearn's `_is_constant_feature`),
// a column is indistinguishable from constant when
//   var <= n * eps * var + (n * eps * mean)^2,
// i.e. in overflow-safe scale space: `s * Sqrt(1 - n * eps) <= n * eps * |mean|`
// where `eps = 2^-52` (`0.0000000000000002220446049250313`) and `n` is the row count. Such columns get `scale = 1.0`,
// while genuinely representable small scales are preserved.

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
    let n = 0.0;
    for (let i = 0; i < x.length(); i += 1) {
      n = n + 1.0;
    }
    let n_eps = n * 0.0000000000000002220446049250313;
    let guard = Sqrt(1.0 - n_eps);
    for (let j = 0; j < d.cols; j += 1) {
      let s = 1.0;
      if (variances[j] > 0.0) {
        let cand = Sqrt(variances[j]);
        if (cand * guard > n_eps * Abs(means[j])) {
          s = cand;
        }
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
  ScalerTransformCore(x, scaler, false)
}

// Internal: finite_input is true only for the successful output of the preceding
// preprocessing stage. Shape, fitted state, and computed outputs remain checked.
function ScalerTransformCore(x: float[][], scaler: StandardScaler, finite_input: bool) -> ScalerTransformResult {
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
    let d = if (finite_input) { CheckMatrix(x) } else { CheckFiniteMatrix(x) };
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
