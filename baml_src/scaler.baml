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
    variances = ColumnVariancesUnchecked(x, d.cols);
  }
  let var_bad = false;
  for (let j = 0; j < variances.length() && !var_bad; j += 1) {
    if (!IsFinite(variances[j])) {
      var_bad = true;
    }
  }
  if (var_bad) {
    err = "scaler_fit: computed non-finite variance (input magnitude too large)";
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
// have the same feature count as the fitted scaler.
function ScalerTransform(x: float[][], scaler: StandardScaler) -> ScalerTransformResult {
  let err = "";
  if (scaler.n_features <= 0 || scaler.means.length() != scaler.n_features || scaler.scales.length() != scaler.n_features) {
    err = "scaler_transform: invalid fitted scaler state";
  }
  let d = CheckFiniteMatrix(x);
  if (err == "" && d.error != "") {
    err = d.error;
  }
  if (err == "" && d.cols != scaler.n_features) {
    err = "scaler_transform: feature count differs from fitted scaler";
  }
  let out: float[][] = [];
  if (err == "") {
    for (let i = 0; i < x.length(); i += 1) {
      let row: float[] = [];
      for (let j = 0; j < d.cols; j += 1) {
        row.push((x[i][j] - scaler.means[j]) / scaler.scales[j]);
      }
      out.push(row);
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
  if (err == "") {
    sc = f.scaler;
  }
  let t = ScalerTransform(x, sc);
  if (err == "") {
    err = t.error;
  }
  let out: float[][] = [];
  if (err == "") {
    out = t.x;
  }
  if (err == "") {
    ScalerFitTransformResult { ok: true, scaler: sc, x: out, error: "" }
  } else {
    ScalerFitTransformResult { ok: false, scaler: EmptyScaler(), x: [], error: err }
  }
}
