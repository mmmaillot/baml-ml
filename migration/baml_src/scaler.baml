// Inference-only implementation; training code is preserved at pre-inference-cleanup.

class StandardScaler {
  means float[]
  scales float[]
  n_features int
}

class ScalerTransformResult {
  ok bool
  x float[][]
  error string
}

function EmptyScaler() -> StandardScaler {
  StandardScaler { means: [], scales: [], n_features: 0 }
}

function ScalerTransform(x: float[][], scaler: StandardScaler) -> ScalerTransformResult {
  ScalerTransformCore(x, scaler, false)
}

// Internal only: skip the finite scan after successful imputation.
function ScalerTransformCore(x: float[][], scaler: StandardScaler, finite_input: bool, validated_state: bool = false) -> ScalerTransformResult {
  let err = "";
  if (!validated_state && (scaler.n_features <= 0 || scaler.means.length() != scaler.n_features || scaler.scales.length() != scaler.n_features)) {
    err = "scaler_transform: invalid fitted scaler state";
  }
  if (!validated_state && err == "") {
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
