// Mean SimpleImputer over dense row-major matrices with NaN as missing value.
//
// Fitted state is an explicit MeanImputer value holding the per-column means
// computed only from non-missing training values, plus the feature count.
// An entirely missing training column is rejected with an error (no silent
// feature dropping). Transform preserves shape and all non-missing entries
// and always uses the stored training statistics.

class MeanImputer {
  means float[]
  n_features int
}

class ImputerFitResult {
  ok bool
  imputer MeanImputer
  error string
}

class ImputerTransformResult {
  ok bool
  x float[][]
  error string
}

class ImputerFitTransformResult {
  ok bool
  imputer MeanImputer
  x float[][]
  error string
}

function EmptyImputer() -> MeanImputer {
  MeanImputer { means: [], n_features: 0 }
}

// Fit per-column means from non-missing (non-NaN) finite training values only.
// Infinite entries and all-missing columns are rejected explicitly.
function ImputerFit(x: float[][]) -> ImputerFitResult {
  let d = CheckMatrix(x);
  let err = d.error;
  let means: float[] = [];
  let has_inf = false;
  let all_missing = false;
  let non_finite_mean = false;
  if (err == "") {
    for (let j = 0; j < d.cols; j += 1) {
      let s = 0.0;
      let n = 0.0;
      for (let i = 0; i < x.length(); i += 1) {
        let v = x[i][j];
        if (!IsNaN(v)) {
          if (!IsFinite(v)) {
            has_inf = true;
          } else {
            s = s + v;
            n = n + 1.0;
          }
        }
      }
      if (n == 0.0) {
        all_missing = true;
      } else {
        let mean = s / n;
        if (!IsFinite(mean)) {
          non_finite_mean = true;
        }
        means.push(mean);
      }
    }
  }
  if (err == "" && has_inf) {
    err = "imputer_fit: input contains infinite values";
  }
  if (err == "" && all_missing) {
    err = "imputer_fit: a training column is entirely missing";
  }
  if (err == "" && non_finite_mean) {
    err = "imputer_fit: computed column mean is not finite";
  }
  if (err == "") {
    ImputerFitResult { ok: true, imputer: MeanImputer { means: means, n_features: d.cols }, error: "" }
  } else {
    ImputerFitResult { ok: false, imputer: EmptyImputer(), error: err }
  }
}

// Replace NaN cells with the stored per-column means. Shape and finite
// non-missing entries are preserved; infinite entries, malformed imputer
// states, and mismatched feature counts are rejected.
function ImputerTransform(x: float[][], imputer: MeanImputer) -> ImputerTransformResult {
  let err = "";
  if (imputer.n_features <= 0 || imputer.means.length() != imputer.n_features) {
    err = "imputer_transform: invalid fitted imputer state";
  }
  if (err == "") {
    for (let j = 0; j < imputer.means.length() && err == ""; j += 1) {
      if (!IsFinite(imputer.means[j])) {
        err = "imputer_transform: fitted imputer contains non-finite means";
      }
    }
  }
  let d = CheckMatrix(x);
  if (err == "" && d.error != "") {
    err = d.error;
  }
  if (err == "" && d.cols != imputer.n_features) {
    err = "imputer_transform: feature count differs from fitted imputer";
  }
  let out: float[][] = [];
  let has_inf = false;
  if (err == "") {
    for (let i = 0; i < x.length(); i += 1) {
      let row: float[] = [];
      for (let j = 0; j < d.cols; j += 1) {
        let v = x[i][j];
        if (IsNaN(v)) {
          row.push(imputer.means[j]);
        } else {
          if (!IsFinite(v)) {
            has_inf = true;
          }
          row.push(v);
        }
      }
      out.push(row);
    }
  }
  if (err == "" && has_inf) {
    err = "imputer_transform: input contains infinite values";
  }
  if (err == "") {
    ImputerTransformResult { ok: true, x: out, error: "" }
  } else {
    ImputerTransformResult { ok: false, x: [], error: err }
  }
}

function ImputerFitTransform(x: float[][]) -> ImputerFitTransformResult {
  let fit = ImputerFit(x);
  let err = fit.error;
  let imp = EmptyImputer();
  let out: float[][] = [];
  if (err == "") {
    imp = fit.imputer;
    let t = ImputerTransform(x, imp);
    err = t.error;
    if (err == "") {
      out = t.x;
    }
  }
  if (err == "") {
    ImputerFitTransformResult { ok: true, imputer: imp, x: out, error: "" }
  } else {
    ImputerFitTransformResult { ok: false, imputer: EmptyImputer(), x: [], error: err }
  }
}
