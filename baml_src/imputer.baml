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

// Fit per-column means from non-missing (non-NaN) training values only.
function ImputerFit(x: float[][]) -> ImputerFitResult {
  let d = CheckMatrix(x);
  let err = d.error;
  let means: float[] = [];
  let all_missing = false;
  if (err == "") {
    for (let j = 0; j < d.cols; j += 1) {
      let s = 0.0;
      let n = 0.0;
      for (let i = 0; i < x.length(); i += 1) {
        let v = x[i][j];
        if (v == v) {
          s = s + v;
          n = n + 1.0;
        }
      }
      if (n == 0.0) {
        all_missing = true;
      } else {
        means.push(s / n);
      }
    }
  }
  if (all_missing) {
    err = "imputer_fit: a training column is entirely missing";
  }
  if (err == "") {
    ImputerFitResult { ok: true, imputer: MeanImputer { means: means, n_features: d.cols }, error: "" }
  } else {
    ImputerFitResult { ok: false, imputer: EmptyImputer(), error: err }
  }
}

// Replace NaN cells with the stored per-column means. Shape and non-missing
// entries are preserved; the feature count must match the fitted imputer.
function ImputerTransform(x: float[][], imputer: MeanImputer) -> ImputerTransformResult {
  let d = CheckMatrix(x);
  let err = d.error;
  let out: float[][] = [];
  if (err == "" && d.cols != imputer.n_features) {
    err = "imputer_transform: feature count differs from fitted imputer";
  }
  if (err == "") {
    for (let i = 0; i < x.length(); i += 1) {
      let row: float[] = [];
      for (let j = 0; j < d.cols; j += 1) {
        let v = x[i][j];
        if (v == v) {
          row.push(v);
        } else {
          row.push(imputer.means[j]);
        }
      }
      out.push(row);
    }
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
