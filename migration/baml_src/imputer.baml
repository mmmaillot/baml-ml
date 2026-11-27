// Inference-only implementation; training code is preserved at pre-inference-cleanup.

class MeanImputer {
  means float[]
  n_features int
}

class ImputerTransformResult {
  ok bool
  x float[][]
  error string
}

function EmptyImputer() -> MeanImputer {
  MeanImputer { means: [], n_features: 0 }
}

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
