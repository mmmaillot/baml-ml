// Mean imputation: legacy float[][] core and v3 Cell[][] step core.
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
  ImputerTransformCore(x, imputer)
}

// Internal: validated_state is reserved for factory-owned prepared models.

function ImputerTransformCore(x: float[][], imputer: MeanImputer, validated_state: bool = false) -> ImputerTransformResult {
  let err = "";
  if (!validated_state && (imputer.n_features <= 0 || imputer.means.length() != imputer.n_features)) {
    err = "imputer_transform: invalid fitted imputer state";
  }
  if (!validated_state && err == "") {
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

class SimpleImputerStep {
  strategy string // "mean" | "median" | "most_frequent" | "constant"
  value_kind string // "numeric" | "string"
  statistics float[]
  fill_strings string[]
}


function ImputerStepCore(cells: Cell[][], imp: SimpleImputerStep) -> CellMatrixResult {
  let n = if (imp.value_kind == "numeric") { imp.statistics.length() } else { imp.fill_strings.length() };
  let err = CheckCellMatrixDims(cells, n);
  let out: Cell[][] = [];
  if (err == "" && imp.value_kind == "string") {
    for (let i = 0; i < cells.length() && err == ""; i += 1) {
      let row: Cell[] = [];
      for (let j = 0; j < n; j += 1) {
        let c = cells[i][j];
        if (c.kind == "number") {
          err = "imputer_transform: numeric value in string imputer";
        } else if (c.kind == "null") {
          row.push(Cell { kind: "string", num: 0.0, str: imp.fill_strings[j] });
        } else {
          row.push(c);
        }
      }
      if (err == "") { out.push(row); }
    }
  }
  let x: float[][] = [];
  if (err == "" && imp.value_kind == "numeric") {
    for (let i = 0; i < cells.length(); i += 1) {
      let row: float[] = [];
      for (let j = 0; j < n; j += 1) {
        let c = cells[i][j];
        if (c.kind == "string") {
          err = "imputer_transform: string value in numeric imputer";
        } else if (c.kind == "null") {
          row.push(float.nan());
        } else {
          row.push(c.num);
        }
      }
      if (err != "") { break; }
      x.push(row);
    }
  }
  let r = ImputerTransformCore(x, MeanImputer { means: imp.statistics, n_features: n }, validated_state = true);
  if (err == "" && imp.value_kind == "numeric") { err = r.error; }
  if (err == "" && imp.value_kind == "numeric") {
    out = [];
    for (let i = 0; i < r.x.length(); i += 1) {
      let row: Cell[] = [];
      for (let j = 0; j < n; j += 1) {
        row.push(Cell { kind: "number", num: r.x[i][j], str: "" });
      }
      out.push(row);
    }
  }
  if (err == "") {
    CellMatrixResult { ok: true, cells: out, error: "" }
  } else {
    CellMatrixResult { ok: false, cells: [], error: err }
  }
}

// Internal: validated step state. Scaler output is finite by construction.

