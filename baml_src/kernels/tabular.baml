// Heterogeneous Cell[][] tabular container, conversions and matrix checks.
// Schema v3: arbitrary-length sequential pipelines (0..N typed preprocessing
// steps followed by 1 supported estimator), data-only. The v1 and v2 artifact,
// loader, BatchOutput and ShadowOutput contracts are unchanged.


class Cell {
  kind string // "number" | "string" | "null"
  num float
  str string
}


class CellMatrixResult {
  ok bool
  cells Cell[][]
  error string
}


class PipelineExecution {
  // Per-row maximum absolute z-score aggregated across every standard_scaler
  // step/branch execution (Adaptive extreme_zscore); null when no scaler ran.
  z_abs_max float[]?
}

// Internal: fold one scaler output into the per-row max-abs-z aggregate so
// trigger evaluation never depends on branch execution order.

function MergeScalerAbsMax(exec: PipelineExecution, x: float[][]) -> void {
  let cur = exec.z_abs_max;
  if (cur == null) {
    let m: float[] = [];
    for (let i = 0; i < x.length(); i += 1) { m.push(MaxAbsoluteValue(x[i])); }
    exec.z_abs_max = m;
  } else {
    for (let i = 0; i < x.length() && i < cur.length(); i += 1) {
      let v = MaxAbsoluteValue(x[i]);
      if (v > cur[i]) { cur[i] = v; }
    }
  }
}


function CellOf(v: float | string | null) -> Cell {
  if (v == null) { return Cell { kind: "null", num: 0.0, str: "" }; }
  if (v is string) { return Cell { kind: "string", num: 0.0, str: v }; }
  Cell { kind: "number", num: v, str: "" }
}


function ShadowCells(rows: (float | string | null)[][]) -> Cell[][] {
  let out: Cell[][] = [];
  for (let i = 0; i < rows.length(); i += 1) {
    let row: Cell[] = [];
    for (let j = 0; j < rows[i].length(); j += 1) {
      row.push(CellOf(rows[i][j]));
    }
    out.push(row);
  }
  out
}

// Programmatic float matrices use NaN for missing values, like the v1/v2 API.

function NumericCells(x: float[][]) -> Cell[][] {
  let out: Cell[][] = [];
  for (let i = 0; i < x.length(); i += 1) {
    let row: Cell[] = [];
    for (let j = 0; j < x[i].length(); j += 1) {
      if (IsNaN(x[i][j])) {
        row.push(Cell { kind: "null", num: 0.0, str: "" });
      } else {
        row.push(Cell { kind: "number", num: x[i][j], str: "" });
      }
    }
    out.push(row);
  }
  out
}

// Internal: numeric-only consumers (v1/v2 selective path, scaler output).
// String cells are rejected; null cells become NaN.

function CellsToFloatMatrix(cells: Cell[][]) -> float[][] throws InferenceFailure {
  let out: float[][] = [];
  for (let i = 0; i < cells.length(); i += 1) {
    let row: float[] = [];
    for (let j = 0; j < cells[i].length(); j += 1) {
      let c = cells[i][j];
      if (c.kind == "string") {
        throw InferenceFailure { stage: "input_validation", message: "batch: string value for a numeric-only model" };
      }
      if (c.kind == "null") {
        row.push(float.nan());
      } else {
        row.push(c.num);
      }
    }
    out.push(row);
  }
  out
}


function CheckCellMatrixDims(cells: Cell[][], n: int) -> string {
  if (cells.length() == 0) { return "matrix has no rows"; }
  let cols = cells[0].length();
  if (cols == 0) { return "matrix has zero columns"; }
  for (let i = 1; i < cells.length(); i += 1) {
    if (cells[i].length() != cols) { return "matrix is ragged: rows have different lengths"; }
  }
  if (cols != n) { return "feature count differs from fitted step"; }
  ""
}

// Internal: validated step state. Numeric imputation reuses the checked
// float64 core (null cells are NaN, string cells are rejected); string
// imputation replaces null cells with the fitted fill string.

function PipelineEstimatorMatrix(cells: Cell[][]) -> float[][] throws InferenceFailure {
  let out: float[][] = [];
  for (let i = 0; i < cells.length(); i += 1) {
    let row: float[] = [];
    for (let j = 0; j < cells[i].length(); j += 1) {
      let c = cells[i][j];
      if (c.kind == "string") {
        throw InferenceFailure { stage: "input_validation", message: "pipeline: non-numeric feature reached the estimator" };
      }
      if (c.kind == "null") {
        throw InferenceFailure { stage: "input_validation", message: "pipeline: missing value reached the estimator without imputation" };
      }
      if (!IsFinite(c.num)) {
        throw InferenceFailure { stage: "input_validation", message: "pipeline: non-finite feature reached the estimator" };
      }
      row.push(c.num);
    }
    out.push(row);
  }
  out
}


