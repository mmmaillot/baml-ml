// Pure scalar/matrix primitives and result types. No tracing, no policies.
// Inference-only implementation; training code is preserved at pre-inference-cleanup.


class NumArrayResult {
  ok bool
  value float[]
  error string
}


class MatrixResult {
  ok bool
  value float[][]
  error string
}


class DimsResult {
  ok bool
  rows int
  cols int
  error string
}


class IntArrayResult {
  ok bool
  value int[]
  error string
}

class InferenceFailure {
  stage string
  message string
}

// Native float comparisons use a total order; check NaN explicitly.

function IsNaN(x: float) -> bool { x.is_nan() }


function IsFinite(x: float) -> bool { x.is_finite() }


function Abs(x: float) -> float { x.abs() }


function MaxAbsoluteValue(row: float[]) -> float {
    let maximum = 0.0;
    for (let value in row) {
        let magnitude = Abs(value);
        if (magnitude > maximum) {
            maximum = magnitude;
        }
    }
    maximum
}


function Exp(x: float) -> float { x.exp() }


function CheckMatrix(m: float[][]) -> DimsResult {
  let err = "";
  let cols = 0;
  if (m.length() > 0) {
    cols = m[0].length();
  }
  if (m.length() == 0) {
    err = "matrix has no rows";
  }
  if (m.length() > 0 && cols == 0) {
    err = "matrix has zero columns";
  }
  for (let i = 1; i < m.length() && err == ""; i += 1) {
    if (m[i].length() != cols) {
      err = "matrix is ragged: rows have different lengths";
    }
  }
  if (err == "") {
    DimsResult { ok: true, rows: m.length(), cols: cols, error: "" }
  } else {
    DimsResult { ok: false, rows: 0, cols: 0, error: err }
  }
}


function CheckFiniteMatrix(m: float[][]) -> DimsResult {
  let d = CheckMatrix(m);
  let err = d.error;
  if (err == "") {
    for (let i = 0; i < m.length() && err == ""; i += 1) {
      for (let j = 0; j < d.cols && err == ""; j += 1) {
        if (!IsFinite(m[i][j])) {
          err = "matrix contains NaN or infinite values";
        }
      }
    }
  }
  if (err == "") {
    DimsResult { ok: true, rows: d.rows, cols: d.cols, error: "" }
  } else {
    DimsResult { ok: false, rows: 0, cols: 0, error: err }
  }
}

// Internal: caller validates equal vector lengths.

function DotUnchecked(a: float[], b: float[]) -> float {
  let s = 0.0;
  for (let i = 0; i < a.length(); i += 1) {
    s = s + a[i] * b[i];
  }
  s
}

