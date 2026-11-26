// Shared dense numeric primitives for baml-ml.
//
// Representation conventions (see README.md):
// - Vector: float[]
// - Matrix: float[][] in row-major order (a list of rows); ragged or empty
//   matrices are rejected by CheckMatrix.
// - Missing values are NaN (detected via x != x), not BAML null.
// - int and float never mix implicitly: counters that are divided are kept
//   as float (e.g. n = n + 1.0).
//
// Error convention: every public function validates its shapes and returns a
// result class with a non-nullable `ok` flag and `error` string ("" when ok),
// because BAML 0.226.3 expression functions cannot throw and cannot narrow
// unions. Python callers check `.ok` / `.error`; BAML callers check `ok`.
//
// Style note: the 0.226.3 beta compiler miscompiles if/else expressions
// nested inside branches that also contain statements, so every function
// below uses flat statement-level validation followed by a single tail
// if/else that builds the result.

class NumResult {
  ok bool
  value float
  error string
}

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

// Shape check for a row-major dense matrix: non-empty, at least one column,
// and every row of equal length.
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

// Dot product of two vectors. Vectors must have equal length (two empty
// vectors give 0.0).
function Dot(a: float[], b: float[]) -> NumResult {
  let err = "";
  let s = 0.0;
  if (a.length() != b.length()) {
    err = "dot: vector lengths differ";
  }
  if (err == "") {
    for (let i = 0; i < a.length(); i += 1) {
      s = s + a[i] * b[i];
    }
  }
  if (err == "") {
    NumResult { ok: true, value: s, error: "" }
  } else {
    NumResult { ok: false, value: 0.0, error: err }
  }
}

// Internal: per-column means of a matrix already validated by CheckMatrix.
function ColumnMeansUnchecked(m: float[][], cols: int) -> float[] {
  let means: float[] = [];
  for (let j = 0; j < cols; j += 1) {
    let s = 0.0;
    let n = 0.0;
    for (let i = 0; i < m.length(); i += 1) {
      s = s + m[i][j];
      n = n + 1.0;
    }
    means.push(s / n);
  }
  means
}

// Per-column means. Assumes complete data (no NaN); NaN would propagate.
function ColumnMeans(m: float[][]) -> NumArrayResult {
  let d = CheckMatrix(m);
  let err = d.error;
  let means: float[] = [];
  if (err == "") {
    means = ColumnMeansUnchecked(m, d.cols);
  }
  if (err == "") {
    NumArrayResult { ok: true, value: means, error: "" }
  } else {
    NumArrayResult { ok: false, value: [], error: err }
  }
}

// Per-column population variances (ddof=0), two-pass for numerical stability.
function ColumnVariances(m: float[][]) -> NumArrayResult {
  let d = CheckMatrix(m);
  let err = d.error;
  let variances: float[] = [];
  if (err == "") {
    let means = ColumnMeansUnchecked(m, d.cols);
    let n = 0.0;
    for (let i = 0; i < m.length(); i += 1) {
      n = n + 1.0;
    }
    for (let j = 0; j < d.cols; j += 1) {
      let acc = 0.0;
      for (let i = 0; i < m.length(); i += 1) {
        let diff = m[i][j] - means[j];
        acc = acc + diff * diff;
      }
      variances.push(acc / n);
    }
  }
  if (err == "") {
    NumArrayResult { ok: true, value: variances, error: "" }
  } else {
    NumArrayResult { ok: false, value: [], error: err }
  }
}

function VecAdd(a: float[], b: float[]) -> NumArrayResult {
  let err = "";
  let out: float[] = [];
  if (a.length() != b.length()) {
    err = "vec_add: vector lengths differ";
  }
  if (err == "") {
    for (let i = 0; i < a.length(); i += 1) {
      out.push(a[i] + b[i]);
    }
  }
  if (err == "") {
    NumArrayResult { ok: true, value: out, error: "" }
  } else {
    NumArrayResult { ok: false, value: [], error: err }
  }
}

function VecSub(a: float[], b: float[]) -> NumArrayResult {
  let err = "";
  let out: float[] = [];
  if (a.length() != b.length()) {
    err = "vec_sub: vector lengths differ";
  }
  if (err == "") {
    for (let i = 0; i < a.length(); i += 1) {
      out.push(a[i] - b[i]);
    }
  }
  if (err == "") {
    NumArrayResult { ok: true, value: out, error: "" }
  } else {
    NumArrayResult { ok: false, value: [], error: err }
  }
}

function VecScale(a: float[], s: float) -> NumArrayResult {
  let out: float[] = [];
  for (let i = 0; i < a.length(); i += 1) {
    out.push(a[i] * s);
  }
  NumArrayResult { ok: true, value: out, error: "" }
}

// Matrix-vector product m @ v: requires cols(m) == length(v).
function MatVec(m: float[][], v: float[]) -> NumArrayResult {
  let d = CheckMatrix(m);
  let err = d.error;
  let out: float[] = [];
  if (err == "" && d.cols != v.length()) {
    err = "matvec: matrix column count differs from vector length";
  }
  if (err == "") {
    for (let i = 0; i < m.length(); i += 1) {
      let s = 0.0;
      for (let j = 0; j < d.cols; j += 1) {
        s = s + m[i][j] * v[j];
      }
      out.push(s);
    }
  }
  if (err == "") {
    NumArrayResult { ok: true, value: out, error: "" }
  } else {
    NumArrayResult { ok: false, value: [], error: err }
  }
}

function Transpose(m: float[][]) -> MatrixResult {
  let d = CheckMatrix(m);
  let err = d.error;
  let out: float[][] = [];
  if (err == "") {
    for (let j = 0; j < d.cols; j += 1) {
      let row: float[] = [];
      for (let i = 0; i < m.length(); i += 1) {
        row.push(m[i][j]);
      }
      out.push(row);
    }
  }
  if (err == "") {
    MatrixResult { ok: true, value: out, error: "" }
  } else {
    MatrixResult { ok: false, value: [], error: err }
  }
}
