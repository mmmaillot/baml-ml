// BAML language 0.25.0 port of the v0.1.0 numerical primitives.
// Keep float[] / float[][] and the legacy result classes.
// NaN comparisons use a total order in the new language: always use
// IsNaN/IsFinite rather than x != x or comparisons with NaN.
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

class IntArrayResult {
  ok bool
  value int[]
  error string
}

// Native float operations; retain the historical wrappers and contracts.
function IsNaN(x: float) -> bool { x.is_nan() }

function IsFinite(x: float) -> bool { x.is_finite() }

function Abs(x: float) -> float { x.abs() }

// Native exp may retain subnormal results that the old reciprocal-based
// approximation rounded to zero. This is an intentional accuracy improvement.
function Exp(x: float) -> float { x.exp() }

// Preserve the old finite negative sentinel for nonpositive inputs.
function Log(x: float) -> float {
  if (x.is_nan()) { x } else {
    if (x <= 0.0) { -8.98846567431158e307 } else { x.ln() }
  }
}

// Preserve NaN propagation and the historical clamp for x <= 0.
function Sqrt(x: float) -> float {
  if (x.is_nan()) { x } else {
    if (x <= 0.0) { 0.0 } else { x.sqrt() }
  }
}

// Shape check for a row-major dense matrix: non-empty, at least one column,
// and every row of equal length. Cell values (including NaN) are not checked.
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

// Shape and finiteness check for estimators requiring complete finite data
// (no NaN and no ±Inf).
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

// Internal: dot product assuming equal lengths (caller validates).
function DotUnchecked(a: float[], b: float[]) -> float {
  let s = 0.0;
  for (let i = 0; i < a.length(); i += 1) {
    s = s + a[i] * b[i];
  }
  s
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
    s = DotUnchecked(a, b);
  }
  if (err == "") {
    NumResult { ok: true, value: s, error: "" }
  } else {
    NumResult { ok: false, value: 0.0, error: err }
  }
}

// Euclidean norm of a vector scaled by max_i |v[i]| (dnrm2-style) so squares
// do not overflow for large finite entries (~1e200) or underflow for tiny ones.
function VecNorm(v: float[]) -> float {
  let max_abs = 0.0;
  let has_nan = false;
  let has_inf = false;
  for (let i = 0; i < v.length(); i += 1) {
    let vi = v[i];
    if (IsNaN(vi)) {
      has_nan = true;
    } else {
      if (!IsFinite(vi)) {
        has_inf = true;
      } else {
        let a = Abs(vi);
        if (a > max_abs) {
          max_abs = a;
        }
      }
    }
  }
  let res = 0.0;
  if (has_nan || has_inf) {
    // DotUnchecked(v, v) produces NaN if any entry is NaN, else +Inf.
    res = Sqrt(DotUnchecked(v, v));
  } else {
    if (max_abs > 0.0) {
      let acc = 0.0;
      for (let i = 0; i < v.length(); i += 1) {
        let u = v[i] / max_abs;
        acc = acc + u * u;
      }
      res = max_abs * Sqrt(acc);
    }
  }
  res
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

// Internal: per-column population variances (ddof=0) of a validated matrix
// from precomputed column means, using Chan, Golub & LeVeque's (1983)
// corrected two-pass formula:
//   var = max(0, (sum((x_i - mean)^2) - (sum(x_i - mean))^2 / n) / n)
// Subtracting the squared residual sum cancels first-order float64 mean
// rounding (giving exact 0.0 on constant columns like [0.1, 0.1, 0.1]) while
// preserving two-pass accuracy on large-offset columns (e.g. 1e12 + N(0, 1)),
// where textbook single-pass Welford loses ~10 digits of precision.
function ColumnVariancesFromMeansUnchecked(m: float[][], means: float[], cols: int) -> float[] {
  let n = 0.0;
  for (let i = 0; i < m.length(); i += 1) {
    n = n + 1.0;
  }
  let variances: float[] = [];
  for (let j = 0; j < cols; j += 1) {
    let acc_sq = 0.0;
    let acc_diff = 0.0;
    for (let i = 0; i < m.length(); i += 1) {
      let diff = m[i][j] - means[j];
      acc_diff = acc_diff + diff;
      acc_sq = acc_sq + diff * diff;
    }
    let corrected = acc_sq;
    if (IsFinite(acc_sq) && IsFinite(acc_diff)) {
      corrected = acc_sq - (acc_diff / n) * acc_diff;
      if (corrected < 0.0) {
        corrected = 0.0;
      }
    }
    variances.push(corrected / n);
  }
  variances
}

// Internal: per-column population variances (ddof=0) of a validated matrix,
// corrected two-pass for numerical stability.
function ColumnVariancesUnchecked(m: float[][], cols: int) -> float[] {
  let means = ColumnMeansUnchecked(m, cols);
  ColumnVariancesFromMeansUnchecked(m, means, cols)
}

// Per-column population variances (ddof=0), corrected two-pass for numerical stability.
function ColumnVariances(m: float[][]) -> NumArrayResult {
  let d = CheckMatrix(m);
  let err = d.error;
  let variances: float[] = [];
  if (err == "") {
    variances = ColumnVariancesUnchecked(m, d.cols);
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
