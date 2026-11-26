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
// Error convention: every public entry point validates its inputs and returns
// a `*Result` class with a non-nullable `ok` flag and `error` string ("" when
// ok), because BAML 0.226.3 expression functions cannot throw and cannot
// narrow unions. Python callers check `.ok` / `.error`; BAML callers check
// `ok`. BAML has no `private` visibility modifier, so internal BAML-to-BAML
// helpers (`IsNaN`, `IsFinite`, `*Unchecked`, `Empty*`) also appear on the
// generated client `b` but return raw values and are not public entry points.
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

class IntArrayResult {
  ok bool
  value int[]
  error string
}

// IEEE-754 NaN test: true iff x is NaN.
function IsNaN(x: float) -> bool {
  x != x
}

// IEEE-754 finiteness test without a math library: x - x == 0.0 holds for
// every finite float (including ±0.0, subnormals, and ±DBL_MAX) and evaluates
// to NaN == 0.0 (false) for NaN, +Inf, and -Inf.
function IsFinite(x: float) -> bool {
  (x - x) == 0.0
}

// Absolute value without a math library (maps -0.0 to 0.0 and -Inf to +Inf;
// preserves NaN).
function Abs(x: float) -> float {
  let y = x;
  if (x <= 0.0) {
    y = 0.0 - x;
  }
  y
}

// Square root via Newton's method (BAML 0.226.3 beta exposes no float math
// methods). Deterministic: fixed 600 iterations from y0 = max(x, 1.0), which
// reaches machine precision even for subnormal and near-DBL_MAX finite inputs.
// Preserves NaN and +Inf; clamps finite/infinite x <= 0 to 0.0.
// Natural exponential, float-only (no math library): halve the argument until
// |r| <= 0.5, evaluate a 30-term Taylor series there, then square back
// (exp(x) = exp(x/2)^2). Halving+squaring keeps the relative error at
// ~2^depth * eps (<= ~5e-13 for |x| <= 709) with no accumulated reduction
// error. exp(+Inf) = +Inf, exp(-Inf) = 0.0, NaN propagates; exp(-x) = 1/exp(x),
// so very negative x underflows to 0.0 (values below ~5e-324 are lost).
// Deterministic.
function Exp(x: float) -> float {
  let res = 0.0;
  if (IsNaN(x)) {
    res = x;
  }
  if (!IsFinite(x) && x > 0.0) {
    res = x;
  }
  if (!IsFinite(x) && x < 0.0) {
    res = 0.0;
  }
  if (IsFinite(x)) {
    let neg = false;
    let r = x;
    if (r < 0.0) {
      neg = true;
      r = 0.0 - r;
    }
    let depth = 0.0;
    while (r > 0.5) {
      r = r * 0.5;
      depth = depth + 1.0;
    }
    let term = 1.0;
    let sum = 1.0;
    let kf = 0.0;
    for (let k = 0; k < 30; k += 1) {
      kf = kf + 1.0;
      term = term * r / kf;
      sum = sum + term;
    }
    res = sum;
    let i = 0.0;
    while (i < depth) {
      res = res * res;
      i = i + 1.0;
    }
    if (neg) {
      res = 1.0 / res;
    }
  }
  res
}

// Natural logarithm for x > 0 via scaling into [1, 2) by powers of two plus a
// 40-term atanh series: ln(m) = 2 * sum_{k} y^(2k+1)/(2k+1), y = (m-1)/(m+1).
// log(+Inf) = +Inf; for x <= 0 (finite) returns -1e308 as a finite "-Inf"
// sentinel (callers in this repo only evaluate log of sums >= 1 via LogSumExp).
function Log(x: float) -> float {
  let res = 0.0;
  if (IsNaN(x)) {
    res = x;
  }
  if (!IsFinite(x) && x > 0.0) {
    res = x;
  }
  if (IsFinite(x) && x <= 0.0) {
    let big = 1.0;
    for (let i = 0; i < 1023; i += 1) {
      big = big * 2.0;
    }
    res = 0.0 - big;
  }
  if (IsFinite(x) && x > 0.0) {
    let m = x;
    let e = 0.0;
    while (m >= 2.0) {
      m = m * 0.5;
      e = e + 1.0;
    }
    while (m < 1.0) {
      m = m * 2.0;
      e = e - 1.0;
    }
    let y = (m - 1.0) / (m + 1.0);
    let y2 = y * y;
    let term = y;
    let sum = y;
    let k = 1.0;
    for (let i = 0; i < 40; i += 1) {
      k = k + 2.0;
      term = term * y2;
      sum = sum + term / k;
    }
    res = e * 0.6931471805599453 + 2.0 * sum;
  }
  res
}

function Sqrt(x: float) -> float {
  let y = 0.0;
  if (IsNaN(x)) {
    y = x;
  } else {
    if (x > 0.0 && !IsFinite(x)) {
      y = x;
    }
    if (x > 0.0 && IsFinite(x)) {
      let y0 = 1.0;
      if (x > 1.0) {
        y0 = x;
      }
      y = y0;
      for (let i = 0; i < 600; i += 1) {
        y = 0.5 * (y + x / y);
      }
    }
  }
  y
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
