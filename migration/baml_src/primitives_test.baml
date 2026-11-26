// Port of tests/test_primitives.py at v0.1.0, plus native exp/log contracts.
// Reference values come from the historical tests and standard f64 math.
function ApproxRelative(actual: float, expected: float, rel: float) -> bool {
  if (expected == 0.0) { actual == 0.0 } else {
    (actual - expected).abs() <= expected.abs() * rel
  }
}

test "dot_known_values" {
  assert.equal(Dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]).value, 32.0)
  assert.equal(Dot([-1.0, 2.0], [3.0, -4.0]).value, -11.0)
  assert.equal(Dot([0.5], [2.0]).value, 1.0)
}
test "dot_empty" {
  let r = Dot([], []);
  assert.is_true(r.ok)
  assert.equal(r.value, 0.0)
}
test "dot_mismatch" {
  let r = Dot([1.0], [1.0, 2.0]);
  assert.is_true(!r.ok)
  assert.equal(r.error, "dot: vector lengths differ")
}
test "matrix_valid" {
  let d = CheckMatrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
  assert.is_true(d.ok)
  assert.equal(d.rows, 2)
  assert.equal(d.cols, 3)
  assert.equal(d.error, "")
}
test "matrix_invalid" {
  assert.is_true(!CheckMatrix([]).ok)
  assert.is_true(!CheckMatrix([[]]).ok)
  let d = CheckMatrix([[1.0, 2.0], [3.0]]);
  assert.is_true(!d.ok)
  assert.equal(d.error, "matrix is ragged: rows have different lengths")
}
test "column_means" {
  let r = ColumnMeans([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]]);
  assert.is_true(r.ok)
  assert.equal(r.value, [3.0, 20.0])
}
test "column_population_variance" {
  let r = ColumnVariances([[1.0, 10.0], [3.0, 20.0]]);
  assert.is_true(r.ok)
  assert.equal(r.value, [1.0, 25.0])
}
test "column_invalid" {
  assert.is_true(!ColumnMeans([]).ok)
  assert.is_true(!ColumnMeans([[1.0], [2.0, 3.0]]).ok)
  assert.is_true(!ColumnVariances([]).ok)
  assert.is_true(!ColumnVariances([[1.0, 2.0], [3.0]]).ok)
}
test "column_single_row" {
  let r = ColumnVariances([[4.0, -2.0]]);
  assert.is_true(r.ok)
  assert.equal(r.value, [0.0, 0.0])
}
test "column_variance_stability" {
  let constant = ColumnVariances([[0.1], [0.1], [0.1]]);
  assert.is_true(constant.ok)
  assert.equal(constant.value, [0.0])
  let shifted = ColumnVariances([[1e12 - 1.0], [1e12 + 1.0]]);
  assert.is_true(shifted.ok)
  assert.is_true(ApproxRelative(shifted.value[0], 1.0, 1e-12))
}
test "vector_arithmetic" {
  assert.equal(VecAdd([1.0, 2.0], [10.0, 20.0]).value, [11.0, 22.0])
  assert.equal(VecSub([1.0, 2.0], [10.0, 20.0]).value, [-9.0, -18.0])
  assert.equal(VecScale([1.0, -2.0], 3.0).value, [3.0, -6.0])
  assert.is_true(!VecAdd([1.0], [1.0, 2.0]).ok)
  assert.is_true(!VecSub([1.0], []).ok)
}
test "matvec" {
  let r = MatVec([[1.0, 2.0], [3.0, 4.0]], [1.0, 1.0]);
  assert.is_true(r.ok)
  assert.equal(r.value, [3.0, 7.0])
  let bad = MatVec([[1.0, 2.0]], [1.0]);
  assert.is_true(!bad.ok)
  assert.equal(bad.error, "matvec: matrix column count differs from vector length")
  assert.is_true(!MatVec([], [1.0]).ok)
}
test "transpose" {
  let r = Transpose([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
  assert.is_true(r.ok)
  assert.equal(r.value, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
  assert.is_true(!Transpose([]).ok)
  assert.is_true(!Transpose([[1.0], [2.0, 3.0]]).ok)
}
test "nan_propagation" {
  let r = ColumnMeans([[1.0, float.nan()], [3.0, 4.0]]);
  assert.is_true(r.ok)
  assert.is_true(IsNaN(r.value[1]))
  assert.equal(r.value[0], 2.0)
}
test "nan_and_finiteness" {
  assert.is_true(!IsNaN(0.0))
  assert.is_true(IsFinite(-0.0))
  assert.is_true(IsFinite(1.0))
  assert.is_true(IsFinite(-42.5))
  assert.is_true(IsFinite(5e-324))
  assert.is_true(IsFinite(2.2250738585072014e-308))
  assert.is_true(IsFinite(1.7976931348623157e308))
  assert.is_true(IsFinite(-1.7976931348623157e308))
  assert.is_true(IsNaN(float.nan()))
  assert.is_true(!IsFinite(float.nan()))
  assert.is_true(!IsNaN(float.inf()))
  assert.is_true(!IsNaN(-float.inf()))
  assert.is_true(!IsFinite(float.inf()))
  assert.is_true(!IsFinite(-float.inf()))
}
test "finite_matrix" {
  let d = CheckFiniteMatrix([[1.0, -2.0], [0.0, 1.7976931348623157e308]]);
  assert.is_true(d.ok)
  assert.equal(d.rows, 2)
  assert.equal(d.cols, 2)
  assert.is_true(!CheckFiniteMatrix([]).ok)
  assert.is_true(!CheckFiniteMatrix([[1.0], [2.0, 3.0]]).ok)
  let n = CheckFiniteMatrix([[1.0, float.nan()], [2.0, 3.0]]);
  assert.is_true(!n.ok)
  assert.equal(n.error, "matrix contains NaN or infinite values")
  let inf = CheckFiniteMatrix([[1.0, 2.0], [-float.inf(), 3.0]]);
  assert.is_true(!inf.ok)
  assert.equal(inf.error, "matrix contains NaN or infinite values")
}
test "abs_and_sqrt" {
  assert.equal(Abs(-3.5), 3.5)
  assert.equal(Abs(3.5), 3.5)
  assert.equal(Abs(-0.0).signum(), 1.0)
  assert.equal(Abs(-float.inf()), float.inf())
  assert.is_true(IsNaN(Abs(float.nan())))
  assert.equal(Sqrt(0.0), 0.0)
  assert.equal(Sqrt(-0.0).signum(), 1.0)
  assert.equal(Sqrt(1.0), 1.0)
  assert.is_true(ApproxRelative(Sqrt(2.0), 1.4142135623730951, 1e-12))
  assert.equal(Sqrt(0.25), 0.5)
  assert.is_true(ApproxRelative(Sqrt(5e-324), 2.2227587494850775e-162, 1e-12))
  assert.is_true(ApproxRelative(Sqrt(2.2250738585072014e-308), 1.4916681462400413e-154, 1e-12))
  assert.is_true(ApproxRelative(Sqrt(1.7976931348623157e308), 1.3407807929942596e154, 1e-12))
  assert.equal(Sqrt(-4.0), 0.0)
  assert.equal(Sqrt(-float.inf()), 0.0)
  assert.equal(Sqrt(float.inf()), float.inf())
  assert.is_true(IsNaN(Sqrt(float.nan())))
}
test "vector_norm_extremes" {
  assert.equal(VecNorm([]), 0.0)
  assert.is_true(ApproxRelative(VecNorm([3.0, -4.0]), 5.0, 1e-12))
  assert.is_true(ApproxRelative(VecNorm([3e200, 4e200]), 5e200, 1e-12))
  assert.is_true(ApproxRelative(VecNorm([3e-200, 4e-200]), 5e-200, 1e-12))
  assert.equal(VecNorm([float.inf(), 1.0]), float.inf())
  assert.is_true(IsNaN(VecNorm([float.nan(), 1.0])))
}
test "exp_native" {
  assert.equal(Exp(0.0), 1.0)
  assert.is_true(ApproxRelative(Exp(1.0), 2.718281828459045, 1e-12))
  assert.is_true(ApproxRelative(Exp(-10.0), 4.5399929762484854e-5, 1e-12))
  assert.is_true(ApproxRelative(Exp(700.0), 1.0142320547350045e304, 1e-12))
  assert.equal(Exp(-1000.0), 0.0)
  assert.equal(Exp(1000.0), float.inf())
  assert.equal(Exp(-float.inf()), 0.0)
  assert.equal(Exp(float.inf()), float.inf())
  assert.is_true(IsNaN(Exp(float.nan())))
  assert.is_true(Exp(-720.0) > 0.0)
}
test "log_native_legacy_domain" {
  assert.equal(Log(1.0), 0.0)
  assert.is_true(ApproxRelative(Log(2.0), 0.6931471805599453, 1e-12))
  assert.is_true(ApproxRelative(Log(0.5), -0.6931471805599453, 1e-12))
  assert.is_true(ApproxRelative(Log(5e-324), -744.4400719213812, 1e-12))
  assert.equal(Log(0.0), -8.98846567431158e307)
  assert.equal(Log(-1.0), -8.98846567431158e307)
  assert.equal(Log(-float.inf()), -8.98846567431158e307)
  assert.equal(Log(float.inf()), float.inf())
  assert.is_true(IsNaN(Log(float.nan())))
}
