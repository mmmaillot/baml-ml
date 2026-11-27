// Inference-only implementation; training code is preserved at pre-inference-cleanup.

function ApproxRelative(actual: float, expected: float, rel: float) -> bool {
  if (expected == 0.0) { actual == 0.0 } else {
    (actual - expected).abs() <= expected.abs() * rel
  }
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

test "dot_unchecked_known_values" {
  assert.equal(DotUnchecked([1.0,2.0,3.0],[4.0,5.0,6.0]),32.0)
  assert.equal(DotUnchecked([-1.0,2.0],[3.0,-4.0]),-11.0)
  assert.equal(DotUnchecked([],[]),0.0)
}
test "abs_native" {
  assert.equal(Abs(-3.5),3.5)
  assert.equal(Abs(-0.0).signum(),1.0)
  assert.equal(Abs(-float.inf()),float.inf())
  assert.is_true(IsNaN(Abs(float.nan())))
}
