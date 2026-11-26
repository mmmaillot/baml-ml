import math
import sys

import pytest

from baml_client import b


def test_dot_known_values():
    assert b.Dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]).value == 32.0
    assert b.Dot([-1.0, 2.0], [3.0, -4.0]).value == -11.0
    assert b.Dot([0.5], [2.0]).value == 1.0


def test_dot_empty_vectors_give_zero():
    result = b.Dot([], [])
    assert result.ok
    assert result.value == 0.0


def test_dot_length_mismatch_is_rejected():
    result = b.Dot([1.0], [1.0, 2.0])
    assert not result.ok
    assert "length" in result.error


def test_check_matrix_valid():
    d = b.CheckMatrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert d.ok
    assert d.rows == 2
    assert d.cols == 3
    assert d.error == ""


def test_check_matrix_rejects_empty_and_ragged():
    assert not b.CheckMatrix([]).ok
    assert not b.CheckMatrix([[]]).ok
    ragged = b.CheckMatrix([[1.0, 2.0], [3.0]])
    assert not ragged.ok
    assert "ragged" in ragged.error


def test_column_means_known_values():
    result = b.ColumnMeans([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
    assert result.ok
    assert result.value == pytest.approx([3.0, 20.0])


def test_column_variances_are_population_ddof0():
    result = b.ColumnVariances([[1.0, 10.0], [3.0, 20.0]])
    assert result.ok
    assert result.value == pytest.approx([1.0, 25.0])


def test_column_stats_reject_bad_shapes():
    assert not b.ColumnMeans([]).ok
    assert not b.ColumnMeans([[1.0], [2.0, 3.0]]).ok
    assert not b.ColumnVariances([]).ok
    assert not b.ColumnVariances([[1.0, 2.0], [3.0]]).ok


def test_column_stats_single_row_variance_is_zero():
    result = b.ColumnVariances([[4.0, -2.0]])
    assert result.ok
    assert result.value == pytest.approx([0.0, 0.0])


def test_vec_add_sub_scale():
    assert b.VecAdd([1.0, 2.0], [10.0, 20.0]).value == [11.0, 22.0]
    assert b.VecSub([1.0, 2.0], [10.0, 20.0]).value == [-9.0, -18.0]
    assert b.VecScale([1.0, -2.0], 3.0).value == [3.0, -6.0]
    assert not b.VecAdd([1.0], [1.0, 2.0]).ok
    assert not b.VecSub([1.0], []).ok


def test_matvec_known_values_and_shape_error():
    result = b.MatVec([[1.0, 2.0], [3.0, 4.0]], [1.0, 1.0])
    assert result.ok
    assert result.value == [3.0, 7.0]
    bad = b.MatVec([[1.0, 2.0]], [1.0])
    assert not bad.ok
    assert "matvec" in bad.error
    assert not b.MatVec([], [1.0]).ok


def test_transpose_known_values_and_shape_error():
    result = b.Transpose([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert result.ok
    assert result.value == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    assert not b.Transpose([]).ok
    assert not b.Transpose([[1.0], [2.0, 3.0]]).ok


def test_nan_propagates_through_primitives():
    nan = math.nan
    result = b.ColumnMeans([[1.0, nan], [3.0, 4.0]])
    assert result.ok
    assert math.isnan(result.value[1])
    assert result.value[0] == 2.0


def test_isnan_and_isfinite_ieee754_cases():
    for finite in [
        0.0,
        -0.0,
        1.0,
        -42.5,
        5e-324,
        sys.float_info.min,
        sys.float_info.max,
        -sys.float_info.max,
    ]:
        assert b.IsNaN(finite) is False
        assert b.IsFinite(finite) is True

    assert b.IsNaN(math.nan) is True
    assert b.IsFinite(math.nan) is False

    for inf in [math.inf, -math.inf]:
        assert b.IsNaN(inf) is False
        assert b.IsFinite(inf) is False


def test_check_finite_matrix():
    ok = b.CheckFiniteMatrix([[1.0, -2.0], [0.0, sys.float_info.max]])
    assert ok.ok
    assert (ok.rows, ok.cols) == (2, 2)

    assert not b.CheckFiniteMatrix([]).ok
    assert not b.CheckFiniteMatrix([[1.0], [2.0, 3.0]]).ok

    with_nan = b.CheckFiniteMatrix([[1.0, math.nan], [2.0, 3.0]])
    assert not with_nan.ok
    assert "NaN or infinite" in with_nan.error

    with_inf = b.CheckFiniteMatrix([[1.0, 2.0], [-math.inf, 3.0]])
    assert not with_inf.ok
    assert "NaN or infinite" in with_inf.error
