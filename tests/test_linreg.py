import math

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression as SkLinearRegression

from baml_client import b, types

# y = 3 + 2*x1 - 1*x2 exactly
X_EXACT = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 3.0], [4.0, 1.0]]
Y_EXACT = [3.0 + 2.0 * r[0] - 1.0 * r[1] for r in X_EXACT]


def test_recovers_coefficients_and_intercept_exactly():
    fit = b.LinearRegressionFit(X_EXACT, Y_EXACT)
    assert fit.ok, fit.error
    assert fit.model.n_features == 2
    assert fit.model.intercept == pytest.approx(3.0, abs=1e-9)
    assert fit.model.coefficients == pytest.approx([2.0, -1.0], abs=1e-9)


def test_predict_unseen_rows():
    fit = b.LinearRegressionFit(X_EXACT, Y_EXACT)
    assert fit.ok
    x_new = [[10.0, 1.0], [0.0, 0.0]]
    p = b.LinearRegressionPredict(x_new, fit.model)
    assert p.ok, p.error
    assert p.value == pytest.approx([3.0 + 20.0 - 1.0, 3.0], abs=1e-9)


def test_rejects_incompatible_feature_count():
    fit = b.LinearRegressionFit(X_EXACT, Y_EXACT)
    assert fit.ok
    p = b.LinearRegressionPredict([[1.0, 2.0, 3.0]], fit.model)
    assert not p.ok
    assert "feature count" in p.error


def test_rank_deficient_input_rejected():
    x = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]  # col2 = 2*col1
    y = [1.0, 2.0, 3.0]
    fit = b.LinearRegressionFit(x, y)
    assert not fit.ok
    assert "rank-deficient" in fit.error


def test_zero_variance_feature_is_rank_deficient():
    x = [[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]]  # constant column
    y = [1.0, 2.0, 3.0]
    fit = b.LinearRegressionFit(x, y)
    assert not fit.ok
    assert "rank-deficient" in fit.error


def test_underdetermined_system_fewer_rows_than_parameters_rejected():
    # n = 2 rows, p = 2 features -> [1 | X] is 2x3 (always rank-deficient).
    # Near-collinear columns previously tricked MGS via orthogonality loss.
    fit = b.LinearRegressionFit([[1.0, 0.0], [1.0 + 1e-9, 1.0]], [1.0, 2.0])
    assert not fit.ok
    assert "rank-deficient" in fit.error
    assert not b.LinearRegressionFit([[1.0]], [2.0]).ok  # n = 1 < p + 1 = 2


def test_large_finite_magnitudes_do_not_crash_vm():
    # 1e200 squares used to overflow VecNorm -> NaN -> ArrayIndexOutOfBounds.
    fit_200 = b.LinearRegressionFit([[1e200], [2e200], [3e200]], [1.0, 2.0, 3.0])
    assert fit_200.ok
    # Near DBL_MAX overflows QR intermediates cleanly without crashing the VM.
    fit_max = b.LinearRegressionFit([[1e308], [1.5e308]], [1.0, 2.0])
    assert not fit_max.ok
    assert fit_max.error != ""


def test_rejects_bad_shapes_and_non_finite_preserving_x_errors_first():
    empty_x = b.LinearRegressionFit([], [1.0])
    assert not empty_x.ok
    assert "no rows" in empty_x.error

    ragged_x = b.LinearRegressionFit([[1.0], [2.0, 3.0]], [1.0])
    assert not ragged_x.ok
    assert "ragged" in ragged_x.error

    nan_both = b.LinearRegressionFit([[math.nan]], [math.nan])
    assert not nan_both.ok
    assert "matrix contains NaN" in nan_both.error

    assert not b.LinearRegressionFit([[1.0, math.inf], [2.0, 3.0]], [1.0, 2.0]).ok
    bad_y_len = b.LinearRegressionFit([[1.0], [2.0]], [1.0])
    assert not bad_y_len.ok
    assert "y length" in bad_y_len.error
    bad_y_nan = b.LinearRegressionFit([[1.0], [2.0]], [1.0, math.nan])
    assert not bad_y_nan.ok
    assert "y contains NaN" in bad_y_nan.error


@pytest.mark.parametrize(
    "bad_model,expected_msg",
    [
        (b.EmptyLinRegModel(), "invalid fitted model"),
        (types.LinearRegressionModel(coefficients=[1.0], intercept=0.0, n_features=2), "invalid fitted model"),
        (types.LinearRegressionModel(coefficients=[math.nan, 1.0], intercept=0.0, n_features=2), "non-finite parameters"),
        (types.LinearRegressionModel(coefficients=[1.0, math.inf], intercept=0.0, n_features=2), "non-finite parameters"),
        (types.LinearRegressionModel(coefficients=[1.0, 1.0], intercept=math.nan, n_features=2), "non-finite parameters"),
        (types.LinearRegressionModel(coefficients=[1.0, 1.0], intercept=-math.inf, n_features=2), "non-finite parameters"),
    ],
)
def test_malformed_model_state_rejected(bad_model, expected_msg):
    p = b.LinearRegressionPredict([[1.0, 2.0]], bad_model)
    assert not p.ok
    assert expected_msg in p.error


def test_predict_rejects_non_finite_input_and_overflowed_output():
    fit = b.LinearRegressionFit(X_EXACT, Y_EXACT)
    assert fit.ok
    assert not b.LinearRegressionPredict([[1.0, math.nan]], fit.model).ok
    assert not b.LinearRegressionPredict([[1.0, math.inf]], fit.model).ok

    huge_model = types.LinearRegressionModel(coefficients=[1e200, -1e200], intercept=0.0, n_features=2)
    ovf_nan = b.LinearRegressionPredict([[1e200, 1e200]], huge_model)
    assert not ovf_nan.ok
    assert "not finite" in ovf_nan.error
    ovf_inf = b.LinearRegressionPredict([[1e200, 0.0]], huge_model)
    assert not ovf_inf.ok
    assert "not finite" in ovf_inf.error


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_sklearn_linear_regression(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(60, 4)) * 3.0
    true_w = np.array([1.5, -2.0, 0.5, 4.0])
    y = 0.7 + x @ true_w + rng.normal(scale=1e-6, size=60)  # well-conditioned
    fit = b.LinearRegressionFit(x.tolist(), y.tolist())
    assert fit.ok, fit.error

    ref = SkLinearRegression().fit(x, y)
    assert fit.model.intercept == pytest.approx(ref.intercept_, abs=1e-6, rel=1e-6)
    assert np.allclose(fit.model.coefficients, ref.coef_, atol=1e-6, rtol=1e-6)

    x_test = rng.normal(size=(10, 4)) * 3.0
    ours = np.asarray(b.LinearRegressionPredict(x_test.tolist(), fit.model).value)
    assert np.allclose(ours, ref.predict(x_test), atol=1e-6, rtol=1e-6)
