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


def test_rejects_bad_shapes_and_non_finite():
    assert not b.LinearRegressionFit([], []).ok
    assert not b.LinearRegressionFit([[1.0], [2.0, 3.0]], [1.0, 2.0]).ok
    assert not b.LinearRegressionFit([[1.0, math.nan]], [1.0]).ok
    assert not b.LinearRegressionFit([[1.0, math.inf]], [1.0]).ok
    assert not b.LinearRegressionFit([[1.0]], [1.0, 2.0]).ok  # y length mismatch
    assert not b.LinearRegressionFit([[1.0]], [math.nan]).ok


def test_malformed_model_state_rejected():
    bad = types.LinearRegressionModel(coefficients=[1.0], intercept=0.0, n_features=2)
    p = b.LinearRegressionPredict([[1.0, 2.0]], bad)
    assert not p.ok
    assert "invalid fitted model" in p.error


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
