"""Differential tests inspired by scikit-learn's numerical test suites (issue #14).

sklearn is used as an oracle only; all estimator math runs in BAML. No
third-party test code is copied. Intentional divergences from sklearn (rejecting
all-missing imputer columns, rejecting rank-deficient OLS inputs) are covered
by the dedicated tests in test_imputer.py / test_linreg.py and remain intact.

LogisticRegression extreme-logit cases are noted for issue #6 (not landed yet).
"""

import math

import numpy as np
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.preprocessing import StandardScaler as SkScaler

from baml_client import b

NAN = math.nan


# ---------------------------------------------------------------- StandardScaler


@pytest.mark.parametrize("const", [0.1, math.log(1e-5), 1e100, 1e-100, 0.0, -3.75])
def test_constant_columns_get_scale_one_and_near_zero_output(const):
    x = [[const] for _ in range(5)]
    ft = b.ScalerFitTransform(x)
    assert ft.ok, ft.error
    assert ft.scaler.scales == [1.0]
    assert all(abs(v) <= 1e-12 for row in ft.x for v in row)

    ref = SkScaler().fit(np.asarray(x))
    assert ref.scale_ == pytest.approx([1.0])
    assert np.allclose(np.asarray(ft.x), ref.transform(np.asarray(x)), atol=1e-12)


@pytest.mark.parametrize("base,delta", [(1.0, 1e-17), (1.0, 1e-30), (0.1, 1e-18), (1e100, 1e83)])
def test_near_constant_below_float64_precision_gets_scale_one(base, delta):
    # base + delta rounds to base in float64 -> bit-identical column.
    assert base + delta == base
    x = [[base], [base + delta], [base], [base + delta]]
    fit = b.ScalerFit(x)
    assert fit.ok, fit.error
    assert fit.scaler.scales == [1.0]
    t = b.ScalerTransform(x, fit.scaler)
    assert all(abs(v) <= 1e-12 for row in t.x for v in row)


def test_representable_small_scale_is_preserved():
    # mean +/- scale are representably distinct in float64: scale must be kept.
    x = [[1.0 - 1e-8], [1.0 + 1e-8]] * 3
    fit = b.ScalerFit(x)
    assert fit.ok, fit.error
    assert fit.scaler.scales[0] != 1.0
    assert fit.scaler.scales[0] == pytest.approx(1e-8, rel=1e-6)

    ref = SkScaler().fit(np.asarray(x))
    assert np.allclose(fit.scaler.scales, ref.scale_, rtol=1e-9)
    t = b.ScalerTransform(x, fit.scaler)
    assert np.allclose(np.asarray(t.x), ref.transform(np.asarray(x)), atol=1e-9)


def test_post_scaling_invariants_mixed_matrix():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(50, 3))
    x[:, 1] = 0.1  # constant column in the middle
    ft = b.ScalerFitTransform(x.tolist())
    assert ft.ok, ft.error
    z = np.asarray(ft.x)
    assert np.allclose(z[:, 0].mean(), 0.0, atol=1e-12)
    assert np.allclose(z[:, 0].std(), 1.0, atol=1e-9)
    assert np.allclose(z[:, 2].mean(), 0.0, atol=1e-12)
    assert np.allclose(z[:, 2].std(), 1.0, atol=1e-9)
    assert np.allclose(z[:, 1], 0.0, atol=1e-12)  # constant column -> zeros
    assert ft.scaler.scales[1] == 1.0

    ref = SkScaler().fit(x)
    assert np.allclose(ft.scaler.scales, ref.scale_, rtol=1e-9, atol=1e-12)
    assert np.allclose(z, ref.transform(x), atol=1e-9)


def test_scaler_differential_random_with_constant_column():
    rng = np.random.default_rng(21)
    x = rng.normal(size=(80, 4)) * rng.uniform(0.01, 50.0, size=(1, 4))
    x[:, 2] = math.log(1e-5)  # non-exact binary constant
    fit = b.ScalerFit(x.tolist())
    assert fit.ok, fit.error
    ours = np.asarray(b.ScalerTransform(x.tolist(), fit.scaler).x)
    ref = SkScaler().fit(x)
    assert fit.scaler.scales[2] == 1.0
    assert np.allclose(ours, ref.transform(x), atol=1e-9, rtol=1e-9)


# ------------------------------------------------------------------ SimpleImputer


def test_imputer_alternating_nan_rows_match_sklearn():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 3)) * 7.0
    x[::2] = np.nan  # alternating rows fully missing; odd rows observed
    fit = b.ImputerFit(x.tolist())
    assert fit.ok, fit.error
    ours = np.asarray(b.ImputerTransform(x.tolist(), fit.imputer).x)

    ref = SimpleImputer(strategy="mean").fit(x)
    assert np.allclose(fit.imputer.means, ref.statistics_, atol=1e-12, rtol=1e-12)
    assert np.allclose(ours, ref.transform(x), atol=1e-12, rtol=1e-12)
    # Non-missing entries preserved exactly.
    observed = ~np.isnan(x)
    assert np.array_equal(ours[observed], x[observed])


def test_imputer_single_observed_value_column():
    x = [[NAN, 1.0], [NAN, NAN], [4.0, NAN]]
    fit = b.ImputerFit(x)
    assert fit.ok, fit.error
    assert fit.imputer.means == pytest.approx([4.0, 1.0])
    t = b.ImputerTransform(x, fit.imputer)
    assert t.ok
    assert np.allclose(np.asarray(t.x), [[4.0, 1.0], [4.0, 1.0], [4.0, 1.0]], atol=1e-15)

    ref = SimpleImputer(strategy="mean").fit(np.asarray(x, dtype=float))
    assert np.allclose(fit.imputer.means, ref.statistics_, atol=1e-12)
    assert np.allclose(np.asarray(t.x), ref.transform(np.asarray(x, dtype=float)), atol=1e-12)


# --------------------------------------------------------------- LinearRegression


def test_linreg_strongly_correlated_full_rank_features_match_sklearn():
    rng = np.random.default_rng(3)
    n = 200
    x1 = rng.normal(size=n)
    x2 = 0.3 * x1 + 0.1 * rng.normal(size=n)  # strongly correlated, full rank
    x3 = rng.normal(size=n)
    x = np.column_stack([x1, x2, x3])
    y = 1.0 + 2.0 * x1 - 1.0 * x2 + 0.5 * x3 + 0.05 * rng.normal(size=n)

    fit = b.LinearRegressionFit(x.tolist(), y.tolist())
    assert fit.ok, fit.error
    ref = SkLinearRegression().fit(x, y)

    assert fit.model.intercept == pytest.approx(ref.intercept_, abs=1e-6, rel=1e-6)
    assert np.allclose(fit.model.coefficients, ref.coef_, atol=1e-6, rtol=1e-6)

    ours_pred = np.asarray(b.LinearRegressionPredict(x.tolist(), fit.model).value)
    ref_pred = ref.predict(x)
    assert np.allclose(ours_pred, ref_pred, atol=1e-6, rtol=1e-6)
    rmse_ours = float(np.sqrt(np.mean((ours_pred - y) ** 2)))
    rmse_ref = float(np.sqrt(np.mean((ref_pred - y) ** 2)))
    assert abs(rmse_ours - rmse_ref) < 1e-6


# ------------------------------------------------------------- LogisticRegression
# Issue #6 (LogisticRegression) has not landed yet: extreme-logit finiteness
# (|logit| = 1000, probabilities finite and rows summing to 1.0) and 3-class
# multinomial parity are noted here for #6 and will be added with it.
