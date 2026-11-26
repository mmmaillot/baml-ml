import math

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from baml_client import b, types

X = [[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]]
# means = [3, 20]; population variances = [8/3, 200/3]
EXPECTED_MEANS = [3.0, 20.0]
EXPECTED_VARIANCES = [8.0 / 3.0, 200.0 / 3.0]


def test_sqrt_newton_matches_math():
    for v in [0.0, 1.0, 2.0, 0.25, 1e6, 1e-6, 123456.789]:
        assert b.Sqrt(v) == pytest.approx(math.sqrt(v), rel=1e-12, abs=1e-300)
    assert b.Sqrt(0.0) == 0.0


def test_vec_norm():
    assert b.VecNorm([3.0, 4.0]) == pytest.approx(5.0)
    assert b.VecNorm([]) == 0.0


def test_fit_hand_computed():
    fit = b.ScalerFit(X)
    assert fit.ok, fit.error
    assert fit.scaler.n_features == 2
    assert fit.scaler.means == pytest.approx(EXPECTED_MEANS)
    assert fit.scaler.scales == pytest.approx([math.sqrt(v) for v in EXPECTED_VARIANCES])


def test_transform_hand_computed():
    fit = b.ScalerFit(X)
    assert fit.ok
    t = b.ScalerTransform(X, fit.scaler)
    assert t.ok, t.error
    expected = [
        [(X[i][j] - EXPECTED_MEANS[j]) / math.sqrt(EXPECTED_VARIANCES[j]) for j in range(2)]
        for i in range(3)
    ]
    assert np.allclose(np.asarray(t.x), np.asarray(expected), atol=1e-12)


def test_constant_column_yields_zero_for_training_rows():
    x = [[1.0, 5.0], [3.0, 5.0], [7.0, 5.0]]
    fit = b.ScalerFit(x)
    assert fit.ok
    assert fit.scaler.scales[1] == 1.0
    t = b.ScalerTransform(x, fit.scaler)
    assert t.ok
    assert [row[1] for row in t.x] == pytest.approx([0.0, 0.0, 0.0])


def test_transform_uses_training_statistics_on_held_out_rows():
    fit = b.ScalerFit(X)
    assert fit.ok
    x_new = [[100.0, 200.0]]
    t = b.ScalerTransform(x_new, fit.scaler)
    assert t.ok
    expected = [(100.0 - 3.0) / math.sqrt(8.0 / 3.0), (200.0 - 20.0) / math.sqrt(200.0 / 3.0)]
    assert t.x[0] == pytest.approx(expected)


def test_fit_transform_matches_fit_then_transform():
    ft = b.ScalerFitTransform(X)
    assert ft.ok, ft.error
    fit = b.ScalerFit(X)
    t = b.ScalerTransform(X, fit.scaler)
    assert ft.scaler.means == pytest.approx(fit.scaler.means)
    assert np.allclose(np.asarray(ft.x), np.asarray(t.x), atol=1e-15)


def test_rejects_non_finite_or_bad_shapes():
    assert not b.ScalerFit([]).ok
    assert not b.ScalerFit([[1.0], [2.0, 3.0]]).ok
    assert not b.ScalerFit([[1.0, math.nan]]).ok
    assert not b.ScalerFit([[1.0, math.inf]]).ok
    assert not b.ScalerFitTransform([[1.0, math.nan]]).ok
    fit = b.ScalerFit(X)
    assert not b.ScalerTransform([[1.0, 2.0, 3.0]], fit.scaler).ok
    assert not b.ScalerTransform([[1.0], [2.0, math.nan]], fit.scaler).ok


def test_fit_and_transform_reject_overflow_to_infinity():
    # Mean overflows: 1e308 + 1e308 -> inf
    mean_ovf = b.ScalerFit([[1e308], [1e308]])
    assert not mean_ovf.ok
    assert "not finite" in mean_ovf.error

    # Mean is 0.0 (finite), but squared deviations overflow: (1e200)^2 -> inf
    var_ovf = b.ScalerFit([[1e200], [-1e200]])
    assert not var_ovf.ok
    assert "non-finite variance" in var_ovf.error

    # Valid tiny-scale training set, held-out row overflows (x - mean) / scale
    tiny_fit = b.ScalerFit([[0.0], [1e-100]])
    assert tiny_fit.ok
    out_ovf = b.ScalerTransform([[1e250]], tiny_fit.scaler)
    assert not out_ovf.ok
    assert "not finite" in out_ovf.error


@pytest.mark.parametrize(
    "bad_scaler,expected_msg",
    [
        (b.EmptyScaler(), "invalid fitted scaler"),
        (types.StandardScaler(means=[1.0], scales=[1.0, 2.0], n_features=2), "invalid fitted scaler"),
        (types.StandardScaler(means=[1.0, 2.0], scales=[1.0], n_features=2), "invalid fitted scaler"),
        (types.StandardScaler(means=[math.nan, 0.0], scales=[1.0, 1.0], n_features=2), "non-finite means"),
        (types.StandardScaler(means=[math.inf, 0.0], scales=[1.0, 1.0], n_features=2), "non-finite means"),
        (types.StandardScaler(means=[0.0, 0.0], scales=[0.0, 1.0], n_features=2), "invalid scales"),
        (types.StandardScaler(means=[0.0, 0.0], scales=[-0.0, 1.0], n_features=2), "invalid scales"),
        (types.StandardScaler(means=[0.0, 0.0], scales=[-2.0, 1.0], n_features=2), "invalid scales"),
        (types.StandardScaler(means=[0.0, 0.0], scales=[math.nan, 1.0], n_features=2), "invalid scales"),
        (types.StandardScaler(means=[0.0, 0.0], scales=[math.inf, 1.0], n_features=2), "invalid scales"),
    ],
)
def test_malformed_scaler_state_rejected_without_vm_crash(bad_scaler, expected_msg):
    res = b.ScalerTransform([[1.0, 2.0]], bad_scaler)
    assert not res.ok
    assert expected_msg in res.error


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_sklearn_standard_scaler(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(40, 5)) * rng.uniform(0.1, 100.0, size=(1, 5))
    fit = b.ScalerFit(x.tolist())
    assert fit.ok, fit.error
    t = b.ScalerTransform(x.tolist(), fit.scaler)
    assert t.ok

    ref = StandardScaler().fit(x)
    expected = ref.transform(x)
    assert np.allclose(fit.scaler.means, ref.mean_, atol=1e-10, rtol=1e-10)
    assert np.allclose(fit.scaler.scales, ref.scale_, atol=1e-10, rtol=1e-10)
    assert np.allclose(np.asarray(t.x), expected, atol=1e-9, rtol=1e-9)


def test_sklearn_parity_on_held_out_rows():
    rng = np.random.default_rng(11)
    x_train = rng.normal(size=(30, 3)) * 5.0
    x_test = rng.normal(size=(8, 3)) * 5.0
    fit = b.ScalerFit(x_train.tolist())
    assert fit.ok
    ours = np.asarray(b.ScalerTransform(x_test.tolist(), fit.scaler).x)
    expected = StandardScaler().fit(x_train).transform(x_test)
    assert np.allclose(ours, expected, atol=1e-9, rtol=1e-9)
