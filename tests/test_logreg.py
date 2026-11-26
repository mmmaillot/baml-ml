import math

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.model_selection import train_test_split

from baml_client import b, types


# ------------------------------------------------------------------ Exp / Log


# Relative error of the halving+squaring Exp grows like 2^depth * eps
# (~1e-13 at |x|=100, ~1e-11 at |x|=700); irrelevant for softmax, where logits
# are max-subtracted first.
@pytest.mark.parametrize(
    "x,rel",
    [(-20.0, 1e-12), (-1.0, 1e-12), (0.0, 1e-12), (0.5, 1e-12), (1.0, 1e-12),
     (20.0, 1e-12), (100.0, 1e-12), (-100.0, 1e-12), (700.0, 1e-9), (-700.0, 1e-9)],
)
def test_exp_accuracy(x, rel):
    assert b.Exp(x) == pytest.approx(math.exp(x), rel=rel)


def test_exp_edge_cases():
    assert b.Exp(-1000.0) == 0.0  # underflow to zero, stays finite
    assert b.Exp(math.inf) == math.inf
    assert math.isnan(b.Exp(math.nan))


@pytest.mark.parametrize("x", [1e-300, 0.5, 1.0, 2.0, math.e, 1e300])
def test_log_accuracy(x):
    assert b.Log(x) == pytest.approx(math.log(x), rel=1e-12)


def test_log_edges():
    assert b.Log(1.0) == 0.0
    assert b.Log(math.inf) == math.inf
    assert b.Log(0.0) < -1e300  # finite "-inf" sentinel


# ------------------------------------------------------- softmax / proba edges


def test_predict_proba_extreme_logits_stay_finite_and_sum_to_one():
    # weights produce logits of +/-1000
    model = types.LogisticRegressionModel(
        classes=[0, 1],
        weights=[[1000.0], [-1000.0]],
        intercepts=[0.0, 0.0],
        n_features=1,
        n_iter=1,
        converged=True,
    )
    res = b.LogRegPredictProba([[1.0], [-1.0], [0.0]], model)
    assert res.ok, res.error
    p = np.asarray(res.probs)
    assert np.all(np.isfinite(p))
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-12)
    assert p[0, 0] == pytest.approx(1.0)
    assert p[1, 1] == pytest.approx(1.0)
    assert p[2, 0] == pytest.approx(0.5)


def test_predict_uses_class_order_and_argmax():
    model = types.LogisticRegressionModel(
        classes=[7, 42],  # non-contiguous labels, sorted class order
        weights=[[0.0], [2.0]],
        intercepts=[0.0, 0.0],
        n_features=1,
        n_iter=1,
        converged=True,
    )
    pred = b.LogRegPredict([[5.0], [-5.0]], model)
    assert pred.ok
    assert pred.value == [42, 7]  # argmax -> classes[best]


# ------------------------------------------------------------- gradient check


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 2)).tolist()
    y = [0, 1, 0, 1, 0, 1]
    classes = [0, 1]
    c = 1.0
    theta = types.LogRegParams(
        weights=[[0.3, -0.2], [-0.1, 0.4]], intercepts=[0.05, -0.05]
    )
    grad = b.LogRegGradient(x, y, classes, theta, c)
    h = 1e-6
    for k in range(2):
        for j in range(2):
            wp = [list(r) for r in theta.weights]
            wm = [list(r) for r in theta.weights]
            wp[k][j] += h
            wm[k][j] -= h
            lp = b.LogRegLoss(x, y, classes, types.LogRegParams(weights=wp, intercepts=theta.intercepts), c)
            lm = b.LogRegLoss(x, y, classes, types.LogRegParams(weights=wm, intercepts=theta.intercepts), c)
            fd = (lp - lm) / (2.0 * h)
            assert grad.weights[k][j] == pytest.approx(fd, abs=1e-5, rel=1e-4)
        bp = list(theta.intercepts)
        bm = list(theta.intercepts)
        bp[k] += h
        bm[k] -= h
        lp = b.LogRegLoss(x, y, classes, types.LogRegParams(weights=theta.weights, intercepts=bp), c)
        lm = b.LogRegLoss(x, y, classes, types.LogRegParams(weights=theta.weights, intercepts=bm), c)
        fd = (lp - lm) / (2.0 * h)
        assert grad.intercepts[k] == pytest.approx(fd, abs=1e-5, rel=1e-4)


# ------------------------------------------------------------------ fit/predict


def test_fit_separable_three_classes():
    rng = np.random.default_rng(1)
    centers = np.array([[3.0, 3.0], [-3.0, 3.0], [0.0, -4.0]])
    xs = np.vstack([c + 0.1 * rng.normal(size=(20, 2)) for c in centers])
    ys = np.repeat([0, 1, 2], 20).tolist()
    fit = b.LogRegFit(xs.tolist(), ys, 1.0, 2000, 0.0000001)
    assert fit.ok, fit.error
    assert fit.model.converged
    assert fit.model.classes == [0, 1, 2]
    proba = b.LogRegPredictProba(xs.tolist(), fit.model)
    assert proba.ok
    p = np.asarray(proba.probs)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-12)
    pred = b.LogRegPredict(xs.tolist(), fit.model)
    assert np.mean(np.asarray(pred.value) == np.asarray(ys)) == 1.0


def test_iris_parity_with_converged_sklearn():
    X, y = load_iris(return_X_y=True)
    x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    # Standardize with our BAML scaler (fit on training data only).
    scaler = b.ScalerFit(x_tr.tolist())
    assert scaler.ok
    x_tr_s = b.ScalerTransform(x_tr.tolist(), scaler.scaler).x
    x_te_s = b.ScalerTransform(x_te.tolist(), scaler.scaler).x

    fit = b.LogRegFit(x_tr_s, y_tr.tolist(), 1.0, 5000, 0.00000001)
    assert fit.ok, fit.error
    assert fit.model.converged

    ours = np.asarray(b.LogRegPredictProba(x_te_s, fit.model).probs)
    ref = SkLogisticRegression(C=1.0, tol=1e-10, max_iter=10000).fit(x_tr_s, y_tr)
    expected = ref.predict_proba(x_te_s)
    assert np.allclose(ours, expected, atol=1e-3)  # target from issue #6

    pred = b.LogRegPredict(x_te_s, fit.model)
    assert pred.ok
    assert np.array_equal(np.asarray(pred.value), ref.predict(x_te_s))


def test_convergence_status_reported():
    X = [[0.5, -1.0], [1.5, 0.0], [-1.0, 1.0], [0.0, 0.5]]
    y = [0, 1, 0, 1]
    ok_fit = b.LogRegFit(X, y, 1.0, 500, 0.0000001)
    assert ok_fit.ok and ok_fit.model.converged
    stuck = b.LogRegFit(X, y, 1.0, 1, 0.0000001)
    assert not stuck.ok
    assert "did not converge" in stuck.error


def test_rejects_bad_shapes_and_invalid_args():
    X = [[0.5, -1.0], [1.5, 0.0], [-1.0, 1.0], [0.0, 0.5]]
    y = [0, 1, 0, 1]
    fit = b.LogRegFit(X, y, 1.0, 500, 0.0000001)
    assert fit.ok
    # feature count mismatch
    assert not b.LogRegPredictProba([[1.0, 2.0, 3.0]], fit.model).ok
    assert not b.LogRegPredict([[1.0, 2.0, 3.0]], fit.model).ok
    # malformed model states
    bad_dims = types.LogisticRegressionModel(
        classes=[0, 1], weights=[[1.0]], intercepts=[0.0, 0.0], n_features=1, n_iter=1, converged=True
    )
    r = b.LogRegPredictProba([[1.0]], bad_dims)
    assert not r.ok and "invalid fitted model" in r.error
    bad_nan = types.LogisticRegressionModel(
        classes=[0, 1], weights=[[math.nan], [1.0]], intercepts=[0.0, 0.0], n_features=1, n_iter=1, converged=True
    )
    r = b.LogRegPredictProba([[1.0]], bad_nan)
    assert not r.ok and "non-finite" in r.error
    # fit-time validation
    assert not b.LogRegFit(X, [0, 1, 0], 1.0, 500, 0.0000001).ok  # y length
    assert not b.LogRegFit(X, [1, 1, 1, 1], 1.0, 500, 0.0000001).ok  # one class
    assert not b.LogRegFit(X, y, 0.0, 500, 0.0000001).ok  # c <= 0
    assert not b.LogRegFit(X, y, 1.0, 0, 0.0000001).ok  # max_iter < 1
    assert not b.LogRegFit(X, y, 1.0, 500, -1.0).ok  # tol <= 0
    assert not b.LogRegFit([[math.nan, 1.0]], [0], 1.0, 500, 0.0000001).ok  # non-finite x
    assert not b.LogRegFit([], [], 1.0, 500, 0.0000001).ok
