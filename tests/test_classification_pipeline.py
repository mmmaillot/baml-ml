import math

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer as SkImputer
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler as SkScaler

from baml_client import b, types

NAN = math.nan

C = 1.0
MAX_ITER = 5000
TOL = 0.00000001

X_TRAIN = [[1.0, NAN, 2.0], [3.0, 4.0, NAN], [NAN, 10.0, 1.0], [5.0, 2.0, 3.0],
           [1.5, NAN, 0.5], [4.0, 3.0, NAN], [NAN, 9.0, 2.0], [6.0, 1.0, 4.0]]
Y_TRAIN = [0, 0, 0, 1, 0, 1, 0, 1]
X_TEST = [[NAN, 2.0, 1.0], [4.0, NAN, NAN]]


def _fit():
    fit = b.ClassificationPipelineFit(X_TRAIN, Y_TRAIN, C, MAX_ITER, TOL)
    assert fit.ok, fit.error
    return fit.pipeline


def _manual_composition(x_train, y_train, x_new):
    imp = b.ImputerFit(x_train)
    assert imp.ok, imp.error
    x1_train = b.ImputerTransform(x_train, imp.imputer).x
    sc = b.ScalerFit(x1_train)
    assert sc.ok, sc.error
    x2_train = b.ScalerTransform(x1_train, sc.scaler).x
    model = b.LogRegFit(x2_train, y_train, C, MAX_ITER, TOL)
    assert model.ok, model.error
    x1_new = b.ImputerTransform(x_new, imp.imputer).x
    x2_new = b.ScalerTransform(x1_new, sc.scaler).x
    proba = b.LogRegPredictProba(x2_new, model.model)
    assert proba.ok, proba.error
    pred = b.LogRegPredict(x2_new, model.model)
    assert pred.ok, pred.error
    return imp.imputer, sc.scaler, model.model, proba.probs, pred.value


def test_fit_predict_and_proba_end_to_end():
    p = _fit()
    assert p.imputer.n_features == 3
    assert p.scaler.n_features == 3
    assert p.model.n_features == 3
    assert p.model.classes == [0, 1]
    assert p.model.converged
    pred = b.ClassificationPipelinePredict(X_TEST, p)
    assert pred.ok, pred.error
    assert len(pred.value) == 2
    assert all(v in p.model.classes for v in pred.value)
    proba = b.ClassificationPipelinePredictProba(X_TEST, p)
    assert proba.ok, proba.error
    probs = np.asarray(proba.probs)
    assert probs.shape == (2, 2)
    assert np.all(np.isfinite(probs))
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-12)
    # predict is the argmax over predict_proba in the stored class order
    for i, row in enumerate(probs):
        assert pred.value[i] == p.model.classes[int(np.argmax(row))]


def test_predictions_match_manual_composition_of_same_steps():
    fit = b.ClassificationPipelineFit(X_TRAIN, Y_TRAIN, C, MAX_ITER, TOL)
    assert fit.ok
    ours_proba = b.ClassificationPipelinePredictProba(X_TEST, fit.pipeline)
    ours_pred = b.ClassificationPipelinePredict(X_TEST, fit.pipeline)
    assert ours_proba.ok and ours_pred.ok
    imp, sc, model, expected_proba, expected_pred = _manual_composition(X_TRAIN, Y_TRAIN, X_TEST)
    assert fit.pipeline.imputer.means == pytest.approx(imp.means)
    assert fit.pipeline.scaler.means == pytest.approx(sc.means)
    assert fit.pipeline.scaler.scales == pytest.approx(sc.scales)
    assert fit.pipeline.model.classes == model.classes
    assert np.allclose(fit.pipeline.model.weights, model.weights, atol=1e-15)
    assert np.allclose(fit.pipeline.model.intercepts, model.intercepts, atol=1e-15)
    assert np.allclose(np.asarray(ours_proba.probs), np.asarray(expected_proba), atol=1e-12)
    assert ours_pred.value == expected_pred


def test_preprocess_entry_point_matches_manual_impute_then_scale():
    p = _fit()
    ours = b.ClassificationPipelinePreprocess(X_TEST, p)
    assert ours.ok, ours.error
    shared = b.PipelinePreprocessSteps(X_TEST, p.imputer, p.scaler)
    assert shared.ok, shared.error
    x1 = b.ImputerTransform(X_TEST, p.imputer).x
    x2 = b.ScalerTransform(x1, p.scaler).x
    assert np.allclose(np.asarray(ours.value), np.asarray(x2), atol=1e-15)
    assert np.allclose(np.asarray(shared.value), np.asarray(x2), atol=1e-15)


def test_held_out_input_does_not_update_fitted_statistics():
    p = _fit()
    before_means = list(p.imputer.means)
    before_scales = list(p.scaler.scales)
    before_weights = [list(r) for r in p.model.weights]
    pred1 = b.ClassificationPipelinePredict(X_TEST, p)
    proba1 = b.ClassificationPipelinePredictProba(X_TEST, p)
    pred2 = b.ClassificationPipelinePredict(X_TEST, p)
    proba2 = b.ClassificationPipelinePredictProba(X_TEST, p)
    assert pred1.value == pred2.value
    assert np.allclose(np.asarray(proba1.probs), np.asarray(proba2.probs), atol=1e-15)
    assert p.imputer.means == pytest.approx(before_means)
    assert p.scaler.scales == pytest.approx(before_scales)
    assert np.allclose(np.asarray(p.model.weights), np.asarray(before_weights), atol=1e-15)
    direct_imp = b.ImputerFit(X_TRAIN)
    assert p.imputer.means == pytest.approx(direct_imp.imputer.means)


def test_rejects_incompatible_dimensions_and_bad_input():
    p = _fit()
    bad = b.ClassificationPipelinePredict([[1.0, 2.0]], p)
    assert not bad.ok
    assert not b.ClassificationPipelinePredictProba([[1.0, 2.0]], p).ok
    assert not b.ClassificationPipelinePredict([[1.0], [2.0, 3.0, 4.0]], p).ok
    assert not b.ClassificationPipelinePreprocess([[1.0, 2.0]], p).ok
    assert not b.ClassificationPipelineFit([[NAN, NAN, NAN], [NAN, NAN, NAN]], [0, 1], C, MAX_ITER, TOL).ok  # all-missing columns
    assert not b.ClassificationPipelineFit([], [], C, MAX_ITER, TOL).ok
    r = b.ClassificationPipelineFit(X_TRAIN, [0, 1, 0], C, MAX_ITER, TOL)
    assert not r.ok and "y length differs" in r.error  # y length mismatch
    r = b.ClassificationPipelineFit(X_TRAIN, [1] * len(X_TRAIN), C, MAX_ITER, TOL)
    assert not r.ok and "at least two distinct classes" in r.error  # one class
    r = b.ClassificationPipelineFit(X_TRAIN, Y_TRAIN, 0.0, MAX_ITER, TOL)
    assert not r.ok and "c must be finite and positive" in r.error  # c <= 0
    r = b.ClassificationPipelineFit(X_TRAIN, Y_TRAIN, C, 0, TOL)
    assert not r.ok and "max_iter must be >= 1" in r.error  # max_iter < 1
    r = b.ClassificationPipelineFit(X_TRAIN, Y_TRAIN, C, MAX_ITER, -1.0)
    assert not r.ok and "tol must be finite and positive" in r.error  # tol <= 0
    assert not b.ClassificationPipelineFit(X_TRAIN, Y_TRAIN, C, 1, TOL).ok  # non-convergence reported


def test_fit_validates_labels_and_hyperparameters_before_preprocessing():
    # Invalid hyperparameters/labels are rejected up front: the error must not
    # come from the imputer/scaler steps even when the data would also fail them.
    all_missing = [[NAN, NAN, NAN], [NAN, NAN, NAN]]
    r = b.ClassificationPipelineFit(all_missing, [0, 1], 0.0, MAX_ITER, TOL)
    assert not r.ok
    assert "classification_pipeline_fit: c must be finite and positive" in r.error
    r = b.ClassificationPipelineFit(all_missing, [0, 1], C, MAX_ITER, TOL)
    assert not r.ok
    assert "imputer_fit" in r.error  # data error surfaces only once args are valid
    r = b.ClassificationPipelineFit(X_TRAIN, [1] * len(X_TRAIN), C, MAX_ITER, TOL)
    assert not r.ok
    assert r.error.startswith("classification_pipeline_fit:")


@pytest.mark.parametrize(
    "bad_pipeline,expected_msg",
    [
        (b.EmptyClassificationPipeline(), "invalid fitted pipeline"),
        (
            types.ClassificationPipeline(
                imputer=types.MeanImputer(means=[0.0], n_features=1),
                scaler=types.StandardScaler(means=[0.0, 0.0], scales=[1.0, 1.0], n_features=2),
                model=types.LogisticRegressionModel(classes=[0, 1], weights=[[1.0], [1.0]], intercepts=[0.0, 0.0], n_features=1, n_iter=1, converged=True),
            ),
            "invalid fitted pipeline",
        ),
        (
            types.ClassificationPipeline(
                imputer=types.MeanImputer(means=[0.0], n_features=1),
                scaler=types.StandardScaler(means=[0.0], scales=[1.0], n_features=1),
                model=types.LogisticRegressionModel(classes=[0, 1], weights=[[1.0, 2.0], [1.0, 2.0]], intercepts=[0.0, 0.0], n_features=2, n_iter=1, converged=True),
            ),
            "invalid fitted pipeline",
        ),
        (
            types.ClassificationPipeline(
                imputer=types.MeanImputer(means=[0.0], n_features=1),
                scaler=types.StandardScaler(means=[0.0], scales=[0.0], n_features=1),
                model=types.LogisticRegressionModel(classes=[0, 1], weights=[[1.0], [1.0]], intercepts=[0.0, 0.0], n_features=1, n_iter=1, converged=True),
            ),
            "invalid scales",
        ),
        (
            types.ClassificationPipeline(
                imputer=types.MeanImputer(means=[0.0], n_features=1),
                scaler=types.StandardScaler(means=[0.0], scales=[1.0], n_features=1),
                model=types.LogisticRegressionModel(classes=[0, 1], weights=[[NAN], [1.0]], intercepts=[0.0, 0.0], n_features=1, n_iter=1, converged=True),
            ),
            "non-finite",
        ),
        (
            types.ClassificationPipeline(
                imputer=types.MeanImputer(means=[0.0], n_features=1),
                scaler=types.StandardScaler(means=[0.0], scales=[1.0], n_features=1),
                model=types.LogisticRegressionModel(classes=[0], weights=[[1.0]], intercepts=[0.0], n_features=1, n_iter=1, converged=True),
            ),
            "invalid fitted model",
        ),
    ],
)
def test_malformed_pipeline_state_rejected_without_vm_crash(bad_pipeline, expected_msg):
    res = b.ClassificationPipelinePredict([[1.0]], bad_pipeline)
    assert not res.ok
    assert expected_msg in res.error
    res_proba = b.ClassificationPipelinePredictProba([[1.0]], bad_pipeline)
    assert not res_proba.ok
    assert expected_msg in res_proba.error


def test_iris_with_missing_values_matches_sklearn_pipeline():
    X, y = load_iris(return_X_y=True)
    x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    rng = np.random.default_rng(0)
    x_tr = x_tr.copy()
    x_tr[rng.random(size=x_tr.shape) < 0.10] = np.nan
    x_tr[0, :] = 1.0  # no all-missing column
    x_te = x_te.copy()
    x_te[rng.random(size=x_te.shape) < 0.10] = np.nan

    fit = b.ClassificationPipelineFit(x_tr.tolist(), y_tr.tolist(), C, MAX_ITER, TOL)
    assert fit.ok, fit.error
    assert fit.pipeline.model.converged

    ours_proba = np.asarray(b.ClassificationPipelinePredictProba(x_te.tolist(), fit.pipeline).probs)
    ours_pred = np.asarray(b.ClassificationPipelinePredict(x_te.tolist(), fit.pipeline).value)

    ref = SkPipeline(
        [
            ("imputer", SkImputer(strategy="mean")),
            ("scaler", SkScaler()),
            ("clf", SkLogisticRegression(C=C, tol=1e-10, max_iter=10000)),
        ]
    ).fit(x_tr, y_tr)
    expected_proba = ref.predict_proba(x_te)
    assert np.allclose(ours_proba, expected_proba, atol=1e-3)
    assert np.array_equal(ours_pred, ref.predict(x_te))
    assert np.mean(ours_pred == y_te) >= 0.9
