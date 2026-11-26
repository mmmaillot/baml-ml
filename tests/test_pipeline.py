import math

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler as SkScaler
from sklearn.impute import SimpleImputer as SkImputer

from baml_client import b, types

NAN = math.nan

X_TRAIN = [[1.0, NAN, 2.0], [3.0, 4.0, NAN], [NAN, 10.0, 1.0], [5.0, 2.0, 3.0]]
Y_TRAIN = [1.0, 2.5, 0.5, 4.0]
X_TEST = [[NAN, 2.0, 1.0], [4.0, NAN, NAN]]


def _manual_composition(x_train, y_train, x_new):
    imp = b.ImputerFit(x_train)
    assert imp.ok, imp.error
    x1_train = b.ImputerTransform(x_train, imp.imputer).x
    sc = b.ScalerFit(x1_train)
    assert sc.ok, sc.error
    x2_train = b.ScalerTransform(x1_train, sc.scaler).x
    model = b.LinearRegressionFit(x2_train, y_train)
    assert model.ok, model.error
    x1_new = b.ImputerTransform(x_new, imp.imputer).x
    x2_new = b.ScalerTransform(x1_new, sc.scaler).x
    return imp.imputer, sc.scaler, model.model, b.LinearRegressionPredict(x2_new, model.model).value


def test_fit_and_predict_end_to_end():
    fit = b.RegressionPipelineFit(X_TRAIN, Y_TRAIN)
    assert fit.ok, fit.error
    p = fit.pipeline
    assert p.imputer.n_features == 3
    assert p.scaler.n_features == 3
    assert p.model.n_features == 3
    pred = b.RegressionPipelinePredict(X_TEST, p)
    assert pred.ok, pred.error
    assert len(pred.value) == 2
    assert all(math.isfinite(v) for v in pred.value)


def test_predictions_match_manual_composition_of_same_steps():
    fit = b.RegressionPipelineFit(X_TRAIN, Y_TRAIN)
    assert fit.ok
    ours = b.RegressionPipelinePredict(X_TEST, fit.pipeline)
    assert ours.ok
    imp, sc, model, expected = _manual_composition(X_TRAIN, Y_TRAIN, X_TEST)
    assert fit.pipeline.imputer.means == pytest.approx(imp.means)
    assert fit.pipeline.scaler.means == pytest.approx(sc.means)
    assert np.allclose(fit.pipeline.model.coefficients, model.coefficients, atol=1e-15)
    assert ours.value == pytest.approx(expected, abs=1e-12)


def test_preprocess_entry_point_matches_manual_impute_then_scale():
    fit = b.RegressionPipelineFit(X_TRAIN, Y_TRAIN)
    assert fit.ok
    ours = b.RegressionPipelinePreprocess(X_TEST, fit.pipeline)
    assert ours.ok, ours.error
    shared = b.PipelinePreprocessSteps(X_TEST, fit.pipeline.imputer, fit.pipeline.scaler)
    assert shared.ok, shared.error
    x1 = b.ImputerTransform(X_TEST, fit.pipeline.imputer).x
    x2 = b.ScalerTransform(x1, fit.pipeline.scaler).x
    assert np.allclose(np.asarray(ours.value), np.asarray(x2), atol=1e-15)
    assert np.allclose(np.asarray(shared.value), np.asarray(x2), atol=1e-15)


def test_held_out_input_does_not_update_fitted_statistics():
    fit = b.RegressionPipelineFit(X_TRAIN, Y_TRAIN)
    assert fit.ok
    before_means = list(fit.pipeline.imputer.means)
    before_scales = list(fit.pipeline.scaler.scales)
    pred1 = b.RegressionPipelinePredict(X_TEST, fit.pipeline)
    pred2 = b.RegressionPipelinePredict(X_TEST, fit.pipeline)
    assert pred1.value == pytest.approx(pred2.value, abs=1e-15)
    assert fit.pipeline.imputer.means == pytest.approx(before_means)
    assert fit.pipeline.scaler.scales == pytest.approx(before_scales)
    direct = b.ImputerFit(X_TRAIN)
    assert fit.pipeline.imputer.means == pytest.approx(direct.imputer.means)


def test_rejects_incompatible_dimensions_and_bad_input():
    fit = b.RegressionPipelineFit(X_TRAIN, Y_TRAIN)
    assert fit.ok
    bad = b.RegressionPipelinePredict([[1.0, 2.0]], fit.pipeline)
    assert not bad.ok
    assert not b.RegressionPipelinePredict([[1.0], [2.0, 3.0, 4.0]], fit.pipeline).ok
    assert not b.RegressionPipelineFit([[NAN, NAN, NAN]], [1.0]).ok  # all-missing columns
    assert not b.RegressionPipelineFit([], []).ok
    assert not b.RegressionPipelineFit(X_TRAIN, [1.0, 2.0]).ok  # y length mismatch


@pytest.mark.parametrize(
    "bad_pipeline,expected_msg",
    [
        (b.EmptyRegressionPipeline(), "invalid fitted pipeline"),
        (
            types.RegressionPipeline(
                imputer=types.MeanImputer(means=[0.0], n_features=1),
                scaler=types.StandardScaler(means=[0.0, 0.0], scales=[1.0, 1.0], n_features=2),
                model=types.LinearRegressionModel(coefficients=[1.0], intercept=0.0, n_features=1),
            ),
            "invalid fitted pipeline",
        ),
        (
            types.RegressionPipeline(
                imputer=types.MeanImputer(means=[0.0], n_features=1),
                scaler=types.StandardScaler(means=[0.0], scales=[1.0], n_features=1),
                model=types.LinearRegressionModel(coefficients=[1.0, 2.0], intercept=0.0, n_features=2),
            ),
            "invalid fitted pipeline",
        ),
        (
            types.RegressionPipeline(
                imputer=types.MeanImputer(means=[0.0], n_features=1),
                scaler=types.StandardScaler(means=[0.0], scales=[0.0], n_features=1),
                model=types.LinearRegressionModel(coefficients=[1.0], intercept=0.0, n_features=1),
            ),
            "invalid scales",
        ),
        (
            types.RegressionPipeline(
                imputer=types.MeanImputer(means=[0.0], n_features=1),
                scaler=types.StandardScaler(means=[0.0], scales=[1.0], n_features=1),
                model=types.LinearRegressionModel(coefficients=[NAN], intercept=0.0, n_features=1),
            ),
            "non-finite parameters",
        ),
    ],
)
def test_malformed_pipeline_state_rejected_without_vm_crash(bad_pipeline, expected_msg):
    res = b.RegressionPipelinePredict([[1.0]], bad_pipeline)
    assert not res.ok
    assert expected_msg in res.error


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_sklearn_pipeline(seed):
    rng = np.random.default_rng(seed)
    x_clean = rng.normal(size=(50, 4)) * 4.0
    y_train = 2.0 + x_clean[:, 0] - 0.5 * x_clean[:, 2] + rng.normal(scale=0.01, size=50)
    x_train = x_clean.copy()
    x_train[rng.random(size=x_train.shape) < 0.15] = np.nan
    x_train[0, :] = 1.0  # no all-missing column
    x_test = rng.normal(size=(12, 4)) * 4.0
    x_test[rng.random(size=x_test.shape) < 0.2] = np.nan

    fit = b.RegressionPipelineFit(x_train.tolist(), y_train.tolist())
    assert fit.ok, fit.error
    ours = np.asarray(b.RegressionPipelinePredict(x_test.tolist(), fit.pipeline).value)

    ref = SkPipeline(
        [
            ("imputer", SkImputer(strategy="mean")),
            ("scaler", SkScaler()),
            ("lr", SkLinearRegression()),
        ]
    ).fit(x_train, y_train)
    expected = ref.predict(x_test)
    assert np.allclose(ours, expected, atol=1e-6, rtol=1e-6)
