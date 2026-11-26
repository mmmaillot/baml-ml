import math

import numpy as np
import pytest
from sklearn.impute import SimpleImputer

from baml_client import b, types

NAN = math.nan
INF = math.inf

# Hand-computed fixture: column 0 mean = (1 + 3) / 2 = 2,
# column 1 mean = (4 + 10) / 2 = 7.
X_TRAIN = [[1.0, NAN], [3.0, 4.0], [NAN, 10.0]]
EXPECTED_MEANS = [2.0, 7.0]
EXPECTED_IMPUTED = [[1.0, 7.0], [3.0, 4.0], [2.0, 10.0]]


def assert_matrix_close(actual, expected):
    assert np.allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float))


def test_fit_hand_computed_means():
    result = b.ImputerFit(X_TRAIN)
    assert result.ok, result.error
    assert result.imputer.n_features == 2
    assert result.imputer.means == pytest.approx(EXPECTED_MEANS)


def test_transform_hand_computed_output():
    fit = b.ImputerFit(X_TRAIN)
    assert fit.ok
    result = b.ImputerTransform(X_TRAIN, fit.imputer)
    assert result.ok, result.error
    assert_matrix_close(result.x, EXPECTED_IMPUTED)


def test_complete_rows_unchanged_and_shape_preserved():
    fit = b.ImputerFit(X_TRAIN)
    x_new = [[5.0, 6.0], [NAN, NAN]]
    result = b.ImputerTransform(x_new, fit.imputer)
    assert result.ok
    assert len(result.x) == 2
    assert all(len(row) == 2 for row in result.x)
    assert result.x[0] == [5.0, 6.0]
    assert result.x[1] == pytest.approx(EXPECTED_MEANS)


def test_transform_uses_training_statistics_not_recomputed():
    fit = b.ImputerFit(X_TRAIN)
    # Held-out data with a very different column mean must still be filled
    # with the training means [2.0, 7.0].
    x_new = [[100.0, 200.0], [NAN, NAN]]
    result = b.ImputerTransform(x_new, fit.imputer)
    assert result.ok
    assert result.x[1] == pytest.approx(EXPECTED_MEANS)


def test_fit_transform_matches_fit_then_transform():
    ft = b.ImputerFitTransform(X_TRAIN)
    assert ft.ok, ft.error
    assert ft.imputer.means == pytest.approx(EXPECTED_MEANS)
    assert_matrix_close(ft.x, EXPECTED_IMPUTED)
    fit = b.ImputerFit(X_TRAIN)
    t = b.ImputerTransform(X_TRAIN, fit.imputer)
    assert_matrix_close(ft.x, t.x)


def test_all_missing_column_is_rejected():
    result = b.ImputerFit([[NAN, 1.0], [NAN, 2.0]])
    assert not result.ok
    assert "entirely missing" in result.error


def test_all_missing_column_rejected_by_fit_transform_too():
    result = b.ImputerFitTransform([[NAN, NAN]])
    assert not result.ok
    assert "entirely missing" in result.error


def test_incompatible_feature_count_is_rejected():
    fit = b.ImputerFit(X_TRAIN)
    assert fit.ok
    too_many = b.ImputerTransform([[1.0, 2.0, 3.0]], fit.imputer)
    assert not too_many.ok
    assert "feature count" in too_many.error
    too_few = b.ImputerTransform([[1.0]], fit.imputer)
    assert not too_few.ok


def test_ragged_or_empty_input_is_rejected():
    assert not b.ImputerFit([[1.0], [2.0, 3.0]]).ok
    assert not b.ImputerFit([]).ok
    fit = b.ImputerFit(X_TRAIN)
    assert not b.ImputerTransform([[1.0], [2.0, 3.0]], fit.imputer).ok


def test_no_missing_values_passes_through_unchanged():
    x = [[1.0, 2.0], [3.0, 4.0]]
    ft = b.ImputerFitTransform(x)
    assert ft.ok
    assert ft.imputer.means == pytest.approx([2.0, 3.0])
    assert_matrix_close(ft.x, x)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_matches_sklearn_simple_imputer_mean(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(25, 4)) * 10.0
    mask = rng.random(size=x.shape) < 0.25
    # Keep at least one observed value per column so the case is supported by
    # both implementations (sklearn drops all-missing columns by default).
    mask[0, :] = False
    x[mask] = np.nan

    fit = b.ImputerFit(x.tolist())
    assert fit.ok, fit.error
    transformed = b.ImputerTransform(x.tolist(), fit.imputer)
    assert transformed.ok, transformed.error

    ref = SimpleImputer(strategy="mean")
    ref.fit(x)
    expected = ref.transform(x)

    assert np.allclose(fit.imputer.means, ref.statistics_, atol=1e-12, rtol=1e-12)
    assert np.allclose(np.asarray(transformed.x), expected, atol=1e-12, rtol=1e-12)
    # Non-missing entries are preserved exactly.
    observed = ~mask
    assert np.array_equal(np.asarray(transformed.x)[observed], x[observed])


def test_sklearn_parity_on_held_out_rows():
    rng = np.random.default_rng(7)
    x_train = rng.normal(size=(30, 3))
    x_train[rng.random(size=x_train.shape) < 0.2] = np.nan
    x_train[0, :] = 1.0  # no all-missing column
    x_test = rng.normal(size=(10, 3))
    x_test[rng.random(size=x_test.shape) < 0.3] = np.nan

    fit = b.ImputerFit(x_train.tolist())
    assert fit.ok
    ours = np.asarray(b.ImputerTransform(x_test.tolist(), fit.imputer).x)

    ref = SimpleImputer(strategy="mean").fit(x_train)
    expected = ref.transform(x_test)
    assert np.allclose(ours, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "bad_train",
    [
        [[INF, 1.0], [2.0, 3.0]],
        [[-INF, 1.0], [2.0, 3.0]],
        [[INF, 1.0], [-INF, 2.0]],  # inf + (-inf) would otherwise yield NaN mean
    ],
)
def test_infinite_values_rejected_in_fit_and_fit_transform(bad_train):
    fit = b.ImputerFit(bad_train)
    assert not fit.ok
    assert "infinite" in fit.error

    ft = b.ImputerFitTransform(bad_train)
    assert not ft.ok
    assert "infinite" in ft.error


def test_overflowed_column_mean_is_rejected():
    fit = b.ImputerFit([[1e308], [1e308]])
    assert not fit.ok
    assert "not finite" in fit.error


@pytest.mark.parametrize("bad_val", [INF, -INF])
def test_infinite_values_rejected_in_transform(bad_val):
    fit = b.ImputerFit(X_TRAIN)
    assert fit.ok
    res = b.ImputerTransform([[bad_val, 1.0], [NAN, 2.0]], fit.imputer)
    assert not res.ok
    assert "infinite" in res.error


@pytest.mark.parametrize(
    "bad_imputer,expected_msg",
    [
        (b.EmptyImputer(), "invalid fitted imputer"),
        (types.MeanImputer(means=[1.0], n_features=2), "invalid fitted imputer"),
        (types.MeanImputer(means=[1.0, 2.0, 3.0], n_features=2), "invalid fitted imputer"),
        (types.MeanImputer(means=[NAN, 2.0], n_features=2), "non-finite means"),
        (types.MeanImputer(means=[INF, 2.0], n_features=2), "non-finite means"),
    ],
)
def test_malformed_imputer_state_is_rejected_without_vm_crash(bad_imputer, expected_msg):
    res = b.ImputerTransform([[NAN, NAN]], bad_imputer)
    assert not res.ok
    assert expected_msg in res.error
