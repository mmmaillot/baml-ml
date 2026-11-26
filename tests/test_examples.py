"""End-to-end parity tests for the issue #8 examples: BAML pipeline vs the
explicit sklearn reference on identical seed-42 splits, clean and with the
fixed-seed missing-value variant, plus a no-preprocessing-leakage check.

Tolerances are explicit and justified by the solver difference:
- classification: BAML gradient descent + Armijo line search vs sklearn lbfgs
  both converge to the same L2-regularized multinomial optimum; measured max
  probability difference on these splits is ~1e-7, so 1e-3 leaves a wide margin
  while still catching real bugs; labels and accuracy must match exactly and
  log loss (measured diff ~1.5e-8) within 1e-6.
- regression: BAML OLS via QR (modified Gram-Schmidt) vs sklearn LAPACK on
  standardized features differ only at roundoff level (measured ~5e-13), so
  predictions, RMSE and R^2 are compared within 1e-9.
"""

import numpy as np
import pytest

from benchmarks.datasets import MISSING_FRAC, SEED, load_split
from benchmarks.reference import fit_baml_pipeline
from examples import diabetes_example, iris_example

PROBA_ATOL = 1e-3
LOGLOSS_ATOL = 1e-6
PRED_ATOL = 1e-9
METRIC_ATOL = 1e-9


@pytest.mark.parametrize("missing_frac", [0.0, MISSING_FRAC])
def test_iris_example_parity(missing_frac):
    m = iris_example.run(missing_frac=missing_frac, verbose=False)
    assert m["dataset"] == "iris"
    assert m["n_train"] == 120 and m["n_test"] == 30
    assert m["labels_match"]
    assert m["accuracy_baml"] == m["accuracy_ref"]
    assert m["max_proba_abs_diff"] <= PROBA_ATOL
    assert abs(m["log_loss_baml"] - m["log_loss_ref"]) <= LOGLOSS_ATOL


@pytest.mark.parametrize("missing_frac", [0.0, MISSING_FRAC])
def test_diabetes_example_parity(missing_frac):
    m = diabetes_example.run(missing_frac=missing_frac, verbose=False)
    assert m["dataset"] == "diabetes"
    assert m["n_train"] == 353 and m["n_test"] == 89
    assert m["max_pred_abs_diff"] <= PRED_ATOL
    assert abs(m["rmse_baml"] - m["rmse_ref"]) <= METRIC_ATOL
    assert abs(m["r2_baml"] - m["r2_ref"]) <= METRIC_ATOL


def test_splits_are_deterministic_and_seeded():
    a = load_split("iris", missing_frac=MISSING_FRAC)
    b = load_split("iris", missing_frac=MISSING_FRAC)
    assert a.x_train.shape == (120, 4) and a.x_test.shape == (30, 4)
    for f in ("x_train", "x_test", "y_train", "y_test"):
        assert np.array_equal(getattr(a, f), getattr(b, f), equal_nan=True)
    d = load_split("diabetes")
    assert d.x_train.shape[0] + d.x_test.shape[0] == 442
    # stratified Iris split keeps the class proportions
    for cls in (0, 1, 2):
        assert (a.y_test == cls).sum() == 10


@pytest.mark.parametrize("name", ["iris", "diabetes"])
def test_no_preprocessing_leakage(name):
    split = load_split(name, missing_frac=MISSING_FRAC)
    pipeline = fit_baml_pipeline(split)
    # Oracle in numpy: imputer means come from non-missing TRAINING cells only.
    train_means = np.nanmean(split.x_train, axis=0)
    assert pipeline.imputer.means == pytest.approx(list(train_means), abs=1e-12)
    x_imputed = np.where(np.isnan(split.x_train), train_means, split.x_train)
    assert pipeline.scaler.means == pytest.approx(list(x_imputed.mean(axis=0)), abs=1e-12)
    # If test rows leaked into the fit, the means would match the pooled stats.
    pooled = np.nanmean(np.vstack([split.x_train, split.x_test]), axis=0)
    assert not np.allclose(np.asarray(pipeline.imputer.means), pooled, atol=1e-12)


def test_missing_variant_actually_contains_nans_but_no_empty_column():
    for name in ("iris", "diabetes"):
        split = load_split(name, missing_frac=MISSING_FRAC)
        for x in (split.x_train, split.x_test):
            frac = float(np.isnan(x).mean())
            assert 0.02 < frac < 0.2
            assert not np.isnan(x).all(axis=0).any()


def test_seed_is_42():
    assert SEED == 42
