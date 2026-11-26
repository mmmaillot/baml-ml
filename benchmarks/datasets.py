"""Shared dataset fixtures for the examples (#8) and the benchmarks (#9).

Deterministic: fixed seed 42, 80/20 train/test split, stratified for Iris
only (classification), plain split for Diabetes. The exact same arrays and
splits are handed to the BAML pipeline and to the sklearn reference.

An optional fixed-seed missing-value variant injects NaNs (10% of cells by
default) so the imputation stage is exercised; a column is never left
entirely missing, which both SimpleImputer implementations reject or drop.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.2
MISSING_FRAC = 0.10

DATASETS = ("iris", "diabetes")


@dataclass(frozen=True)
class Split:
    name: str
    task: str  # "classification" or "regression"
    missing_frac: float
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def inject_missing(x, frac, seed):
    """Return a copy of x with a deterministic `frac` of cells set to NaN.

    Any column that would become entirely missing keeps its first original
    value, so the fixture stays compatible with mean imputation.
    """
    out = np.array(x, dtype=float, copy=True)
    if frac > 0.0:
        rng = np.random.default_rng(seed)
        out[rng.random(out.shape) < frac] = np.nan
        for j in range(out.shape[1]):
            if np.isnan(out[:, j]).all():
                out[0, j] = x[0, j]
    return out


def load_split(name, missing_frac=0.0, seed=SEED):
    """Load one dataset and split it deterministically (seed 42 by default)."""
    if name == "iris":
        x, y = load_iris(return_X_y=True)
        stratify, task = y, "classification"
    elif name == "diabetes":
        x, y = load_diabetes(scaled=False, return_X_y=True)
        stratify, task = None, "regression"
    else:
        raise ValueError(f"unknown dataset: {name}")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=seed, stratify=stratify
    )
    if missing_frac > 0.0:
        x_train = inject_missing(x_train, missing_frac, seed)
        x_test = inject_missing(x_test, missing_frac, seed + 1)
    return Split(name, task, float(missing_frac), x_train, x_test, y_train, y_test)
