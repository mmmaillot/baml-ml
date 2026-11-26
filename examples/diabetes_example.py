"""End-to-end Diabetes example: BAML RegressionPipeline vs sklearn Pipeline (issue #8).

Run from the repository root on a clean checkout:

    .venv/bin/python examples/diabetes_example.py

Prints sklearn-reference and BAML metrics for the clean dataset
(load_diabetes(scaled=False)) and for a deterministic fixed-seed missing-value
variant (10% NaN, seed 42). Both runtimes consume the exact same seed-42 80/20
split; preprocessing is fitted on training rows only. Correctness only: no
performance claim. Automated parity assertions with explicit tolerances live
in tests/test_examples.py.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error

from benchmarks.datasets import MISSING_FRAC, load_split
from benchmarks.reference import fit_baml_pipeline, make_sklearn_pipeline, predict_baml


def run(missing_frac=0.0, verbose=True):
    split = load_split("diabetes", missing_frac=missing_frac)
    pipeline = fit_baml_pipeline(split)
    pred_baml, _ = predict_baml(split, pipeline, split.x_test)

    ref = make_sklearn_pipeline(split.task).fit(split.x_train, split.y_train)
    pred_ref = ref.predict(split.x_test)

    y = split.y_test
    metrics = {
        "dataset": split.name,
        "missing_frac": split.missing_frac,
        "n_train": int(split.x_train.shape[0]),
        "n_test": int(split.x_test.shape[0]),
        "rmse_baml": float(root_mean_squared_error(y, pred_baml)),
        "rmse_ref": float(root_mean_squared_error(y, pred_ref)),
        "r2_baml": float(r2_score(y, pred_baml)),
        "r2_ref": float(r2_score(y, pred_ref)),
        "max_pred_abs_diff": float(np.max(np.abs(pred_baml - pred_ref))),
    }
    if verbose:
        tag = "missing" if missing_frac > 0.0 else "clean"
        print(f"[diabetes/{tag}] n_train={metrics['n_train']} n_test={metrics['n_test']}")
        print(f"  rmse  baml={metrics['rmse_baml']:.8f} ref={metrics['rmse_ref']:.8f}")
        print(f"  r2    baml={metrics['r2_baml']:.8f} ref={metrics['r2_ref']:.8f}")
        print(f"  max |pred diff|={metrics['max_pred_abs_diff']:.3e}")
    return metrics


def main():
    run(0.0)
    run(MISSING_FRAC)


if __name__ == "__main__":
    main()
