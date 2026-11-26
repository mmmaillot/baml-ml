"""End-to-end Iris example: BAML ClassificationPipeline vs sklearn Pipeline (issue #8).

Run from the repository root on a clean checkout:

    .venv/bin/python examples/iris_example.py

Prints sklearn-reference and BAML metrics for the clean dataset and for a
deterministic fixed-seed missing-value variant (10% NaN, seed 42), so the
imputation stage is exercised. Both runtimes consume the exact same seed-42
80/20 split (stratified); preprocessing is fitted on training rows only.
Correctness only: no performance claim. Automated parity assertions with
explicit tolerances live in tests/test_examples.py.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from benchmarks.datasets import MISSING_FRAC, load_split
from benchmarks.reference import fit_baml_pipeline, make_sklearn_pipeline, predict_baml


def run(missing_frac=0.0, verbose=True):
    split = load_split("iris", missing_frac=missing_frac)
    pipeline = fit_baml_pipeline(split)
    pred_baml, proba_baml = predict_baml(split, pipeline, split.x_test)

    ref = make_sklearn_pipeline(split.task).fit(split.x_train, split.y_train)
    pred_ref = ref.predict(split.x_test)
    proba_ref = ref.predict_proba(split.x_test)

    y = split.y_test
    metrics = {
        "dataset": split.name,
        "missing_frac": split.missing_frac,
        "n_train": int(split.x_train.shape[0]),
        "n_test": int(split.x_test.shape[0]),
        "accuracy_baml": float(accuracy_score(y, pred_baml)),
        "accuracy_ref": float(accuracy_score(y, pred_ref)),
        "log_loss_baml": float(log_loss(y, proba_baml)),
        "log_loss_ref": float(log_loss(y, proba_ref)),
        "max_proba_abs_diff": float(np.max(np.abs(proba_baml - proba_ref))),
        "labels_match": bool(np.array_equal(pred_baml, pred_ref)),
    }
    if verbose:
        tag = "missing" if missing_frac > 0.0 else "clean"
        print(f"[iris/{tag}] n_train={metrics['n_train']} n_test={metrics['n_test']}")
        print(f"  accuracy  baml={metrics['accuracy_baml']:.6f} ref={metrics['accuracy_ref']:.6f}")
        print(f"  log_loss  baml={metrics['log_loss_baml']:.8f} ref={metrics['log_loss_ref']:.8f}")
        print(f"  max |proba diff|={metrics['max_proba_abs_diff']:.3e} labels_match={metrics['labels_match']}")
    return metrics


def main():
    run(0.0)
    run(MISSING_FRAC)


if __name__ == "__main__":
    main()
