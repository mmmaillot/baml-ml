"""Generate committed schema v3 fixtures and an independent sklearn 1.8.0 oracle.

Usage: python artifacts/generate_v3_fixtures.py [output_dir]

Writes <output_dir>/v3/<name>.json artifacts plus <output_dir>/v3/oracle.json
with the batch input and the sklearn 1.8.0 predict / predict_proba reference
for each fixture. The committed copies live in artifacts/fixtures/v3/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import load_diabetes, load_iris
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

assert sklearn.__version__ == "1.8.0", sklearn.__version__

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_sklearn import export_pipeline_v3

IRIS_FEATURES = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
DIABETES_FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]


def _iris_with_missing():
    x, y = load_iris(return_X_y=True)
    rng = np.random.default_rng(42)
    x_train, y_train = x.copy(), y.copy()
    x_train[3, 0] = np.nan
    x_train[17, 2] = np.nan
    return x_train, y_train, rng


def _diabetes_with_missing():
    x, y = load_diabetes(return_X_y=True)
    x_train, y_train = x.copy(), y.copy()
    x_train[5, 2] = np.nan
    return x_train, y_train


def build_fixtures():
    fixtures = {}

    # 0-step pipeline: bare LinearRegression on complete numeric data.
    xd, yd = _diabetes_with_missing()
    xd = np.delete(xd, 5, axis=0)
    yd = np.delete(yd, 5)
    lin = Pipeline([("est", LinearRegression())]).fit(xd, yd)
    rows = [[float(v) for v in xd[0]], [float(v) for v in xd[1]]]
    fixtures["v3-linear-nosteps"] = (
        lin, DIABETES_FEATURES, None,
        {"features": DIABETES_FEATURES, "rows": rows},
    )

    # 1-step pipeline: StandardScaler -> LogisticRegression on clean Iris.
    xi_clean, yi_clean = load_iris(return_X_y=True)
    xi, yi, _ = _iris_with_missing()
    scaler_logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(xi_clean, yi_clean)
    rows = [[float(v) for v in xi_clean[0]], [float(v) for v in xi_clean[50]]]
    fixtures["v3-logreg-scaler"] = (
        scaler_logreg, IRIS_FEATURES, None,
        {"features": IRIS_FEATURES, "rows": rows},
    )

    # Multi-step pipeline: SimpleImputer(mean) -> StandardScaler -> LogisticRegression.
    imp_scale_logreg = make_pipeline(
        SimpleImputer(strategy="mean"), StandardScaler(), LogisticRegression(max_iter=500)
    ).fit(xi, yi)
    rows = [[None, 3.0, 1.3, 0.2], [6.1, 3.0, 4.9, 1.8]]
    fixtures["v3-logreg-impute-scale"] = (
        imp_scale_logreg, IRIS_FEATURES, None,
        {"features": IRIS_FEATURES, "rows": rows},
    )

    # 1-step pipeline without a scaler: SimpleImputer -> DecisionTreeClassifier.
    imp_tree = make_pipeline(SimpleImputer(strategy="mean"),
                             DecisionTreeClassifier(max_depth=3, random_state=42)).fit(xi, yi)
    rows = [[None, 3.0, 1.3, 0.2], [6.1, 3.0, 4.9, 1.8]]
    fixtures["v3-tree-impute"] = (
        imp_tree, IRIS_FEATURES, None,
        {"features": IRIS_FEATURES, "rows": rows},
    )

    # 1-step pipeline: SimpleImputer -> LinearRegression on diabetes.
    xd2, yd2 = _diabetes_with_missing()
    imp_lin = make_pipeline(SimpleImputer(strategy="mean"), LinearRegression()).fit(xd2, yd2)
    rows = [[None] + [float(v) for v in xd2[0, 1:]], [float(v) for v in xd2[1]]]
    fixtures["v3-linreg-impute"] = (
        imp_lin, DIABETES_FEATURES, None,
        {"features": DIABETES_FEATURES, "rows": rows},
    )
    return fixtures


def _batch_matrix(batch):
    return np.array([[np.nan if v is None else float(v) for v in row] for row in batch["rows"]],
                    dtype=np.float64)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/fixtures")
    v3 = out / "v3"
    v3.mkdir(parents=True, exist_ok=True)
    oracle = {}
    for name, (pipeline, features, kinds, batch) in build_fixtures().items():
        artifact = export_pipeline_v3(pipeline, features, kinds)
        with open(v3 / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2, allow_nan=False)
            f.write("\n")
        x = _batch_matrix(batch)
        pred = pipeline.predict(x)
        entry = {"input": batch, "is_classifier": hasattr(pipeline, "predict_proba")}
        if entry["is_classifier"]:
            entry["predict_labels"] = [int(v) for v in pred]
            entry["predict_proba"] = pipeline.predict_proba(x).tolist()
        else:
            entry["predict_values"] = [float(v) for v in pred]
        oracle[name] = entry
        print(f"wrote {v3 / (name + '.json')}")
    with open(v3 / "oracle.json", "w", encoding="utf-8") as f:
        json.dump(oracle, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"wrote {v3 / 'oracle.json'}")


if __name__ == "__main__":
    main()
