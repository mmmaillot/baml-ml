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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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

    # Extended numeric imputer strategies (#74).
    med = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        LogisticRegression(max_iter=500)).fit(xi, yi)
    fixtures["v3-impute-median-scale-logreg"] = (
        med, IRIS_FEATURES, None, {"features": IRIS_FEATURES, "rows": [[None, 3.0, 1.3, 0.2], [6.1, 3.0, 4.9, 1.8]]},
    )
    mf_tree = make_pipeline(SimpleImputer(strategy="most_frequent"),
                            DecisionTreeClassifier(max_depth=3, random_state=42)).fit(xi, yi)
    fixtures["v3-impute-most-frequent-tree"] = (
        mf_tree, IRIS_FEATURES, None, {"features": IRIS_FEATURES, "rows": [[None, 3.0, 1.3, 0.2], [6.1, 3.0, 4.9, 1.8]]},
    )
    const_lin = make_pipeline(SimpleImputer(strategy="constant", fill_value=-1.0),
                              LinearRegression()).fit(xd2, yd2)
    fixtures["v3-impute-constant-linreg"] = (
        const_lin, DIABETES_FEATURES, None,
        {"features": DIABETES_FEATURES, "rows": [[None] + [float(v) for v in xd2[0, 1:]], [float(v) for v in xd2[1]]]},
    )

    # Categorical data with a missing training cell (#74 string imputer + OHE).
    colors = ["red", "green", "blue", "red", "green", "blue", "red", "green", "blue", "red", "green", "blue"]
    sizes = ["S", "M", "L", "M", "L", "S", "S", "M", "L", "M", "L", "S"]
    ycat = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1])
    xcat = np.array([[c, s] for c, s in zip(colors, sizes)], dtype=object)
    xcat_miss = xcat.copy()
    xcat_miss[0, 0] = np.nan
    cat_features = ["color", "size"]
    str_imp_ohe = make_pipeline(
        SimpleImputer(strategy="most_frequent"), OneHotEncoder(sparse_output=False), LogisticRegression(max_iter=500)
    ).fit(xcat_miss, ycat)
    fixtures["v3-strimpute-ohe-logreg"] = (
        str_imp_ohe, cat_features, ["categorical", "categorical"],
        {"features": cat_features, "rows": [[None, "M"], ["blue", "L"], ["red", None]]},
    )
    ohe_drop = make_pipeline(OneHotEncoder(drop="first", sparse_output=False), LogisticRegression(max_iter=500)).fit(xcat, ycat)
    fixtures["v3-ohe-drop-first"] = (
        ohe_drop, cat_features, ["categorical", "categorical"],
        {"features": cat_features, "rows": [["red", "S"], ["blue", "L"]]},
    )
    ohe_bin = make_pipeline(OneHotEncoder(drop="if_binary", sparse_output=False), LogisticRegression(max_iter=500)).fit(xcat, ycat)
    fixtures["v3-ohe-if-binary"] = (
        ohe_bin, cat_features, ["categorical", "categorical"],
        {"features": cat_features, "rows": [["green", "M"], ["red", "S"]]},
    )
    ohe_ignore = make_pipeline(OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                               LogisticRegression(max_iter=500)).fit(xcat, ycat)
    fixtures["v3-ohe-ignore-unseen"] = (
        ohe_ignore, cat_features, ["categorical", "categorical"],
        {"features": cat_features, "rows": [["purple", "XL"], ["red", "S"]]},
    )
    ohe_error = make_pipeline(OneHotEncoder(handle_unknown="error", sparse_output=False),
                              LogisticRegression(max_iter=500)).fit(xcat, ycat)
    fixtures["v3-ohe-error"] = (
        ohe_error, cat_features, ["categorical", "categorical"],
        {"features": cat_features, "rows": [["red", "S"], ["blue", "L"]]},
    )
    xint = np.array([[1, 2], [2, 3], [3, 1], [1, 3], [2, 1], [3, 2], [1, 1], [2, 2], [3, 3], [1, 2], [2, 3], [3, 1]],
                    dtype=np.int64)
    ohe_int = make_pipeline(OneHotEncoder(sparse_output=False), LogisticRegression(max_iter=500)).fit(xint, ycat)
    fixtures["v3-ohe-int-categories"] = (
        ohe_int, ["code_a", "code_b"], ["categorical", "categorical"],
        {"features": ["code_a", "code_b"], "rows": [[1, 2], [3, 3]]},
    )

    # #75: heterogeneous numeric + categorical ColumnTransformer pipelines.
    rng = np.random.default_rng(7)
    n = 24
    age = rng.uniform(20.0, 60.0, n)
    income = rng.uniform(30.0, 90.0, n) * 1000.0
    cities = np.array(["paris", "lyon", "berlin", "madrid"], dtype=object)[rng.integers(0, 4, n)]
    xh = np.empty((n, 3), dtype=object)
    xh[:, 0] = age
    xh[:, 1] = income
    xh[:, 2] = cities
    xh[2, 0] = np.nan
    xh[5, 2] = np.nan
    yh = (age + income / 1000.0 > 80.0).astype(int)
    hetero_features = ["age", "income", "city"]
    hetero_kinds = ["numeric", "numeric", "categorical"]
    ct = ColumnTransformer(
        [("num", make_pipeline(SimpleImputer(), StandardScaler()), [0, 1]),
         ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                               OneHotEncoder(sparse_output=False)), [2])],
        remainder="drop",
    )
    hetero_rows = [[None, 55000.0, "paris"], [30.0, None, None], [45.0, 60000.0, "berlin"]]
    ct_logreg = Pipeline([("ct", ct), ("est", LogisticRegression(max_iter=1000))]).fit(xh, yh)
    fixtures["v3-ct-hetero-logreg"] = (
        ct_logreg, hetero_features, hetero_kinds,
        {"features": hetero_features, "rows": hetero_rows},
    )
    ct_tree = Pipeline([("ct", ColumnTransformer(
        [("num", make_pipeline(SimpleImputer(), StandardScaler()), [0, 1]),
         ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                               OneHotEncoder(sparse_output=False)), [2])],
        remainder="drop")),
        ("est", DecisionTreeClassifier(max_depth=4, random_state=42))]).fit(xh, yh)
    fixtures["v3-ct-hetero-tree"] = (
        ct_tree, hetero_features, hetero_kinds,
        {"features": hetero_features, "rows": hetero_rows},
    )
    xh_pt = xh.copy()
    xh_pt[2, 0] = 35.0  # passthrough remainder column must be complete
    xh_pt[3, 1] = np.nan  # dropped column may hold missing values
    ct_pt = Pipeline([("ct", ColumnTransformer(
        [("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                               OneHotEncoder(sparse_output=False)), [2]),
         ("dropme", "drop", [1])],
        remainder="passthrough")),
        ("est", LogisticRegression(max_iter=1000))]).fit(xh_pt, yh)
    # The passthrough remainder column (age) must be complete in the batch.
    pt_rows = [[25.0, None, "lyon"], [None if False else 40.0, 70000.0, None], [52.0, 30000.0, "madrid"]]
    fixtures["v3-ct-passthrough-remainder"] = (
        ct_pt, hetero_features, hetero_kinds,
        {"features": hetero_features, "rows": pt_rows},
    )
    ct_scale = Pipeline([("ct", ColumnTransformer(
        [("num", make_pipeline(SimpleImputer(), StandardScaler()), [0, 1]),
         ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                               OneHotEncoder(sparse_output=False)), [2])],
        remainder="drop")),
        ("scale", StandardScaler()),
        ("est", LogisticRegression(max_iter=1000))]).fit(xh, yh)
    fixtures["v3-ct-then-scaler-logreg"] = (
        ct_scale, hetero_features, hetero_kinds,
        {"features": hetero_features, "rows": hetero_rows},
    )

    # #76: end-to-end heterogeneous pipelines feeding a RandomForestClassifier.
    ct_forest = Pipeline([("ct", ColumnTransformer(
        [("num", make_pipeline(SimpleImputer(), StandardScaler()), [0, 1]),
         ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                               OneHotEncoder(sparse_output=False)), [2])],
        remainder="drop")),
        ("est", RandomForestClassifier(n_estimators=8, max_depth=4, random_state=42))]).fit(xh, yh)
    fixtures["v3-ct-hetero-forest"] = (
        ct_forest, hetero_features, hetero_kinds,
        {"features": hetero_features, "rows": hetero_rows},
    )
    forest = make_pipeline(SimpleImputer(), StandardScaler(),
                           RandomForestClassifier(n_estimators=8, max_depth=4, random_state=42)).fit(xi, yi)
    fixtures["v3-forest-impute-scale"] = (
        forest, IRIS_FEATURES, None,
        {"features": IRIS_FEATURES, "rows": [[None, 3.0, 1.3, 0.2], [6.1, 3.0, 4.9, 1.8]]},
    )
    # A second forest (different seed) for shadow disagreement/delta checks.
    forest_b = make_pipeline(SimpleImputer(), StandardScaler(),
                             RandomForestClassifier(n_estimators=8, max_depth=3, random_state=7)).fit(xi, yi)
    fixtures["v3-forest-impute-scale-b"] = (
        forest_b, IRIS_FEATURES, None,
        {"features": IRIS_FEATURES, "rows": [[None, 3.0, 1.3, 0.2], [6.1, 3.0, 4.9, 1.8]]},
    )
    return fixtures


def _batch_matrix(batch):
    has_string = any(isinstance(v, str) for row in batch["rows"] for v in row)
    rows = [[np.nan if v is None else v for v in row] for row in batch["rows"]]
    if has_string:
        return np.array(rows, dtype=object)
    return np.array(rows, dtype=np.float64)


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
