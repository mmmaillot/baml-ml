"""Small exporter contract check; not a full numerical campaign."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from export_sklearn import export_pipeline, export_pipeline_v3

folder = Path(sys.argv[1])
frozen = Path(__file__).parent / "fixtures"
reference = json.loads((frozen / "reference.json").read_text())
for name in ("iris", "diabetes", "binary"):
    actual = json.loads((folder / (name + ".json")).read_text())
    expected = json.loads((frozen / (name + ".json")).read_text())
    identity = actual.pop("model_id")
    assert identity == hashlib.sha256(json.dumps(actual, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    expected.pop("model_id")
    for key in ("imputer_means", "scaler_means", "scaler_scales", "weights", "intercepts"):
        np.testing.assert_allclose(actual.pop(key), expected.pop(key), rtol=1e-5, atol=1e-7)
    assert actual == expected
    obs = json.loads((folder / (name + "_reference.json")).read_text())
    for key in ("imputed", "scaled", "prediction", "probs"):
        if reference[name][key] is not None:
            np.testing.assert_allclose(obs[key], reference[name][key], rtol=1e-5, atol=1e-7)
    clean = json.loads((folder / (name + "_clean.json")).read_text())
    assert clean["estimator_kind"] == actual["estimator_kind"]
    assert clean["model_id"] != identity

x = np.array([[1.0,2.0],[3.0,4.0],[5.0,7.0]])
y = np.array([1.0,2.0,3.0])
for model in (make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LinearRegression()),
              make_pipeline(SimpleImputer(),StandardScaler(with_std=False),LinearRegression()),
              make_pipeline(SimpleImputer(),StandardScaler(),LinearRegression())):
    model.fit(x,y)
    try:
        export_pipeline(model, ["a","a"] if model[0].strategy=="mean" and model[1].with_std else ["a","b"])
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported pipeline exported")

# Schema v3 contract: arbitrary-length pipelines (0..N preprocessing steps),
# dimension-chain validation and strict rejection of unsupported steps.
yc = np.array([0, 1, 0])
v3_cases = [
    (Pipeline([("est", LinearRegression())]).fit(x, y), ["a", "b"], 0),
    (make_pipeline(StandardScaler(), LogisticRegression(max_iter=200)).fit(x, yc), ["a", "b"], 1),
    (make_pipeline(SimpleImputer(), StandardScaler(), DecisionTreeClassifier(random_state=0)).fit(x, yc), ["a", "b"], 2),
    (make_pipeline(SimpleImputer(strategy="median"), LinearRegression()).fit(np.nan_to_num(x), y), ["a", "b"], 1),
]
for model, feats, n_steps in v3_cases:
    a = export_pipeline_v3(model, feats)
    assert a["schema_version"] == 3 and len(a["steps"]) == n_steps
    width = len(feats)
    for i, step in enumerate(a["steps"]):
        assert step["n_features_in"] == width
        width = step["n_features_out"]
    assert a["estimator"]["n_features_in"] == width
    assert a["input_kinds"] == ["numeric", "numeric"]
    assert a["model_id"] == hashlib.sha256(json.dumps({k: v for k, v in a.items() if k != "model_id"},
                                                       sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer, OneHotEncoder
from sklearn.svm import SVC
for bad in (make_pipeline(MinMaxScaler(), LinearRegression()),
            make_pipeline(SimpleImputer(), KBinsDiscretizer(), LinearRegression()),
            make_pipeline(SimpleImputer(), StandardScaler(), MinMaxScaler(), LinearRegression()),
            make_pipeline(SimpleImputer(), StandardScaler(), SVC())):
    bad.fit(x, yc)
    try:
        export_pipeline_v3(bad, ["a", "b"])
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported v3 pipeline exported")

# #74: extended imputer strategies and dense OneHotEncoder contract.
xcat = np.array([["red", "S"], ["green", "M"], ["blue", "L"], ["red", "M"],
                 ["green", "L"], ["blue", "S"]], dtype=object)
ycat = np.array([0, 1, 0, 1, 1, 0])
for strategy, fill in (("median", None), ("most_frequent", None), ("constant", -1.0)):
    model = make_pipeline(SimpleImputer(strategy=strategy, **({"fill_value": fill} if fill is not None else {})),
                          StandardScaler(), LogisticRegression(max_iter=300))
    model.fit(np.nan_to_num(x), yc)
    a = export_pipeline_v3(model, ["a", "b"])
    assert a["steps"][0]["imputer"]["strategy"] == strategy and a["steps"][0]["imputer"]["value_kind"] == "numeric"
strimp = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(sparse_output=False),
                       LogisticRegression(max_iter=300)).fit(xcat, ycat)
a = export_pipeline_v3(strimp, ["c", "s"])
assert a["input_kinds"] == ["categorical", "categorical"]
assert a["steps"][0]["imputer"]["value_kind"] == "string"
assert a["steps"][1]["kind"] == "one_hot_encoder" and a["steps"][1]["n_features_out"] == 6
assert a["estimator"]["n_features_in"] == 6
for drop in (None, "first", "if_binary"):
    a = export_pipeline_v3(make_pipeline(OneHotEncoder(drop=drop, sparse_output=False),
                                          LogisticRegression(max_iter=300)).fit(xcat, ycat), ["c", "s"])
    expected = 4 if drop == "first" else 6  # first drops one category per column; no binary column here
    assert a["steps"][0]["n_features_out"] == expected
for kwargs in (dict(sparse_output=True),  # sparse-only configuration
               dict(sparse_output=False, min_frequency=2),
               dict(sparse_output=False, max_categories=1),
               dict(sparse_output=False, drop="last"),
               dict(sparse_output=False, handle_unknown="infrequent_if_exist")):
    try:
        model = make_pipeline(OneHotEncoder(**kwargs), LogisticRegression(max_iter=300)).fit(xcat, ycat)
        export_pipeline_v3(model, ["c", "s"])
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError("unsupported one-hot encoder exported")
# One-hot on numeric input columns is rejected.
try:
    export_pipeline_v3(make_pipeline(OneHotEncoder(sparse_output=False),
                                     LogisticRegression(max_iter=300)).fit(x, yc), ["a", "b"])
except ValueError:
    pass
else:
    raise AssertionError("one-hot on numeric input exported")

# #75: ColumnTransformer with heterogeneous branches, remainder resolution
# and strict rejection of unsupported configurations.
from sklearn.compose import ColumnTransformer
xh = np.array([[25.0, 50000.0, "paris"], [40.0, 70000.0, "lyon"], [52.0, 30000.0, "berlin"],
               [33.0, 61000.0, "madrid"], [61.0, 45000.0, "paris"], [28.0, 80000.0, "lyon"]], dtype=object)
yh = np.array([0, 1, 1, 0, 1, 1])
ct = ColumnTransformer(
    [("num", make_pipeline(SimpleImputer(), StandardScaler()), [0, 1]),
     ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(sparse_output=False)), [2])],
    remainder="drop")
ct_model = make_pipeline(ct, LogisticRegression(max_iter=1000)).fit(xh, yh)
a = export_pipeline_v3(ct_model, ["age", "income", "c"])
assert a["input_kinds"] == ["numeric", "numeric", "categorical"]
step = a["steps"][0]
assert step["kind"] == "column_transformer" and step["n_features_in"] == 3
branches = step["column_transformer"]["branches"]
assert [b["name"] for b in branches] == ["num", "cat"]
assert branches[0]["columns"] == [0, 1] and branches[1]["columns"] == [2]
assert branches[1]["steps"][-1]["kind"] == "one_hot_encoder"
assert step["n_features_out"] == a["estimator"]["n_features_in"] == 2 + 4
assert step["column_transformer"]["remainder"] == "drop" and step["column_transformer"]["remainder_columns"] == []
ct_pt = make_pipeline(ColumnTransformer([("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                                                OneHotEncoder(sparse_output=False)), [2])],
                                        remainder="passthrough"),
                      LogisticRegression(max_iter=1000)).fit(xh, yh)
a = export_pipeline_v3(ct_pt, ["age", "income", "c"])
assert a["steps"][0]["column_transformer"]["remainder_columns"] == [0, 1]
assert a["steps"][0]["n_features_out"] == 4 + 2

def rejects_ct(**kwargs):
    try:
        model = make_pipeline(ColumnTransformer(**kwargs), LogisticRegression(max_iter=1000)).fit(xh, yh)
        export_pipeline_v3(model, ["age", "income", "c"])
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError("unsupported column transformer exported")

rejects_ct(transformers=[("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                                OneHotEncoder(sparse_output=True)), [2])],
           remainder="drop", sparse_threshold=1.0)  # sparse output
rejects_ct(transformers=[("n", make_pipeline(SimpleImputer(), StandardScaler()), [0, 1]),
                        ("c", ColumnTransformer([("i", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                                                     OneHotEncoder(sparse_output=False)), [0])]), [2])],
           remainder="drop")  # nested CT
rejects_ct(transformers=[("n", make_pipeline(SimpleImputer(), StandardScaler()), lambda X: [0, 1]),
                        ("c", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                             OneHotEncoder(sparse_output=False)), [2])], remainder="drop")  # callable selector
rejects_ct(transformers=[("n", make_pipeline(SimpleImputer(), StandardScaler()), [0, 1]),
                        ("c", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                             OneHotEncoder(sparse_output=False)), [2])],
           remainder="drop", transformer_weights={"n": 2.0})  # transformer_weights

# #76: RandomForestClassifier as a v3 terminal estimator; strict rejection of
# unsupported forest variants.
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
yc3 = np.array([0, 1, 2])
forest = make_pipeline(SimpleImputer(), StandardScaler(),
                       RandomForestClassifier(n_estimators=4, max_depth=2, random_state=42)).fit(x, yc3)
a = export_pipeline_v3(forest, ["a", "b"])
assert a["estimator"]["kind"] == "random_forest_classifier"
assert a["estimator"]["classes"] == [0, 1, 2] and len(a["estimator"]["trees"]) == 4
for tree in a["estimator"]["trees"]:
    assert tree["n_features"] == 2 and tree["classes"] == [0, 1, 2]
    for node in tree["nodes"]:
        assert (node["probabilities"] != []) == (node["left"] == -1)
ct_forest = make_pipeline(ColumnTransformer(
    [("num", make_pipeline(SimpleImputer(), StandardScaler()), [0, 1]),
     ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(sparse_output=False)), [2])],
    remainder="drop"),
    RandomForestClassifier(n_estimators=3, max_depth=2, random_state=42)).fit(xh, yh)
a = export_pipeline_v3(ct_forest, ["age", "income", "c"])
assert a["estimator"]["kind"] == "random_forest_classifier"
assert a["estimator"]["n_features_in"] == a["steps"][0]["n_features_out"] == 6
multi = make_pipeline(SimpleImputer(), StandardScaler(),
                      RandomForestClassifier(n_estimators=2, random_state=42)).fit(x, np.column_stack([yc3, yc3]))
regressor = make_pipeline(SimpleImputer(), StandardScaler(),
                          RandomForestRegressor(n_estimators=2, random_state=42)).fit(x, y.astype(float))
for model in (multi, regressor):
    try:
        export_pipeline_v3(model, ["a", "b"])
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported forest exported")

# Review regressions: output-name propagation across chained ColumnTransformers,
# explicit passthrough branches, remainder complement and zero-width one-hot blocks.
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from export_sklearn import _resolve_columns

# Named selectors require known output names and resolve against them.
for cols, names in ((["age"], None), (["nope"], ["age", "income", "city"])):
    try:
        _resolve_columns(cols, 6, names)
    except ValueError:
        pass
    else:
        raise AssertionError("unresolvable named selector accepted")

df = pd.DataFrame({"age": xh[:, 0].astype(float), "income": xh[:, 1].astype(float), "city": xh[:, 2]})
ct1 = ColumnTransformer(
    [("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(sparse_output=False)), ["city"])],
    remainder="passthrough", verbose_feature_names_out=False)
ct2 = ColumnTransformer(
    [("scaled_age", make_pipeline(SimpleImputer(), StandardScaler()), ["age"])], remainder="drop")
chained = Pipeline([("ct1", ct1), ("ct2", ct2), ("est", LogisticRegression(max_iter=1000))])
chained.set_output(transform="pandas")
chained.fit(df, yh)
a = export_pipeline_v3(chained, ["age", "income", "city"])
ct1_out = a["steps"][0]["column_transformer"]
assert ct1_out["remainder"] == "passthrough" and ct1_out["remainder_columns"] == [0, 1]
ct2_out = a["steps"][1]["column_transformer"]
# "age" resolves against ct1's actual output names, landing at index 4.
assert ct2_out["branches"][0]["columns"] == [4], ct2_out["branches"][0]["columns"]
assert ct2_out["remainder_columns"] == [0, 1, 2, 3, 5]
assert a["steps"][1]["n_features_out"] == a["estimator"]["n_features_in"] == 1

# Explicit "passthrough" exports as passthrough even though sklearn fits it as
# an identity FunctionTransformer; arbitrary FunctionTransformer branches are rejected.
ct_ept = make_pipeline(ColumnTransformer(
    [("keep", "passthrough", [0]),
     ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(sparse_output=False)), [2])],
    remainder="drop"), LogisticRegression(max_iter=1000)).fit(xh, yh)
a = export_pipeline_v3(ct_ept, ["age", "income", "c"])
branches = a["steps"][0]["column_transformer"]["branches"]
assert branches[0]["name"] == "keep" and branches[0]["mode"] == "passthrough" and branches[0]["columns"] == [0]
fn_ct = make_pipeline(ColumnTransformer(
    [("f", FunctionTransformer(), [0, 1]),
     ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(sparse_output=False)), [2])],
    remainder="drop"), LogisticRegression(max_iter=1000)).fit(xh, yh)
try:
    export_pipeline_v3(fn_ct, ["age", "income", "c"])
except ValueError:
    pass
else:
    raise AssertionError("FunctionTransformer branch exported")

# drop="first" on a singleton category is a valid zero-width block; an
# all-singleton encoder has zero total width and is rejected at export.
xs2 = np.array([["a", "x"], ["b", "x"], ["a", "x"], ["b", "x"], ["a", "x"], ["b", "x"]], dtype=object)
ys2 = np.array([0, 1, 0, 1, 0, 1])
a = export_pipeline_v3(make_pipeline(OneHotEncoder(drop="first", sparse_output=False),
                                      LogisticRegression(max_iter=300)).fit(xs2, ys2), ["c0", "c1"])
assert a["steps"][0]["n_features_out"] == a["estimator"]["n_features_in"] == 1
xs_all = np.array([["x", "y"], ["x", "y"], ["x", "y"], ["x", "y"], ["x", "y"], ["x", "y"]], dtype=object)
try:
    export_pipeline_v3(make_pipeline(OneHotEncoder(drop="first", sparse_output=False),
                                     LogisticRegression(max_iter=300)).fit(xs_all, ys2), ["c0", "c1"])
except ValueError:
    pass
else:
    raise AssertionError("zero-width one-hot encoder exported")
print("Exporter contract and regenerated reference fixtures passed.")
