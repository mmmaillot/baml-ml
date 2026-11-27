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

from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
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
print("Exporter contract and regenerated reference fixtures passed.")
