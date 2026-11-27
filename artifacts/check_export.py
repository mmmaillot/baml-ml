"""Small exporter contract check; not a full numerical campaign."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from export_sklearn import export_pipeline

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

# Random forest export contract: v4 payload, one validated tree per estimator,
# and strict rejection of unsupported variants.
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
yr = np.array([-3,2,7])
forest = make_pipeline(SimpleImputer(),StandardScaler(),
                       RandomForestClassifier(n_estimators=4,max_depth=2,random_state=42)).fit(x,yr)
a = export_pipeline(forest, ["a","b"])
assert a["schema_version"] == 4 and a["estimator_kind"] == "random_forest_classifier"
assert a["n_features"] == 2 and a["classes"] == [-3,2,7] and len(a["trees"]) == 4
assert a["model_id"] == hashlib.sha256(json.dumps({k: v for k, v in a.items() if k != "model_id"},
                                                 sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
for tree in a["trees"]:
    assert tree["n_features"] == 2 and tree["classes"] == a["classes"]
    for node in tree["nodes"]:
        assert (node["probabilities"] != []) == (node["left"] == -1)
multi = make_pipeline(SimpleImputer(),StandardScaler(),
                      RandomForestClassifier(n_estimators=2,random_state=42)).fit(x, np.column_stack([yr, yr]))
regressor = make_pipeline(SimpleImputer(),StandardScaler(),
                          RandomForestRegressor(n_estimators=2,random_state=42)).fit(x, yr.astype(float))
for model in (multi, regressor):
    try:
        export_pipeline(model, ["a","b"])
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported forest exported")
print("Exporter contract and regenerated reference fixtures passed.")
