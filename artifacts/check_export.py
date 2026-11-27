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
print("Exporter contract and regenerated reference fixtures passed.")
