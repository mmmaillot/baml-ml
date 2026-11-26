"""Reference pipelines and BAML wrappers shared by the examples (#8) and the
benchmarks (#9).

Python is used only for orchestration, dataset preparation, reference models
and evaluation; every numerical operation inside the BAML pipelines runs in
BAML. The sklearn reference mirrors the BAML pipeline step for step:
SimpleImputer(mean) -> StandardScaler -> estimator.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _ensure_generated_client() -> None:
    baml_src = REPO_ROOT / "baml_src"
    client_dir = REPO_ROOT / "baml_client"
    init_py = client_dir / "__init__.py"
    inlined_py = client_dir / "inlinedbaml.py"
    up_to_date = False
    if init_py.exists() and inlined_py.exists():
        baml_files = list(baml_src.rglob("*.baml"))
        if not baml_files:
            up_to_date = True
        else:
            latest = max(baml_src.stat().st_mtime, *(p.stat().st_mtime for p in baml_files))
            up_to_date = inlined_py.stat().st_mtime >= latest
    if up_to_date:
        return
    local_bin = REPO_ROOT / "node_modules" / ".bin"
    cli = (
        shutil.which("baml", path=str(local_bin))
        or shutil.which("baml-cli", path=str(local_bin))
        or shutil.which("baml")
        or shutil.which("baml-cli")
    )
    if cli is None:
        if init_py.exists():
            return
        raise RuntimeError(
            "baml_client/ is missing and no baml CLI was found; run `npm install` "
            "then `npx baml --features beta generate` from the repository root."
        )
    subprocess.run([cli, "--features", "beta", "generate"], cwd=REPO_ROOT, check=True)


_ensure_generated_client()
from baml_client import b

# Classifier hyperparameters shared by the BAML pipeline and the sklearn
# reference (C=1.0 both sides; sklearn gets a tight tol so the reference is a
# converged optimum, as in the issue #6 parity test).
BAML_C = 1.0
BAML_MAX_ITER = 5000
BAML_TOL = 0.00000001
SKLEARN_TOL = 0.0000000001  # 1e-10
SKLEARN_MAX_ITER = 10000


def make_sklearn_pipeline(task):
    steps = [
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ]
    if task == "classification":
        steps.append(
            ("clf", LogisticRegression(C=BAML_C, tol=SKLEARN_TOL, max_iter=SKLEARN_MAX_ITER))
        )
    elif task == "regression":
        steps.append(("reg", LinearRegression()))
    else:
        raise ValueError(f"unknown task: {task}")
    return Pipeline(steps)


def fit_baml_pipeline(split):
    """Fit the BAML pipeline (issue #7) on the training rows only."""
    if split.task == "classification":
        res = b.ClassificationPipelineFit(
            split.x_train.tolist(),
            [int(v) for v in split.y_train],
            BAML_C,
            BAML_MAX_ITER,
            BAML_TOL,
        )
        assert res.ok, res.error
        return res.pipeline
    res = b.RegressionPipelineFit(split.x_train.tolist(), [float(v) for v in split.y_train])
    assert res.ok, res.error
    return res.pipeline


def predict_baml(split, pipeline, x):
    """Return (predictions, probabilities); probabilities are None for regression."""
    x = np.asarray(x, dtype=float)
    if split.task == "classification":
        proba = b.ClassificationPipelinePredictProba(x.tolist(), pipeline)
        assert proba.ok, proba.error
        pred = b.ClassificationPipelinePredict(x.tolist(), pipeline)
        assert pred.ok, pred.error
        return np.asarray(pred.value), np.asarray(proba.probs)
    res = b.RegressionPipelinePredict(x.tolist(), pipeline)
    assert res.ok, res.error
    return np.asarray(res.value), None


def staged_fit(split, call):
    """Fit the pipeline stage by stage through call(stage, function_name, *args).

    Stage names follow the #9 span convention (pipeline.stage.<step>.<op>) so
    telemetry can name each step. Numerics are identical to the monolithic
    pipeline entry points (asserted in tests): the same BAML functions are
    called in the same order on the same training data only.
    """
    x = split.x_train.tolist()
    if split.task == "classification":
        y = [int(v) for v in split.y_train]
    else:
        y = [float(v) for v in split.y_train]
    r = call("pipeline.stage.imputer.fit", "ImputerFit", x)
    assert r.ok, r.error
    imp = r.imputer
    r = call("pipeline.stage.imputer.transform", "ImputerTransform", x, imp)
    assert r.ok, r.error
    x1 = r.x
    r = call("pipeline.stage.scaler.fit", "ScalerFit", x1)
    assert r.ok, r.error
    sc = r.scaler
    r = call("pipeline.stage.scaler.transform", "ScalerTransform", x1, sc)
    assert r.ok, r.error
    x2 = r.x
    if split.task == "classification":
        r = call("pipeline.stage.clf.fit", "LogRegFit", x2, y, BAML_C, BAML_MAX_ITER, BAML_TOL)
    else:
        r = call("pipeline.stage.reg.fit", "LinearRegressionFit", x2, y)
    assert r.ok, r.error
    return {"imputer": imp, "scaler": sc, "model": r.model}


def staged_predict(split, state, x, call):
    """Staged inference (impute -> scale -> model predict) through call(stage, function_name, *args).

    Returns (predictions, probabilities); probabilities are None for regression.
    """
    xl = np.asarray(x, dtype=float).tolist()
    r = call("pipeline.stage.imputer.transform", "ImputerTransform", xl, state["imputer"])
    assert r.ok, r.error
    r = call("pipeline.stage.scaler.transform", "ScalerTransform", r.x, state["scaler"])
    assert r.ok, r.error
    x2 = r.x
    if split.task == "classification":
        rp = call("pipeline.predict_proba", "LogRegPredictProba", x2, state["model"])
        rd = call("pipeline.predict", "LogRegPredict", x2, state["model"])
        assert rp.ok, rp.error
        assert rd.ok, rd.error
        return np.asarray(rd.value), np.asarray(rp.probs)
    r = call("pipeline.predict", "LinearRegressionPredict", x2, state["model"])
    assert r.ok, r.error
    return np.asarray(r.value), None
