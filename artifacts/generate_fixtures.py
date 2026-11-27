"""Regenerate reference artifacts in a separate directory; sklearn 1.8.0."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from export_sklearn import write_artifact

def generate(out):
    out.mkdir(parents=True, exist_ok=True)
    for name in ("iris", "diabetes", "binary"):
        d = load_diabetes() if name == "diabetes" else load_iris()
        x, y = d.data.copy(), d.target.copy()
        if name == "binary":
            x, y = x[y < 2], y[y < 2]
        train, test, yt, _ = train_test_split(x, y, test_size=0.2, random_state=42,
                                             stratify=None if name == "diabetes" else y)
        for missing in (False, True):
            xt, xs = train.copy(), test[:2].copy()
            if missing:
                xt[0, 0] = np.nan
                xs[0, 0] = np.nan
            est = LinearRegression() if name == "diabetes" else LogisticRegression(max_iter=10000, tol=1e-10)
            p = make_pipeline(SimpleImputer(), StandardScaler(), est).fit(xt, yt)
            stem = name if missing else name + "_clean"
            write_artifact(p, d.feature_names, out / (stem + ".json"))
            imputed = p[0].transform(xs)
            scaled = p[1].transform(imputed)
            reference = dict(input=[[None if np.isnan(v) else float(v) for v in row] for row in xs],
                             imputed=imputed.tolist(), scaled=scaled.tolist(),
                             prediction=p.predict(xs).tolist(),
                             probs=None if name == "diabetes" else p.predict_proba(xs).tolist())
            (out / (stem + "_reference.json")).write_text(json.dumps(reference, indent=2, allow_nan=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    generate(parser.parse_args().out)
