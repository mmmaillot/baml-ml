"""A subprocess baseline using the fitted pipeline produced by run.py."""

import argparse
import io
import json
from pathlib import Path
from time import perf_counter_ns

import joblib
import numpy as np

# Load the modules joblib needs before timing model preparation.
import sklearn.impute
import sklearn.linear_model
import sklearn.pipeline
import sklearn.preprocessing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--calls", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--phases", action="store_true")
    args = parser.parse_args()

    start = perf_counter_ns()
    model_bytes = args.model.read_bytes()
    input_bytes = args.input.read_bytes()
    read_ns = perf_counter_ns() - start

    start = perf_counter_ns()
    model_id, features, model = joblib.load(io.BytesIO(model_bytes))
    prepare_ns = perf_counter_ns() - start

    start = perf_counter_ns()
    batch = json.loads(input_bytes)
    if batch["features"] != features:
        raise ValueError("feature order mismatch")
    rows = np.array(batch["rows"], dtype=np.float64)
    decode_ns = perf_counter_ns() - start

    for _ in range(args.warmups):
        model.predict_proba(rows)
    start = perf_counter_ns()
    for _ in range(args.calls):
        predictions = model.predict_proba(rows)
    inference_ns = perf_counter_ns() - start

    if args.phases:
        result = dict(read_ns=read_ns, prepare_ns=prepare_ns, decode_ns=decode_ns,
                      inference_ns=inference_ns)
    else:
        result = dict(schema_version=1, model_id=model_id, estimator_kind="logistic_multinomial",
                      operation="predict-proba", classes=model.classes_.tolist())
    print(json.dumps(dict(result, predictions=predictions.tolist()),
                     separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
