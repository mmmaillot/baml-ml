"""Concrete inference trace demo: imputation -> scaling -> prediction (issue #11).

Run from the repository root:

    .venv/bin/python examples/trace_demo.py
    .venv/bin/python examples/trace_demo.py --out docs/benchmarks/inference_trace_demo.json

Fits the Iris classification pipeline on the deterministic seed-42 10%-NaN
split, then scores one *synthetic* flower with a missing sepal width
([5.1, NaN, 1.4, 0.2] — no real user or sensitive data) through:
  1. BAML staged inference with a `baml_py.Collector` attached
  2. scikit-learn instrumented with OpenTelemetry (`InMemorySpanExporter`)

Prints the stage-by-stage diagnostic view (missing feature detected and
imputed from training mean, standardized values, predicted probabilities) and
the exact captured Collector FunctionLogs and OTel spans.
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from baml_py import baml_py

from benchmarks.datasets import MISSING_FRAC, load_split
from benchmarks.reference import b, make_sklearn_pipeline, staged_fit
from benchmarks.sklearn_benchmark import _otel, _predict as sklearn_predict
from baml_client.tracing import flush as baml_flush

IRIS_FEATURES = ("sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm")
IRIS_CLASSES = ("setosa", "versicolor", "virginica")
SYNTHETIC_SAMPLE = [5.1, float("nan"), 1.4, 0.2]


def _ok(res):
    assert res.ok, res.error
    return res


def run(out_path=None, verbose=True):
    split = load_split("iris", missing_frac=MISSING_FRAC)
    sample = np.asarray([SYNTHETIC_SAMPLE], dtype=float)

    # Fit both pipelines on the exact same training split (outside traced inference).
    baml_state = staged_fit(split)
    sk_pipeline = make_sklearn_pipeline(split.task).fit(split.x_train, split.y_train)

    # 1) BAML staged inference with Collector attached once.
    collector = baml_py.Collector("trace-demo")
    client = b.with_options(collector=collector)
    raw_row = sample.tolist()

    t0 = time.perf_counter()
    imp_res = _ok(client.ImputerTransform(raw_row, baml_state["imputer"]))
    t1 = time.perf_counter()
    sc_res = _ok(client.ScalerTransform(imp_res.x, baml_state["scaler"]))
    t2 = time.perf_counter()
    prob_res = _ok(client.LogRegPredictProba(sc_res.x, baml_state["model"]))
    t3 = time.perf_counter()
    baml_flush()

    baml_logs = []
    for log in collector.logs:
        raw_str = str(log)
        baml_logs.append(
            {
                "id": log.id,
                "function_name": log.function_name,
                "duration_ms": log.timing.duration_ms,
                "serialized_bytes": len(raw_str.encode("utf-8")),
                "repr": raw_str,
            }
        )
    collector.clear()

    baml_probs = np.asarray(prob_res.probs[0], dtype=float)
    baml_pred_idx = int(np.argmax(baml_probs))

    # 2) scikit-learn inference with OpenTelemetry stage spans.
    provider, exporter, tracer = _otel()
    sk_probs = np.asarray(sklearn_predict(sk_pipeline, sample, tracer, proba=True)[0], dtype=float)
    provider.force_flush()
    otel_spans = []
    for span in exporter.get_finished_spans():
        payload = span.to_json()
        otel_spans.append(
            {
                "name": span.name,
                "duration_ms": round((span.end_time - span.start_time) / 1e6, 4),
                "attributes": dict(span.attributes or {}),
                "serialized_bytes": len(payload.encode("utf-8")),
            }
        )
    provider.shutdown()

    imputed_row = imp_res.x[0]
    scaled_row = sc_res.x[0]
    missing_idx = [i for i, v in enumerate(SYNTHETIC_SAMPLE) if np.isnan(v)]
    diagnostics = {
        "missing_features": [
            {
                "feature": IRIS_FEATURES[i],
                "input": None,
                "imputed_with_train_mean": round(float(imputed_row[i]), 6) + 0.0,
                "z_score_after_scaling": round(float(scaled_row[i]), 6) + 0.0,
            }
            for i in missing_idx
        ],
        "stages": [
            {
                "stage": "imputer.transform",
                "baml_function": "ImputerTransform",
                "wall_ms": round((t1 - t0) * 1e3, 3),
                "output_row": [round(float(v), 6) + 0.0 for v in imputed_row],
            },
            {
                "stage": "scaler.transform",
                "baml_function": "ScalerTransform",
                "wall_ms": round((t2 - t1) * 1e3, 3),
                "output_row": [round(float(v), 6) + 0.0 for v in scaled_row],
            },
            {
                "stage": "clf.predict_proba",
                "baml_function": "LogRegPredictProba",
                "wall_ms": round((t3 - t2) * 1e3, 3),
                "probabilities": {
                    cls: round(float(p), 6) for cls, p in zip(IRIS_CLASSES, baml_probs)
                },
                "predicted_class": IRIS_CLASSES[baml_pred_idx],
            },
        ],
    }

    doc = {
        "dataset": "iris",
        "training_missing_frac": split.missing_frac,
        "synthetic_input": {
            name: (None if np.isnan(val) else val)
            for name, val in zip(IRIS_FEATURES, SYNTHETIC_SAMPLE)
        },
        "parity": {
            "predicted_class_baml": IRIS_CLASSES[baml_pred_idx],
            "predicted_class_sklearn": IRIS_CLASSES[int(np.argmax(sk_probs))],
            "max_proba_abs_diff": float(np.max(np.abs(baml_probs - sk_probs))),
        },
        "diagnostics": diagnostics,
        "baml_collector_trace": {
            "spans": len(baml_logs),
            "total_bytes": sum(x["serialized_bytes"] for x in baml_logs),
            "logs": baml_logs,
        },
        "sklearn_otel_trace": {
            "spans": len(otel_spans),
            "total_bytes": sum(x["serialized_bytes"] for x in otel_spans),
            "spans_detail": otel_spans,
        },
    }

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2) + "\n")

    if verbose:
        print("=== Synthetic Iris Inference Trace Demo (1 missing feature) ===")
        print(f"Input: {doc['synthetic_input']}")
        for m in diagnostics["missing_features"]:
            print(
                f"Diagnostic: missing `{m['feature']}` imputed with training mean "
                f"{m['imputed_with_train_mean']} (scaled z={m['z_score_after_scaling']})"
            )
        print(
            f"Prediction: {doc['parity']['predicted_class_baml']} "
            f"(probs={diagnostics['stages'][2]['probabilities']}, "
            f"max |baml - sklearn| = {doc['parity']['max_proba_abs_diff']:.3e})"
        )
        print(
            f"\nBAML Collector ({doc['baml_collector_trace']['spans']} logs, "
            f"{doc['baml_collector_trace']['total_bytes']} bytes):"
        )
        for st, lg in zip(diagnostics["stages"], baml_logs):
            print(
                f"  - {lg['function_name']} (wall={st['wall_ms']} ms, "
                f"runtime duration_ms={lg['duration_ms']}, bytes={lg['serialized_bytes']})"
            )
        print(
            f"\nscikit-learn OpenTelemetry ({doc['sklearn_otel_trace']['spans']} spans, "
            f"{doc['sklearn_otel_trace']['total_bytes']} bytes):"
        )
        for sp in otel_spans:
            print(f"  - {sp['name']} (duration={sp['duration_ms']} ms, bytes={sp['serialized_bytes']})")

    return doc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="", help="Optional JSON output path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    return run(out_path=args.out or None, verbose=not args.quiet)


if __name__ == "__main__":
    main()
