"""scikit-learn reference benchmark runner: vanilla vs OpenTelemetry (issue #9).

Run from the repository root on a clean checkout:

    .venv/bin/python benchmarks/sklearn_benchmark.py --out benchmarks/results/sklearn_benchmark.json

Variants: "sklearn" (uninstrumented) and "sklearn-otel" (OpenTelemetry spans
around fit/predict and around each named pipeline stage). Datasets, the fixed
seed-42 splits and the reference pipelines are reused from issue #8
(benchmarks/datasets.py, benchmarks/reference.py); dataset loading is excluded
from every timed region.

Measurement protocol: time.perf_counter (monotonic), warm-up runs excluded
from statistics, repetitions recorded per measurement, OTel force_flush called
outside the timed regions. Thread counts are pinned to 1 through environment
variables (setdefault) and recorded, together with threadpoolctl info.

Telemetry configuration (also recorded in the output under "telemetry"):
granularity = one span per operation plus one span per named pipeline stage,
payload policy = default span attributes (no custom event payloads), exporter =
in-memory span exporter (SimpleSpanProcessor); trace volume is estimated by
serializing each finished span with Span.to_json() and summing bytes.

Results schema (schema_version 1, reusable by issue #10): a JSON object with
config, telemetry, environment, parity and results; each result record carries
dataset, variant, operation, batch_size, repetitions, timing_s (seconds:
p50/p99/mean/min/max), throughput_rows_per_s and trace {spans, bytes}.
Measurements that do not apply are null, never fabricated.
"""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Pin BLAS/OpenMP thread counts before numpy/sklearn are imported; recorded in
# the output so runs are comparable and documented.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from threadpoolctl import threadpool_info

from benchmarks.datasets import DATASETS, MISSING_FRAC, SEED, TEST_SIZE, load_split
from benchmarks.reference import make_sklearn_pipeline

SCHEMA_VERSION = 1
VARIANTS = ("sklearn", "sklearn-otel")
DEFAULT_OUT = REPO_ROOT / "benchmarks" / "results" / "sklearn_benchmark.json"


def _otel():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter, provider.get_tracer("baml-ml-bench")


def _fit(pipeline, x, y, tracer):
    if tracer is None:
        return pipeline.fit(x, y)
    with tracer.start_as_current_span(
        "pipeline.fit",
        attributes={"operation": "fit", "n_rows": int(x.shape[0]), "n_features": int(x.shape[1])},
    ):
        for name, step in pipeline.steps[:-1]:
            with tracer.start_as_current_span(
                f"pipeline.stage.{name}.fit_transform", attributes={"stage": name, "step": type(step).__name__}
            ):
                x = step.fit_transform(x, y)
        final_name, final = pipeline.steps[-1]
        with tracer.start_as_current_span(
            f"pipeline.stage.{final_name}.fit", attributes={"stage": final_name, "step": type(final).__name__}
        ):
            final.fit(x, y)
        return pipeline


def _predict(pipeline, x, tracer, proba=False):
    if tracer is None:
        return pipeline.predict_proba(x) if proba else pipeline.predict(x)
    with tracer.start_as_current_span(
        "pipeline.predict_proba" if proba else "pipeline.predict",
        attributes={"operation": "predict_proba" if proba else "predict", "batch_size": int(x.shape[0])},
    ):
        for name, step in pipeline.steps[:-1]:
            with tracer.start_as_current_span(
                f"pipeline.stage.{name}.transform", attributes={"stage": name, "step": type(step).__name__}
            ):
                x = step.transform(x)
        final_name, final = pipeline.steps[-1]
        op = "predict_proba" if proba else "predict"
        with tracer.start_as_current_span(
            f"pipeline.stage.{final_name}.{op}", attributes={"stage": final_name, "step": type(final).__name__}
        ):
            return final.predict_proba(x) if proba else final.predict(x)


def _trace_delta(exporter, span_names):
    if exporter is None:
        return {"spans": 0, "bytes": 0}
    spans = exporter.get_finished_spans()
    span_names.update(s.name for s in spans)
    total = sum(len(s.to_json().encode("utf-8")) for s in spans)
    exporter.clear()
    return {"spans": len(spans), "bytes": total}


def _timing_record(times_s, n_rows):
    arr = np.asarray(times_s, dtype=float)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }, (float(n_rows / arr.mean()) if n_rows else None)


def _measure(variant, split, repetitions, warmups):
    """Measure fit, batch predict and single-row predict for one variant."""
    provider = exporter = tracer = None
    span_names = set()
    if variant == "sklearn-otel":
        provider, exporter, tracer = _otel()

    records = []

    def emit(operation, batch_size, times_s, trace):
        timing_s, throughput = _timing_record(times_s, batch_size if operation != "fit" else None)
        records.append(
            {
                "dataset": split.name,
                "task": split.task,
                "missing_frac": split.missing_frac,
                "variant": variant,
                "operation": operation,
                "batch_size": batch_size,
                "repetitions": repetitions,
                "warmups": warmups,
                "timing_s": timing_s,
                "throughput_rows_per_s": throughput,
                "trace": trace,
            }
        )

    # fit
    for _ in range(warmups):
        _fit(make_sklearn_pipeline(split.task), split.x_train, split.y_train, tracer)
    fit_times = []
    model = None
    for _ in range(repetitions):
        t0 = time.perf_counter()
        model = _fit(make_sklearn_pipeline(split.task), split.x_train, split.y_train, tracer)
        fit_times.append(time.perf_counter() - t0)
    if provider is not None:
        provider.force_flush()  # outside the timed region
    emit("fit", None, fit_times, _trace_delta(exporter, span_names))

    proba = split.task == "classification"

    # batch predict on the full held-out split
    for _ in range(warmups):
        _predict(model, split.x_test, tracer, proba)
    batch_times = []
    last_pred = None
    for _ in range(repetitions):
        t0 = time.perf_counter()
        last_pred = _predict(model, split.x_test, tracer, proba)
        batch_times.append(time.perf_counter() - t0)
    if provider is not None:
        provider.force_flush()
    emit("predict", int(split.x_test.shape[0]), batch_times, _trace_delta(exporter, span_names))

    # single-row predict (batch size 1), one timed call per test row per rep
    single_times = []
    for _ in range(repetitions):
        for i in range(split.x_test.shape[0]):
            t0 = time.perf_counter()
            _predict(model, split.x_test[i : i + 1], tracer, proba)
            single_times.append(time.perf_counter() - t0)
    if provider is not None:
        provider.force_flush()
    emit("predict", 1, single_times, _trace_delta(exporter, span_names))

    if provider is not None:
        provider.shutdown()

    return records, last_pred, sorted(span_names)


def _packages():
    from importlib.metadata import version

    names = ("numpy", "scikit-learn", "opentelemetry-api", "opentelemetry-sdk", "baml-py")
    out = {}
    for n in names:
        try:
            out[n] = version(n)
        except Exception:
            out[n] = None
    return out


def run_benchmark(datasets, variants, repetitions, warmups, missing_frac, out_path):
    results = []
    parity = []
    trace_sample = set()
    for name in datasets:
        split = load_split(name, missing_frac=missing_frac)
        preds = {}
        for variant in variants:
            records, last_pred, span_names = _measure(variant, split, repetitions, warmups)
            results.extend(records)
            trace_sample.update(span_names)
            preds[variant] = last_pred
        if len(preds) == 2:
            a, b_ = (preds[v] for v in variants)
            parity.append(
                {
                    "dataset": name,
                    "missing_frac": split.missing_frac,
                    "variants": list(variants),
                    "predictions_match": bool(np.array_equal(np.asarray(a), np.asarray(b_))),
                    "max_abs_diff": float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b_, dtype=float)))),
                }
            )

    doc = {
        "schema_version": SCHEMA_VERSION,
        "runner": "benchmarks/sklearn_benchmark.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "seed": SEED,
            "test_size": TEST_SIZE,
            "missing_frac": missing_frac,
            "datasets": list(datasets),
            "variants": list(variants),
            "repetitions": repetitions,
            "warmups": warmups,
            "timer": "time.perf_counter",
            "timer_unit": "s",
            "thread_env": {
                k: os.environ.get(k)
                for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
        },
        "telemetry": {
            "variant": "sklearn-otel",
            "granularity": "one span per operation plus one span per named pipeline stage",
            "payload_policy": "default span attributes; no custom event payloads",
            "exporter": "in-memory span exporter via SimpleSpanProcessor; force_flush outside timed regions",
            "volume_serialization": "Span.to_json() UTF-8 bytes per finished span",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "threadpool": [
                {"user_api": t["user_api"], "num_threads": t["num_threads"]} for t in threadpool_info()
            ],
            "packages": _packages(),
        },
        "trace_sample": sorted(trace_sample),
        "parity": parity,
        "results": results,
    }
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2) + "\n")
    return doc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--missing-frac", type=float, default=0.0, help="fixed-seed NaN fraction (default: clean)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="JSON output path (empty string disables writing)")
    args = parser.parse_args(argv)

    if args.repetitions < 1 or args.warmups < 0:
        parser.error("--repetitions must be >= 1 and --warmups >= 0")

    out_path = args.out if args.out else None
    doc = run_benchmark(args.datasets, args.variants, args.repetitions, args.warmups, args.missing_frac, out_path)
    n = len(doc["results"])
    print(f"wrote {n} measurement records for datasets={args.datasets} variants={args.variants} to {out_path}")
    for p in doc["parity"]:
        print(f"parity[{p['dataset']}]: predictions_match={p['predictions_match']} max_abs_diff={p['max_abs_diff']:.3e}")
    return doc


if __name__ == "__main__":
    main()
