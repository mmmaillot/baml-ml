"""BAML benchmark runner: telemetry off vs telemetry on (issue #10).

Run from the repository root on a clean checkout:

    .venv/bin/python benchmarks/baml_benchmark.py --out benchmarks/results/baml_benchmark.json

Variants: "baml-telemetry-off" (plain BAML calls, no collector attached) and
"baml-telemetry-on" (a baml_py Collector is attached to every call, so each
named pipeline stage produces a retained FunctionLog). Both variants execute
the *identical* staged operations (fit / batch predict / single-row predict /
bridge probe) on the *identical* seed-42 splits from issue #8, so the only
difference is telemetry. Dataset loading is excluded from every timed region.
No LLM is ever called and Python performs no numerics.

Telemetry facts verified against the pinned runtime (baml-py 0.226.3), see
"limitations" in the output JSON:
- Expression-function spans are started but never finalized/exported by the
  Python runtime (runtime.drain_stats() shows finalized=sent=0 even after
  flush); BAML_TRACE_FILE / BAML_TRACER have no observable effect there, and no
  OTLP exporter is embedded. The Boundary cloud publisher stays inactive
  without an API key ("No publish channel found"; no network egress, no
  sensitive data leaves the process).
- Therefore the observable telemetry for non-LLM functions is the per-call
  Collector function log (function name, timing, usage). "telemetry-on"
  retains them and we measure their count and serialized bytes (str(log));
  "telemetry-off" retains nothing (0 logs, 0 bytes).
- FunctionLog.timing.duration_ms is None for expression functions, so
  runtime-only timings are reported as null (never fabricated); bridge
  overhead is approximated by a dedicated "bridge_probe" operation (a trivial
  BAML call) and end-to-end Python-to-BAML latency is the wall-clock
  time.perf_counter measurement.
- Environment variables are read lazily per call in this runtime, so a single
  process is enough; no separate processes are needed (we do not rely on
  init-time env switches).

Results use the shared schema from issue #9 (schema_version 1: config,
telemetry, environment, parity, results; timing in seconds via timing_s,
throughput_rows_per_s, non-applicable measurements null), extended with
runtime_only_s and a richer trace object.
"""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Pin BLAS/OpenMP thread counts before numpy/sklearn are imported (same
# protocol as the #9 runner); recorded in the output.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from threadpoolctl import threadpool_info

from baml_client import b
from baml_client.globals import (
    DO_NOT_USE_DIRECTLY_UNLESS_YOU_KNOW_WHAT_YOURE_DOING_RUNTIME as BAML_RUNTIME,
)
from baml_py import baml_py

from benchmarks.datasets import DATASETS, MISSING_FRAC, SEED, TEST_SIZE, load_split
from benchmarks.reference import make_sklearn_pipeline, staged_fit, staged_predict

SCHEMA_VERSION = 1
VARIANTS = ("baml-telemetry-off", "baml-telemetry-on")
TELEMETRY_ON = "baml-telemetry-on"
DEFAULT_OUT = REPO_ROOT / "benchmarks" / "results" / "baml_benchmark.json"

# sklearn parity tolerances from issue #8 (justified there by solver diffs).
SKLEARN_TOL = {"classification": 1e-3, "regression": 1e-9}

LIMITATIONS = [
    "baml-py 0.226.3 starts a span per expression-function call but never finalizes or exports it "
    "(runtime.drain_stats(): finalized=sent=0 even after flush), so no runtime-internal trace "
    "payload is observable from Python.",
    "BAML_TRACE_FILE and BAML_TRACER have no observable effect in the Python runtime of 0.226.3 "
    "(no trace file is written for expression functions); verified in a fresh process.",
    "No OTLP exporter is embedded in the pinned runtime, so OTel payload/granularity from #9 "
    "cannot be matched 1:1; per-stage observability is provided by Collector function logs named "
    "after the #9 stage convention (pipeline.stage.<step>.<op>).",
    "tracing.trace (BamlSpan from Python) cannot wrap a BAML call in 0.226.3 (Call ID mismatch), "
    "so stage spans are realized as collector logs, not Python-side spans.",
    "FunctionLog.timing.duration_ms is None for expression functions: runtime-only timings are "
    "null and bridge overhead is approximated by the bridge_probe operation.",
    "The Boundary cloud trace publisher is inactive without an API key (no upload channel, no "
    "network egress); only benchmark fixture data is ever logged.",
]


def make_caller(variant):
    """Return (call, collector): call(stage, function_name, *args) drives one named BAML stage."""
    if variant == TELEMETRY_ON:
        collector = baml_py.Collector("baml-bench")

        def call(stage, function_name, *args):
            client = b.with_options(collector=collector)
            return getattr(client, function_name)(*args)

        return call, collector

    def call(stage, function_name, *args):
        return getattr(b, function_name)(*args)

    return call, None


def _stats_delta():
    s = BAML_RUNTIME.drain_stats()
    return {
        "runtime_spans_started": s.started,
        "runtime_spans_exported": s.finalized + s.sent + s.done,
        "runtime_spans_failed": s.failed,
    }


def _trace_delta(collector):
    if collector is None:
        return {"spans": 0, "bytes": 0, "function_log_names": []}
    logs = collector.logs
    names = sorted({log.function_name for log in logs})
    out = {
        "spans": len(logs),
        "bytes": sum(len(str(log).encode("utf-8")) for log in logs),
        "function_log_names": names,
    }
    collector.clear()
    return out


def _timing_record(times_s, n_rows):
    arr = np.asarray(times_s, dtype=float)
    timing = {
        "p50": float(np.percentile(arr, 50)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    return timing, (float(n_rows / arr.mean()) if n_rows else None)


def _measure(variant, split, repetitions, fit_repetitions, warmups):
    call, collector = make_caller(variant)
    records = []

    def emit(operation, batch_size, times_s, trace, stats):
        timing_s, throughput = _timing_record(times_s, None if operation in ("fit", "bridge_probe") else batch_size)
        records.append(
            {
                "dataset": split.name,
                "task": split.task,
                "missing_frac": split.missing_frac,
                "variant": variant,
                "operation": operation,
                "batch_size": batch_size,
                "repetitions": fit_repetitions if operation == "fit" else repetitions,
                "warmups": warmups,
                "timing_s": timing_s,
                "throughput_rows_per_s": throughput,
                "runtime_only_s": None,  # duration_ms is None for expr functions in baml-py 0.226.3
                "trace": {**trace, **stats},
            }
        )

    def discard_warmup_telemetry():
        BAML_RUNTIME.flush()
        _trace_delta(collector)
        _stats_delta()

    # bridge probe: trivial BAML call to approximate Python<->BAML bridge overhead
    for _ in range(warmups):
        call("bridge_probe", "IsFinite", 1.0)
    discard_warmup_telemetry()
    probe_times = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        call("bridge_probe", "IsFinite", 1.0)
        probe_times.append(time.perf_counter() - t0)
    BAML_RUNTIME.flush()  # outside the timed region
    emit("bridge_probe", None, probe_times, _trace_delta(collector), _stats_delta())

    # fit (staged, on training rows only)
    for _ in range(warmups):
        staged_fit(split, call)
    discard_warmup_telemetry()
    fit_times = []
    state = None
    for _ in range(fit_repetitions):
        t0 = time.perf_counter()
        state = staged_fit(split, call)
        fit_times.append(time.perf_counter() - t0)
    BAML_RUNTIME.flush()
    emit("fit", None, fit_times, _trace_delta(collector), _stats_delta())

    # batch predict on the full held-out split
    for _ in range(warmups):
        staged_predict(split, state, split.x_test, call)
    discard_warmup_telemetry()
    batch_times = []
    last = None
    for _ in range(repetitions):
        t0 = time.perf_counter()
        last = staged_predict(split, state, split.x_test, call)
        batch_times.append(time.perf_counter() - t0)
    BAML_RUNTIME.flush()
    emit("predict", int(split.x_test.shape[0]), batch_times, _trace_delta(collector), _stats_delta())

    # single-row predict (batch size 1)
    single_times = []
    for _ in range(repetitions):
        for i in range(split.x_test.shape[0]):
            t0 = time.perf_counter()
            staged_predict(split, state, split.x_test[i : i + 1], call)
            single_times.append(time.perf_counter() - t0)
    BAML_RUNTIME.flush()
    emit("predict", 1, single_times, _trace_delta(collector), _stats_delta())

    return records, last


def _packages():
    from importlib.metadata import version

    names = ("numpy", "scikit-learn", "baml-py", "opentelemetry-api", "opentelemetry-sdk")
    out = {}
    for n in names:
        try:
            out[n] = version(n)
        except Exception:
            out[n] = None
    return out


def run_benchmark(datasets, variants, repetitions, fit_repetitions, warmups, missing_frac, out_path):
    results = []
    parity = []
    for name in datasets:
        split = load_split(name, missing_frac=missing_frac)
        preds = {}
        for variant in variants:
            records, last = _measure(variant, split, repetitions, fit_repetitions, warmups)
            results.extend(records)
            preds[variant] = last[1] if split.task == "classification" else last[0]
        entry = {
            "dataset": name,
            "missing_frac": split.missing_frac,
            "variants": list(variants),
            "predictions_match": None,
            "max_abs_diff": None,
            "matches_sklearn_reference": None,
            "sklearn_max_abs_diff": None,
            "sklearn_tol": SKLEARN_TOL[split.task],
        }
        if len(preds) == 2:
            a, b_ = (preds[v] for v in variants)
            entry["predictions_match"] = bool(np.array_equal(np.asarray(a), np.asarray(b_)))
            entry["max_abs_diff"] = float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b_, dtype=float))))
        # #8 parity: staged BAML execution vs the sklearn reference on the same split
        plain_call = make_caller("baml-telemetry-off")[0]
        state = staged_fit(split, plain_call)
        pred, proba = staged_predict(split, state, split.x_test, plain_call)
        baml_out = proba if split.task == "classification" else pred
        ref = make_sklearn_pipeline(split.task).fit(split.x_train, split.y_train)
        ref_out = ref.predict_proba(split.x_test) if split.task == "classification" else ref.predict(split.x_test)
        diff = float(np.max(np.abs(np.asarray(baml_out, dtype=float) - np.asarray(ref_out, dtype=float))))
        entry["matches_sklearn_reference"] = bool(diff <= SKLEARN_TOL[split.task])
        entry["sklearn_max_abs_diff"] = diff
        parity.append(entry)

    doc = {
        "schema_version": SCHEMA_VERSION,
        "runner": "benchmarks/baml_benchmark.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "seed": SEED,
            "test_size": TEST_SIZE,
            "missing_frac": missing_frac,
            "datasets": list(datasets),
            "variants": list(variants),
            "repetitions": repetitions,
            "fit_repetitions": fit_repetitions,
            "warmups": warmups,
            "timer": "time.perf_counter",
            "timer_unit": "s",
            "thread_env": {
                k: os.environ.get(k)
                for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
        },
        "telemetry": {
            "variants": {
                "baml-telemetry-off": "plain calls, no collector, no retained logs",
                "baml-telemetry-on": "baml_py.Collector attached to every call; one FunctionLog per named stage; "
                "runtime flush outside timed regions",
            },
            "granularity": "one retained function log per named pipeline stage (pipeline.stage.<step>.<op>) plus "
            "pipeline.predict / pipeline.predict_proba",
            "payload_policy": "default FunctionLog payload (id, function name, timing, usage); volume estimated from "
            "str(log) UTF-8 bytes; benchmark fixture data only, no real user data",
            "storage": "in-memory Collector",
            "exporter": "none in the pinned Python runtime (see limitations); Boundary cloud publisher inactive "
            "without an API key, no OTLP exporter embedded",
            "env_switches_checked": ["BAML_TRACE_FILE", "BAML_TRACER"],
        },
        "limitations": LIMITATIONS,
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
    parser.add_argument("--repetitions", type=int, default=10, help="predict/bridge repetitions")
    parser.add_argument("--fit-repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--missing-frac", type=float, default=0.0)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="JSON output path (empty string disables writing)")
    args = parser.parse_args(argv)
    if args.repetitions < 1 or args.fit_repetitions < 1 or args.warmups < 0:
        parser.error("repetitions/fit-repetitions must be >= 1 and --warmups >= 0")

    out_path = args.out if args.out else None
    doc = run_benchmark(args.datasets, args.variants, args.repetitions, args.fit_repetitions, args.warmups, args.missing_frac, out_path)
    print(f"wrote {len(doc['results'])} measurement records for datasets={args.datasets} variants={args.variants} to {out_path}")
    for p in doc["parity"]:
        print(
            f"parity[{p['dataset']}]: variants_match={p['predictions_match']} "
            f"sklearn_match={p['matches_sklearn_reference']} (tol={p['sklearn_tol']}, diff={p['sklearn_max_abs_diff']:.3e})"
        )
    return doc


if __name__ == "__main__":
    main()
