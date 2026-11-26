"""BAML benchmark runner: telemetry off vs telemetry on (issue #10).

Run from the repository root on a clean checkout:

    .venv/bin/python benchmarks/baml_benchmark.py --out benchmarks/results/baml_benchmark.json

Variants: "baml-telemetry-off" (plain BAML calls, no collector attached) and
"baml-telemetry-on" (a baml_py Collector attached once via b.with_options, so
each staged BAML function call retains a FunctionLog). Both variants run the
identical staged operations (bridge_probe / fit / batch predict / single-row
predict) on the identical seed-42 splits from issue #8. Dataset loading is
excluded from every timed region. No LLM is called and Python runs no numerics.

Results follow the shared issue #9 schema (schema_version 1: config, telemetry,
environment, trace_sample, parity, results; timing_s in seconds,
throughput_rows_per_s, trace {spans, bytes}), extended with runtime_only_s
(null: FunctionLog.timing.duration_ms is None for expression functions in
baml-py 0.226.3) and top-level limitations documenting verified runtime bounds.
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
from baml_py import baml_py
from threadpoolctl import threadpool_info

from benchmarks.datasets import DATASETS, SEED, TEST_SIZE, load_split
from benchmarks.reference import b, make_sklearn_pipeline, staged_fit, staged_predict
from baml_client.tracing import flush as baml_flush

SCHEMA_VERSION = 1
VARIANTS = ("baml-telemetry-off", "baml-telemetry-on")
TELEMETRY_ON = "baml-telemetry-on"
DEFAULT_OUT = REPO_ROOT / "benchmarks" / "results" / "baml_benchmark.json"

# sklearn parity tolerances from issue #8 (justified there by solver diffs).
SKLEARN_TOL = {"classification": 1e-3, "regression": 1e-9}

LIMITATIONS = [
    "baml-py 0.226.3 starts an internal span per expression-function call but never finalizes or "
    "exports it (drain_stats: finalized=sent=0 even after flush), so no runtime-internal trace "
    "payload is exported.",
    "BAML_TRACE_FILE and BAML_TRACER have no observable effect on expression functions in the "
    "0.226.3 Python runtime; no OTLP exporter is embedded and the Boundary cloud publisher stays "
    "inactive without an API key (no network egress; benchmark fixture data only).",
    "Collector FunctionLogs record BAML function names (e.g. ImputerFit, LogRegFit), not custom "
    "OTel pipeline.stage.<step>.<op> span names; with_options(tags=...) is not retained on "
    "expression-function logs and tracing.trace cannot wrap BAML calls (Call ID mismatch).",
    "Monolithic pipeline functions emit a single FunctionLog, so this runner executes each stage "
    "from Python for per-stage observability at the cost of extra Python<->BAML bridge crossings.",
    "FunctionLog.timing.duration_ms is None for expression functions: runtime_only_s is null and "
    "bridge overhead is approximated by the bridge_probe operation.",
]


def _make_client(variant):
    """Return (client, collector), configuring Collector once outside timed loops."""
    if variant == TELEMETRY_ON:
        collector = baml_py.Collector("baml-bench")
        return b.with_options(collector=collector), collector
    return b, None


def _trace_delta(collector, log_names):
    if collector is None:
        return {"spans": 0, "bytes": 0}
    logs = collector.logs
    log_names.update(log.function_name for log in logs)
    out = {
        "spans": len(logs),
        "bytes": sum(len(str(log).encode("utf-8")) for log in logs),
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
    client, collector = _make_client(variant)
    records = []
    log_names = set()

    def run_op(operation, batch_size, reps, n_warmups, fn, per_row=False):
        for _ in range(n_warmups):
            fn()
        baml_flush()
        if collector is not None:
            collector.clear()

        times_s = []
        last = None
        if per_row:
            for _ in range(reps):
                for i in range(split.x_test.shape[0]):
                    t0 = time.perf_counter()
                    last = fn(i)
                    times_s.append(time.perf_counter() - t0)
        else:
            for _ in range(reps):
                t0 = time.perf_counter()
                last = fn()
                times_s.append(time.perf_counter() - t0)

        baml_flush()  # outside timed region
        n_rows = batch_size if operation == "predict" else None
        timing_s, throughput = _timing_record(times_s, n_rows)
        records.append(
            {
                "dataset": split.name,
                "task": split.task,
                "missing_frac": split.missing_frac,
                "variant": variant,
                "operation": operation,
                "batch_size": batch_size,
                "repetitions": reps,
                "warmups": n_warmups,
                "timing_s": timing_s,
                "throughput_rows_per_s": throughput,
                "runtime_only_s": None,  # duration_ms is None for expr functions in baml-py 0.226.3
                "trace": _trace_delta(collector, log_names),
            }
        )
        return last

    # bridge probe: trivial BAML call to approximate Python<->BAML bridge overhead
    run_op("bridge_probe", None, repetitions, warmups, lambda: client.IsFinite(1.0))

    # fit (staged, on training rows only)
    state = run_op("fit", None, fit_repetitions, warmups, lambda: staged_fit(split, client))

    # batch predict on the full held-out split
    last_pred = run_op(
        "predict",
        int(split.x_test.shape[0]),
        repetitions,
        warmups,
        lambda: staged_predict(split, state, split.x_test, client),
    )

    # single-row predict (batch size 1); already warmed by batch predict
    run_op(
        "predict",
        1,
        repetitions,
        0,
        lambda i: staged_predict(split, state, split.x_test[i : i + 1], client),
        per_row=True,
    )

    return records, last_pred, sorted(log_names)


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
    datasets = list(dict.fromkeys(datasets))
    variants = list(dict.fromkeys(variants))
    results = []
    parity = []
    trace_sample = set()

    for name in datasets:
        split = load_split(name, missing_frac=missing_frac)
        preds = {}
        for variant in variants:
            records, last_pred, log_names = _measure(variant, split, repetitions, fit_repetitions, warmups)
            results.extend(records)
            trace_sample.update(log_names)
            preds[variant] = last_pred

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
            entry["predictions_match"] = bool(np.array_equal(a, b_))
            entry["max_abs_diff"] = float(np.max(np.abs(a.astype(float) - b_.astype(float))))

        # #8 parity: reuse benchmarked BAML predictions vs sklearn reference on the same split
        baml_out = preds.get("baml-telemetry-off", next(iter(preds.values())))
        ref = make_sklearn_pipeline(split.task).fit(split.x_train, split.y_train)
        ref_out = ref.predict_proba(split.x_test) if split.task == "classification" else ref.predict(split.x_test)
        diff = float(np.max(np.abs(baml_out.astype(float) - np.asarray(ref_out, dtype=float))))
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
                "baml-telemetry-on": "baml_py.Collector attached once via with_options; one FunctionLog per BAML "
                "stage call; tracing.flush() outside timed regions",
            },
            "granularity": "one retained FunctionLog per staged BAML function call (ImputerFit/Transform, "
            "ScalerFit/Transform, LogRegFit/PredictProba or LinearRegressionFit/Predict)",
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
    parser.add_argument("--repetitions", type=int, default=10, help="predict/bridge repetitions")
    parser.add_argument("--fit-repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--missing-frac", type=float, default=0.0)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="JSON output path (empty string disables writing)")
    args = parser.parse_args(argv)
    if args.repetitions < 1 or args.fit_repetitions < 1 or args.warmups < 0:
        parser.error("repetitions/fit-repetitions must be >= 1 and --warmups >= 0")

    out_path = args.out if args.out else None
    doc = run_benchmark(
        args.datasets, args.variants, args.repetitions, args.fit_repetitions, args.warmups, args.missing_frac, out_path
    )
    print(f"wrote {len(doc['results'])} measurement records for datasets={args.datasets} variants={args.variants} to {out_path}")
    for p in doc["parity"]:
        print(
            f"parity[{p['dataset']}]: variants_match={p['predictions_match']} "
            f"sklearn_match={p['matches_sklearn_reference']} (tol={p['sklearn_tol']}, diff={p['sklearn_max_abs_diff']:.3e})"
        )
    return doc


if __name__ == "__main__":
    main()
