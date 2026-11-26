"""Measure the migrated Python SDK with the historical staged workload.

Run after host_tests/prepare.py and SDK generation. No legacy SDK is loaded.
Trace fields are null until a supported capture/export path is demonstrated.
"""
import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

MIGRATION = Path(__file__).resolve().parents[1]
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
sys.path[:0] = [str(MIGRATION / "host_tests/generated"), str(MIGRATION / "gen")]

import numpy as np
from threadpoolctl import threadpool_info
from benchmarks.datasets import DATASETS, SEED, TEST_SIZE, load_split
from benchmarks.reference import b, make_sklearn_pipeline, staged_fit, staged_predict

TOL = {"classification": 1e-3, "regression": 1e-9}


def measure(split, repetitions, fit_repetitions, warmups):
    results = []

    def op(name, batch, reps, warm, fn, single=False):
        for _ in range(warm):
            fn()
        samples = []
        last = None
        for _ in range(reps):
            for i in range(len(split.x_test) if single else 1):
                start = time.perf_counter()
                last = fn(i) if single else fn()
                samples.append(time.perf_counter() - start)
        arr = np.asarray(samples)
        assert np.isfinite(arr).all() and (arr > 0).all()
        results.append({
            "dataset": split.name, "task": split.task,
            "missing_frac": split.missing_frac, "variant": "baml-new-sdk",
            "operation": name, "batch_size": batch, "repetitions": reps,
            "warmups": warm, "samples_s": samples,
            "timing_s": {"p50": float(np.percentile(arr, 50)),
                         "p99": float(np.percentile(arr, 99)),
                         "mean": float(arr.mean()), "min": float(arr.min()),
                         "max": float(arr.max())},
            "throughput_rows_per_s": float(batch / arr.mean()) if batch else None,
            "runtime_only_s": None, "trace": {"spans": None, "bytes": None},
        })
        return last

    assert op("bridge_probe", None, repetitions, warmups, lambda: b.IsFinite(1.0))
    model = op("fit", None, fit_repetitions, warmups, lambda: staged_fit(split))
    pred = op("predict", len(split.x_test), repetitions, warmups,
              lambda: staged_predict(split, model, split.x_test))
    op("predict", 1, repetitions, 0,
       lambda i: staged_predict(split, model, split.x_test[i:i+1]), single=True)
    ref = make_sklearn_pipeline(split.task).fit(split.x_train, split.y_train)
    classification = split.task == "classification"
    expected = ref.predict_proba(split.x_test) if classification else ref.predict(split.x_test)
    assert pred.shape == expected.shape and np.isfinite(pred).all()
    diff = float(np.max(np.abs(pred - expected)))
    labels_match = bool(np.array_equal(ref.classes_[pred.argmax(axis=1)], ref.predict(split.x_test))) if classification else None
    singles = np.concatenate([staged_predict(split, model, split.x_test[i:i+1])
                              for i in range(len(split.x_test))], axis=0)
    assert singles.shape == pred.shape and np.isfinite(singles).all()
    single_diff = float(np.max(np.abs(singles - pred)))
    assert diff <= TOL[split.task] and single_diff <= TOL[split.task] and labels_match is not False
    return results, {"dataset": split.name, "missing_frac": split.missing_frac,
                     "matches_sklearn_reference": True, "sklearn_max_abs_diff": diff,
                     "sklearn_tol": TOL[split.task], "labels_match": labels_match,
                     "single_batch_max_abs_diff": single_diff}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--fit-repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--missing-frac", type=float, default=0.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.repetitions, args.fit_repetitions) < 1 or args.warmups < 0 or not 0 <= args.missing_frac < 1:
        parser.error("invalid repetitions, warmups or missing fraction")
    rows, parity = [], []
    for name in args.datasets:
        r, p = measure(load_split(name, missing_frac=args.missing_frac),
                       args.repetitions, args.fit_repetitions, args.warmups)
        rows.extend(r)
        parity.append(p)
    doc = {"schema_version": 1, "runner": "migration/benchmarks/run.py",
           "generated_at_utc": datetime.now(timezone.utc).isoformat(),
           "config": {"seed": SEED, "test_size": TEST_SIZE,
                      "datasets": args.datasets, "missing_frac": args.missing_frac,
                      "repetitions": args.repetitions, "fit_repetitions": args.fit_repetitions,
                      "warmups": args.warmups, "timer": "time.perf_counter",
                      "timer_unit": "s", "thread_env": {k: os.environ[k] for k in
                          ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
           "environment": {"python": platform.python_version(), "platform": platform.platform(),
                           "machine": platform.machine(), "processor": platform.processor(),
                           "packages": {k: version(k) for k in ("baml_bridge", "numpy", "scipy", "scikit-learn")},
                           "threadpool": threadpool_info()},
           "telemetry": {"BAML_TELEMETRY": os.environ.get("BAML_TELEMETRY"),
                         "capture": "no collector attached; runtime trace state not verified",
                         "trace_metrics": "unsupported/unmeasured; null, not zero"},
           "limitations": ["Python SDK wall time includes serialization and staged bridge calls.",
                           "Bridge probe is a round trip, not a pure FFI measurement.",
                           "Runtime-only timing, Rust SDK, and native trace capture not measured.",
                           "No performance assertions on shared CI."],
           "parity": parity, "results": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    for p in parity:
        print("parity", p)
    for r in rows:
        print(r["dataset"], r["operation"], r["batch_size"], "p50_ms",
              r["timing_s"]["p50"] * 1000, "p99_ms", r["timing_s"]["p99"] * 1000)


if __name__ == "__main__":
    main()
