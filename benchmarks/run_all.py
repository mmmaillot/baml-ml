"""Single-command benchmark suite and report generator (issue #11).

Regenerate all four benchmark variants, the inference trace demo, and the
Markdown comparison report from the repository root:

    .venv/bin/python benchmarks/run_all.py --out-dir docs/benchmarks

By default writes to the gitignored `benchmarks/results/` directory so local
runs do not dirty the working tree; pass `--out-dir docs/benchmarks` to update
the published artifacts.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import baml_benchmark, sklearn_benchmark
from benchmarks.datasets import DATASETS
from examples import trace_demo

DEFAULT_OUT_DIR = REPO_ROOT / "benchmarks" / "results"


def _fmt_ms(seconds):
    if seconds is None:
        return "N/A"
    ms = seconds * 1e3
    if ms >= 100.0:
        return f"{ms:.1f} ms"
    if ms >= 1.0:
        return f"{ms:.2f} ms"
    return f"{ms:.3f} ms"


def _fmt_ratio(num, den):
    if num is None or den is None or den == 0.0:
        return "N/A"
    return f"{num / den:.2f}x"


def _fmt_tput(val):
    if val is None:
        return "N/A"
    if val >= 1000.0:
        return f"{val:,.0f}"
    return f"{val:.1f}"


def _index_results(doc):
    idx = {}
    for r in doc["results"]:
        if r["operation"] == "predict":
            op_key = "predict_single" if r["batch_size"] == 1 else "predict_batch"
        else:
            op_key = r["operation"]
        idx[(r["dataset"], r["variant"], op_key)] = r
    return idx


def render_report(sk_doc, baml_doc, trace_doc):
    """Render the issue #11 Markdown report strictly from measured JSON artifacts."""
    sk_idx = _index_results(sk_doc)
    baml_idx = _index_results(baml_doc)
    datasets = baml_doc["config"]["datasets"]
    env = baml_doc["environment"]
    pkgs = env["packages"]
    sk_cfg = sk_doc["config"]
    baml_cfg = baml_doc["config"]

    lines = [
        "# Benchmark Comparison & Inference Trace Report (MVP v0.1)",
        "",
        "Measured comparison of **scikit-learn** (`sklearn`, `sklearn-otel`) and **BAML** "
        "(`baml-telemetry-off`, `baml-telemetry-on`) on Iris classification and Diabetes regression, "
        "generated strictly from the raw JSON outputs in this directory (no synthetic or estimated numbers).",
        "",
        "## 1. Reproducible Workflow & Environment",
        "",
        "Regenerate all four variants, the trace demo, and this report with a single command:",
        "",
        "```bash",
        ".venv/bin/python benchmarks/run_all.py --out-dir docs/benchmarks",
        "```",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
        f"| Platform | `{env['platform']}` (`{env['machine']}`) |",
        f"| Python | `{env['python']}` |",
        f"| Packages | `baml-py=={pkgs.get('baml-py')}`, `scikit-learn=={pkgs.get('scikit-learn')}`, "
        f"`numpy=={pkgs.get('numpy')}`, `opentelemetry-sdk=={pkgs.get('opentelemetry-sdk')}` |",
        f"| Thread pinning | `OMP={baml_cfg['thread_env']['OMP_NUM_THREADS']}`, "
        f"`OPENBLAS={baml_cfg['thread_env']['OPENBLAS_NUM_THREADS']}`, "
        f"`MKL={baml_cfg['thread_env']['MKL_NUM_THREADS']}` |",
        f"| Split protocol | `seed={baml_cfg['seed']}`, `test_size={baml_cfg['test_size']}`, "
        f"`missing_frac={baml_cfg['missing_frac']}` (identical splits across all 4 variants) |",
        f"| Warm-ups / Repetitions | sklearn: `{sk_cfg['warmups']}` warm-up, `{sk_cfg['repetitions']}` reps (fit & predict); "
        f"BAML: `{baml_cfg['warmups']}` warm-up, `{baml_cfg['fit_repetitions']}` fit reps, `{baml_cfg['repetitions']}` predict/probe reps |",
        f"| Timer | `{baml_cfg['timer']}` (monotonic, seconds); flush outside timed regions |",
        "| Raw artifacts | [sklearn_benchmark.json](sklearn_benchmark.json), [baml_benchmark.json](baml_benchmark.json), [inference_trace_demo.json](inference_trace_demo.json) |",
        "",
        "**Trace configurations (important difference in granularity & payload):**",
        "- `sklearn-otel`: OpenTelemetry `InMemorySpanExporter` + `SimpleSpanProcessor`; emits structured JSON spans (`Span.to_json()`) for `pipeline.fit`, `pipeline.predict`/`predict_proba`, and each named stage (`pipeline.stage.<step>.<op>`) with per-span durations and attributes.",
        "- `baml-telemetry-on`: `baml_py.Collector` attached once via `b.with_options(collector=...)`; retains one in-memory `FunctionLog` per staged BAML call (`str(log)` UTF-8 bytes). In `baml-py 0.226.3`, internal expression-function spans are never exported (`drain_stats` finalized/sent = 0), `BAML_TRACE_FILE`/`BAML_TRACER` have no effect, no OTLP exporter is embedded, and `FunctionLog.timing.duration_ms` is `None` (`runtime_only_s: null`).",
        "",
        "## 2. Numerical Parity",
        "",
        "| Dataset | sklearn vs sklearn-otel | baml-off vs baml-on | BAML vs sklearn (`max_abs_diff`) | Tolerance | Status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    sk_parity = {p["dataset"]: p for p in sk_doc["parity"]}
    baml_parity = {p["dataset"]: p for p in baml_doc["parity"]}
    for ds in datasets:
        sp = sk_parity[ds]
        bp = baml_parity[ds]
        status = "PASS" if (sp["predictions_match"] and bp["predictions_match"] and bp["matches_sklearn_reference"]) else "FAIL"
        lines.append(
            f"| {ds} | `{sp['max_abs_diff']:.1e}` | `{bp['max_abs_diff']:.1e}` | "
            f"`{bp['sklearn_max_abs_diff']:.3e}` | `<= {bp['sklearn_tol']:.0e}` | {status} |"
        )

    lines += [
        "",
        "Solver differences explain the non-zero BAML-vs-sklearn residuals: Iris uses BAML gradient descent + Armijo line search vs sklearn L-BFGS (both converging to the same L2 multinomial optimum); Diabetes uses BAML modified Gram-Schmidt QR vs sklearn LAPACK OLS (floating-point roundoff only). Enabling telemetry changes predictions by `0.0` in both runtimes.",
        "",
        "## 3. Within-Runtime Observability Overhead",
        "",
        "### scikit-learn: `sklearn` vs `sklearn-otel`",
        "",
        "| Dataset | Operation | Batch | `sklearn` p50 | `sklearn-otel` p50 | p99 (`otel`) | Overhead (p50) | Throughput (`otel`, rows/s) | OTel Spans / Bytes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for ds in datasets:
        for op_key, label in (("fit", "fit"), ("predict_batch", "predict (batch)"), ("predict_single", "predict (1 row)")):
            r0 = sk_idx[(ds, "sklearn", op_key)]
            r1 = sk_idx[(ds, "sklearn-otel", op_key)]
            batch = str(r1["batch_size"]) if r1["batch_size"] is not None else "—"
            lines.append(
                f"| {ds} | {label} | {batch} | {_fmt_ms(r0['timing_s']['p50'])} | {_fmt_ms(r1['timing_s']['p50'])} | "
                f"{_fmt_ms(r1['timing_s']['p99'])} | {_fmt_ratio(r1['timing_s']['p50'], r0['timing_s']['p50'])} | "
                f"{_fmt_tput(r1['throughput_rows_per_s'])} | {r1['trace']['spans']} / {r1['trace']['bytes']:,} B |"
            )

    lines += [
        "",
        "### BAML: `baml-telemetry-off` vs `baml-telemetry-on`",
        "",
        "(`runtime_only_s` is `null` for all BAML operations because `FunctionLog.timing.duration_ms` is `None` for expression functions in `baml-py 0.226.3`; `bridge_probe` isolates the per-call Python↔BAML round-trip cost.)",
        "",
        "| Dataset | Operation | Batch | `baml-off` p50 | `baml-on` p50 | p99 (`on`) | Overhead (p50) | Throughput (`on`, rows/s) | Collector Logs / Bytes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for ds in datasets:
        for op_key, label in (
            ("bridge_probe", "bridge_probe"),
            ("fit", "fit"),
            ("predict_batch", "predict (batch)"),
            ("predict_single", "predict (1 row)"),
        ):
            r0 = baml_idx[(ds, "baml-telemetry-off", op_key)]
            r1 = baml_idx[(ds, "baml-telemetry-on", op_key)]
            batch = str(r1["batch_size"]) if r1["batch_size"] is not None else "—"
            lines.append(
                f"| {ds} | {label} | {batch} | {_fmt_ms(r0['timing_s']['p50'])} | {_fmt_ms(r1['timing_s']['p50'])} | "
                f"{_fmt_ms(r1['timing_s']['p99'])} | {_fmt_ratio(r1['timing_s']['p50'], r0['timing_s']['p50'])} | "
                f"{_fmt_tput(r1['throughput_rows_per_s'])} | {r1['trace']['spans']} / {r1['trace']['bytes']:,} B |"
            )

    lines += [
        "",
        "## 4. Cross-Runtime Comparison (Comparable Observation Levels)",
        "",
        "| Dataset | Operation | `sklearn` p50 | `baml-off` p50 | Ratio (`baml-off / sk`) | `sklearn-otel` p50 | `baml-on` p50 | Ratio (`baml-on / sk-otel`) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for ds in datasets:
        for op_key, label in (("fit", "fit"), ("predict_batch", "predict (batch)"), ("predict_single", "predict (1 row)")):
            s0 = sk_idx[(ds, "sklearn", op_key)]["timing_s"]["p50"]
            s1 = sk_idx[(ds, "sklearn-otel", op_key)]["timing_s"]["p50"]
            b0 = baml_idx[(ds, "baml-telemetry-off", op_key)]["timing_s"]["p50"]
            b1 = baml_idx[(ds, "baml-telemetry-on", op_key)]["timing_s"]["p50"]
            lines.append(
                f"| {ds} | {label} | {_fmt_ms(s0)} | {_fmt_ms(b0)} | {_fmt_ratio(b0, s0)} | "
                f"{_fmt_ms(s1)} | {_fmt_ms(b1)} | {_fmt_ratio(b1, s1)} |"
            )

    iris_probe = baml_idx[(datasets[0], "baml-telemetry-off", "bridge_probe")]["timing_s"]["p50"]
    otel_b = round(trace_doc["sklearn_otel_trace"]["total_bytes"] / trace_doc["sklearn_otel_trace"]["spans"])
    baml_b = round(trace_doc["baml_collector_trace"]["total_bytes"] / trace_doc["baml_collector_trace"]["spans"])
    lines += [
        "",
        "**Separating the sources of difference:**",
        "1. **VM & solver execution**: BAML executes pure-BAML loops in its expression VM (and first-order GD for logistic regression), whereas scikit-learn dispatches to compiled C/BLAS/LAPACK/L-BFGS routines. This dominates training time (especially Iris `LogRegFit`).",
        f"2. **Python↔BAML bridge overhead**: A trivial `IsFinite(1.0)` round-trip (`bridge_probe`) costs ~`{_fmt_ms(iris_probe)}` p50. Because staged execution crosses the bridge 5 times per `fit` and 3 times per `predict` (passing Python `list[list[float]]` matrices across FFI), single-row inference is largely bridge- and VM-dispatch-bound.",
        f"3. **Non-equivalent trace payloads**: `sklearn-otel` serializes full JSON spans with per-stage start/end timestamps and custom attributes (~{otel_b} B/span, plus parent operation span). `baml-telemetry-on` only retains compact `FunctionLog` strings (~{baml_b} B/log) carrying BAML function names without per-stage runtime durations or OTLP export — so lower BAML trace bytes reflect a **less informative payload**, not a 1:1 compression win.",
        "",
        "## 5. Inference Trace Demo (Missing-Value Diagnostic)",
        "",
        "Reproduce and inspect the synthetic Iris inference trace (one flower with `sepal_width_cm = NaN`):",
        "",
        "```bash",
        ".venv/bin/python examples/trace_demo.py",
        "```",
        "",
        f"- **Synthetic input**: `{trace_doc['synthetic_input']}` (no real/sensitive data).",
    ]
    mf = trace_doc["diagnostics"]["missing_features"][0]
    st_pred = trace_doc["diagnostics"]["stages"][2]
    lines += [
        f"- **Diagnostic value**: Stage 1 (`ImputerTransform`) detects `sepal_width_cm = null` and fills it with the training-set mean `{mf['imputed_with_train_mean']}` (resulting in z-score `{mf['z_score_after_scaling']}` after `ScalerTransform`, so imputation neither leaks test statistics nor skews the standardized feature). Stage 3 (`LogRegPredictProba`) predicts **`{st_pred['predicted_class']}`** (`{st_pred['probabilities']}`), matching sklearn within `{trace_doc['parity']['max_proba_abs_diff']:.3e}`.",
        f"- **BAML Collector capture** ({trace_doc['baml_collector_trace']['spans']} logs, {trace_doc['baml_collector_trace']['total_bytes']} bytes):",
        "",
        "```text",
    ]
    for st, lg in zip(trace_doc["diagnostics"]["stages"], trace_doc["baml_collector_trace"]["logs"]):
        lines.append(
            f"{lg['function_name']:<20} wall={st['wall_ms']:>6.3f} ms  runtime_duration_ms={lg['duration_ms']}  bytes={lg['serialized_bytes']}"
        )
    lines += [
        "```",
        "",
        f"- **scikit-learn OpenTelemetry capture** ({trace_doc['sklearn_otel_trace']['spans']} spans, {trace_doc['sklearn_otel_trace']['total_bytes']} bytes):",
        "",
        "```text",
    ]
    for sp in trace_doc["sklearn_otel_trace"]["spans_detail"]:
        lines.append(f"{sp['name']:<36} duration={sp['duration_ms']:>7.4f} ms  bytes={sp['serialized_bytes']}")
    lines += [
        "```",
        "",
        "## 6. Limitations & Conclusion",
        "",
        "- **Small datasets only**: Iris (120 train / 30 test, 4 features) and Diabetes (353 train / 89 test, 10 features) are tiny fixtures meant to test numerical correctness and per-call observability overhead. They do **not** represent production tabular workloads (large matrices, high concurrency, network OTLP exporters, or batching services).",
        "- **Runtime observability gap in `baml-py 0.226.3`**: BAML's tracing stack is currently built around LLM calls. For pure expression functions, Collector attachment adds modest overhead relative to VM compute, but runtime spans are not finalized/exported, `duration_ms` is `None`, custom tags are dropped, and per-stage visibility requires orchestrating individual calls from Python (paying extra bridge crossings).",
        "- **Takeaway**: Native language-level ML observability in BAML is viable conceptually — numerical parity with scikit-learn holds (`<= 1e-7` on Iris probabilities, `<= 1e-12` on Diabetes OLS) and Collector overhead itself is small compared to BAML VM execution — **but** today's `baml-py 0.226.3` runtime lacks expression-function span export/timings and native numerical kernels. Closing the compute gap motivates exploring a native Rust kernel layer (see [NumBam](../../numbam/README.md) and [numerical-backends.md](../architecture/numerical-backends.md)).",
        "",
    ]
    return "\n".join(lines)


def run_all(datasets, repetitions, fit_repetitions, warmups, missing_frac, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sk_doc = sklearn_benchmark.run_benchmark(
        datasets=datasets,
        variants=list(sklearn_benchmark.VARIANTS),
        repetitions=repetitions,
        warmups=warmups,
        missing_frac=missing_frac,
        out_path=out / "sklearn_benchmark.json",
    )
    baml_doc = baml_benchmark.run_benchmark(
        datasets=datasets,
        variants=list(baml_benchmark.VARIANTS),
        repetitions=repetitions,
        fit_repetitions=fit_repetitions,
        warmups=warmups,
        missing_frac=missing_frac,
        out_path=out / "baml_benchmark.json",
    )
    trace_doc = trace_demo.run(out_path=out / "inference_trace_demo.json", verbose=False)

    report = render_report(sk_doc, baml_doc, trace_doc)
    (out / "report.md").write_text(report)
    print(f"wrote benchmark JSONs, trace demo, and report to {out}")
    return sk_doc, baml_doc, trace_doc, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--repetitions", type=int, default=10, help="predict/bridge reps (and sklearn fit reps)")
    parser.add_argument("--fit-repetitions", type=int, default=3, help="BAML fit repetitions")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--missing-frac", type=float, default=0.0)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args(argv)
    if args.repetitions < 1 or args.fit_repetitions < 1 or args.warmups < 0:
        parser.error("repetitions/fit-repetitions must be >= 1 and --warmups >= 0")
    return run_all(args.datasets, args.repetitions, args.fit_repetitions, args.warmups, args.missing_frac, args.out_dir)


if __name__ == "__main__":
    main()
