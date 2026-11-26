# Benchmark Comparison & Inference Trace Report (MVP v0.1)

Measured comparison of **scikit-learn** (`sklearn`, `sklearn-otel`) and **BAML** (`baml-telemetry-off`, `baml-telemetry-on`) on Iris classification and Diabetes regression, generated strictly from the raw JSON outputs in this directory (no synthetic or estimated numbers).

## 1. Reproducible Workflow & Environment

Regenerate all four variants, the trace demo, and this report with a single command:

```bash
.venv/bin/python benchmarks/run_all.py --out-dir docs/benchmarks
```

| Parameter | Value |
| --- | --- |
| Platform | `Linux-6.18.57-x86_64-with-glibc2.39` (`x86_64`) |
| Python | `3.12.16` |
| Packages | `baml-py==0.226.3`, `scikit-learn==1.9.1`, `numpy==2.5.5`, `opentelemetry-sdk==1.45.2` |
| Thread pinning | `OMP=1`, `OPENBLAS=1`, `MKL=1` |
| Split protocol | `seed=42`, `test_size=0.2`, `missing_frac=0.0` (identical splits across all 4 variants) |
| Warm-ups / Repetitions | sklearn: `1` warm-up, `10` reps (fit & predict); BAML: `1` warm-up, `3` fit reps, `10` predict/probe reps |
| Timer | `time.perf_counter` (monotonic, seconds); flush outside timed regions |
| Raw artifacts | [sklearn_benchmark.json](sklearn_benchmark.json), [baml_benchmark.json](baml_benchmark.json), [inference_trace_demo.json](inference_trace_demo.json) |

**Trace configurations (important difference in granularity & payload):**
- `sklearn-otel`: OpenTelemetry `InMemorySpanExporter` + `SimpleSpanProcessor`; emits structured JSON spans (`Span.to_json()`) for `pipeline.fit`, `pipeline.predict`/`predict_proba`, and each named stage (`pipeline.stage.<step>.<op>`) with per-span durations and attributes.
- `baml-telemetry-on`: `baml_py.Collector` attached once via `b.with_options(collector=...)`; retains one in-memory `FunctionLog` per staged BAML call (`str(log)` UTF-8 bytes). In `baml-py 0.226.3`, internal expression-function spans are never exported (`drain_stats` finalized/sent = 0), `BAML_TRACE_FILE`/`BAML_TRACER` have no effect, no OTLP exporter is embedded, and `FunctionLog.timing.duration_ms` is `None` (`runtime_only_s: null`).

## 2. Numerical Parity

| Dataset | sklearn vs sklearn-otel | baml-off vs baml-on | BAML vs sklearn (`max_abs_diff`) | Tolerance | Status |
| --- | --- | --- | --- | --- | --- |
| iris | `0.0e+00` | `0.0e+00` | `8.511e-08` | `<= 1e-03` | PASS |
| diabetes | `0.0e+00` | `0.0e+00` | `3.979e-13` | `<= 1e-09` | PASS |

Solver differences explain the non-zero BAML-vs-sklearn residuals: Iris uses BAML gradient descent + Armijo line search vs sklearn L-BFGS (both converging to the same L2 multinomial optimum); Diabetes uses BAML modified Gram-Schmidt QR vs sklearn LAPACK OLS (floating-point roundoff only). Enabling telemetry changes predictions by `0.0` in both runtimes.

## 3. Within-Runtime Observability Overhead

### scikit-learn: `sklearn` vs `sklearn-otel`

| Dataset | Operation | Batch | `sklearn` p50 | `sklearn-otel` p50 | p99 (`otel`) | Overhead (p50) | Throughput (`otel`, rows/s) | OTel Spans / Bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| iris | fit | — | 13.01 ms | 12.02 ms | 13.31 ms | 0.92x | N/A | 44 / 39,523 B |
| iris | predict (batch) | 30 | 1.90 ms | 1.61 ms | 1.83 ms | 0.85x | 18,141 | 44 / 39,523 B |
| iris | predict (1 row) | 1 | 1.64 ms | 1.68 ms | 2.17 ms | 1.02x | 585.4 | 1200 / 1,077,600 B |
| diabetes | fit | — | 4.25 ms | 4.55 ms | 4.88 ms | 1.07x | N/A | 44 / 39,512 B |
| diabetes | predict (batch) | 89 | 1.68 ms | 1.87 ms | 2.02 ms | 1.11x | 47,630 | 44 / 39,303 B |
| diabetes | predict (1 row) | 1 | 1.61 ms | 1.69 ms | 2.82 ms | 1.05x | 565.7 | 3560 / 3,179,080 B |

### BAML: `baml-telemetry-off` vs `baml-telemetry-on`

(`runtime_only_s` is `null` for all BAML operations because `FunctionLog.timing.duration_ms` is `None` for expression functions in `baml-py 0.226.3`; `bridge_probe` isolates the per-call Python↔BAML round-trip cost.)

| Dataset | Operation | Batch | `baml-off` p50 | `baml-on` p50 | p99 (`on`) | Overhead (p50) | Throughput (`on`, rows/s) | Collector Logs / Bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| iris | bridge_probe | — | 1.14 ms | 1.07 ms | 1.79 ms | 0.94x | N/A | 10 / 2,610 B |
| iris | fit | — | 13563.2 ms | 13429.7 ms | 13547.7 ms | 0.99x | N/A | 15 / 3,972 B |
| iris | predict (batch) | 30 | 13.00 ms | 12.22 ms | 13.03 ms | 0.94x | 2,446 | 30 / 8,080 B |
| iris | predict (1 row) | 1 | 4.53 ms | 4.58 ms | 9.99 ms | 1.01x | 202.0 | 900 / 242,400 B |
| diabetes | bridge_probe | — | 1.20 ms | 1.12 ms | 1.59 ms | 0.93x | N/A | 10 / 2,610 B |
| diabetes | fit | — | 317.8 ms | 266.8 ms | 267.3 ms | 0.84x | N/A | 15 / 4,002 B |
| diabetes | predict (batch) | 89 | 28.21 ms | 28.37 ms | 32.57 ms | 1.01x | 3,107 | 30 / 8,130 B |
| diabetes | predict (1 row) | 1 | 4.57 ms | 5.97 ms | 12.58 ms | 1.31x | 156.8 | 2670 / 723,570 B |

## 4. Cross-Runtime Comparison (Comparable Observation Levels)

| Dataset | Operation | `sklearn` p50 | `baml-off` p50 | Ratio (`baml-off / sk`) | `sklearn-otel` p50 | `baml-on` p50 | Ratio (`baml-on / sk-otel`) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| iris | fit | 13.01 ms | 13563.2 ms | 1042.47x | 12.02 ms | 13429.7 ms | 1116.96x |
| iris | predict (batch) | 1.90 ms | 13.00 ms | 6.83x | 1.61 ms | 12.22 ms | 7.58x |
| iris | predict (1 row) | 1.64 ms | 4.53 ms | 2.76x | 1.68 ms | 4.58 ms | 2.72x |
| diabetes | fit | 4.25 ms | 317.8 ms | 74.74x | 4.55 ms | 266.8 ms | 58.67x |
| diabetes | predict (batch) | 1.68 ms | 28.21 ms | 16.81x | 1.87 ms | 28.37 ms | 15.20x |
| diabetes | predict (1 row) | 1.61 ms | 4.57 ms | 2.84x | 1.69 ms | 5.97 ms | 3.53x |

**Separating the sources of difference:**
1. **VM & solver execution**: BAML executes pure-BAML loops in its expression VM (and first-order GD for logistic regression), whereas scikit-learn dispatches to compiled C/BLAS/LAPACK/L-BFGS routines. This dominates training time (especially Iris `LogRegFit`).
2. **Python↔BAML bridge overhead**: A trivial `IsFinite(1.0)` round-trip (`bridge_probe`) costs ~`1.14 ms` p50. Because staged execution crosses the bridge 5 times per `fit` and 3 times per `predict` (passing Python `list[list[float]]` matrices across FFI), single-row inference is largely bridge- and VM-dispatch-bound.
3. **Non-equivalent trace payloads**: `sklearn-otel` serializes full JSON spans with per-stage start/end timestamps and custom attributes (~898 B/span, plus parent operation span). `baml-telemetry-on` only retains compact `FunctionLog` strings (~269 B/log) carrying BAML function names without per-stage runtime durations or OTLP export — so lower BAML trace bytes reflect a **less informative payload**, not a 1:1 compression win.

## 5. Inference Trace Demo (Missing-Value Diagnostic)

Reproduce and inspect the synthetic Iris inference trace (one flower with `sepal_width_cm = NaN`):

```bash
.venv/bin/python examples/trace_demo.py
```

- **Synthetic input**: `{'sepal_length_cm': 5.1, 'sepal_width_cm': None, 'petal_length_cm': 1.4, 'petal_width_cm': 0.2}` (no real/sensitive data).
- **Diagnostic value**: Stage 1 (`ImputerTransform`) detects `sepal_width_cm = null` and fills it with the training-set mean `3.069643` (resulting in z-score `0.0` after `ScalerTransform`, so imputation neither leaks test statistics nor skews the standardized feature). Stage 3 (`LogRegPredictProba`) predicts **`setosa`** (`{'setosa': 0.95295, 'versicolor': 0.047046, 'virginica': 5e-06}`), matching sklearn within `1.079e-08`.
- **BAML Collector capture** (3 logs, 808 bytes):

```text
ImputerTransform     wall= 1.523 ms  runtime_duration_ms=None  bytes=269
ScalerTransform      wall= 1.444 ms  runtime_duration_ms=None  bytes=268
LogRegPredictProba   wall= 1.565 ms  runtime_duration_ms=None  bytes=271
```

- **scikit-learn OpenTelemetry capture** (4 spans, 3592 bytes):

```text
pipeline.stage.imputer.transform     duration= 1.2149 ms  bytes=904
pipeline.stage.scaler.transform      duration= 0.3461 ms  bytes=903
pipeline.stage.clf.predict_proba     duration= 0.3453 ms  bytes=905
pipeline.predict_proba               duration= 2.2943 ms  bytes=880
```

## 6. Limitations & Conclusion

- **Small datasets only**: Iris (120 train / 30 test, 4 features) and Diabetes (353 train / 89 test, 10 features) are tiny fixtures meant to test numerical correctness and per-call observability overhead. They do **not** represent production tabular workloads (large matrices, high concurrency, network OTLP exporters, or batching services).
- **Runtime observability gap in `baml-py 0.226.3`**: BAML's tracing stack is currently built around LLM calls. For pure expression functions, Collector attachment adds modest overhead relative to VM compute, but runtime spans are not finalized/exported, `duration_ms` is `None`, custom tags are dropped, and per-stage visibility requires orchestrating individual calls from Python (paying extra bridge crossings).
- **Takeaway**: Native language-level ML observability in BAML is viable conceptually — numerical parity with scikit-learn holds (`<= 1e-7` on Iris probabilities, `<= 1e-12` on Diabetes OLS) and Collector overhead itself is small compared to BAML VM execution — **but** today's `baml-py 0.226.3` runtime lacks expression-function span export/timings and native numerical kernels. Closing the compute gap motivates exploring a native Rust kernel layer (see [NumBam](../../numbam/README.md) and [numerical-backends.md](../architecture/numerical-backends.md)).
