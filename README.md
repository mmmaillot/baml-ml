# baml-ml

What would scikit-learn look like if observability were built into the language?

baml-ml is an experiment in classical machine learning written in [BAML](https://github.com/BoundaryML/baml). Both training and inference run in BAML, including the numerical primitives. Python is used for loading data, testing and benchmarking against scikit-learn. No LLMs or API keys are needed.

## What's implemented

- Mean imputation and standard scaling
- Linear regression (OLS)
- Multinomial logistic regression (L2)
- Classification and regression pipelines

The pipelines are tested against scikit-learn on Iris and Diabetes, with and without missing values.

## The experiment

We measure numerical parity, training time, inference latency, throughput and trace volume across four variants:

| Backend | Without tracing | With tracing |
| --- | --- | --- |
| scikit-learn | Vanilla (`sklearn`) | OpenTelemetry (`sklearn-otel`) |
| BAML | Telemetry off (`baml-telemetry-off`) | Collector (`baml-telemetry-on`) |

See the **[Benchmark comparison & inference trace report](docs/benchmarks/report.md)** (raw JSON artifacts under [`docs/benchmarks/`](docs/benchmarks/)). Numerical parity holds (`<= 1e-7` on Iris probabilities, `<= 1e-12` on Diabetes OLS) and BAML Collector overhead is modest relative to VM compute, but `baml-py 0.226.3` does not yet export expression-function spans or runtime durations and remains orders of magnitude slower than compiled C/BLAS kernels on these small datasets.

## Run it

Requires Python 3.12 and Node.js 22. The BAML CLI and runtime are pinned to 0.226.3.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
npm ci
npx baml --features beta generate

.venv/bin/python -m pytest tests/
```

Run the examples and the synthetic missing-value inference trace demo:

```bash
.venv/bin/python examples/iris_example.py
.venv/bin/python examples/diabetes_example.py
.venv/bin/python examples/trace_demo.py
```

Regenerate all four benchmark variants, the trace demo, and the comparison report:

```bash
.venv/bin/python benchmarks/run_all.py
# or overwrite the published report & raw JSON files:
.venv/bin/python benchmarks/run_all.py --out-dir docs/benchmarks
```

Individual runners are also available: [`benchmarks/sklearn_benchmark.py`](benchmarks/sklearn_benchmark.py) and [`benchmarks/baml_benchmark.py`](benchmarks/baml_benchmark.py).

## Next

- [Benchmark report & raw results](docs/benchmarks/report.md)
- [MVP roadmap](https://github.com/mmmaillot/baml-ml/issues/1)
- [Implementation notes](docs/implementation.md), including numerical methods and BAML limitations.
- [NumBam](numbam/README.md): minimal Rust numerical backend in this repository; BAML integration is still experimental.
- [Native backend and embedded runtime design](docs/architecture/numerical-backends.md).

Experimental software, not a production-ready replacement for scikit-learn.

[Apache 2.0](LICENSE).

## New BAML language migration

The original MVP is frozen at [v0.1.0](https://github.com/mmmaillot/baml-ml/tree/v0.1.0). The BAML language 0.25.0 port lives in [migration/](migration/README.md), with native tests and a generated Python SDK regression suite. See that directory for pinned installation, generation and test commands. The historical commands above still target the original MVP toolchain.
