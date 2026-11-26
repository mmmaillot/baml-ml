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

We want to measure the cost of making ML observable at runtime, rather than manually instrumenting each pipeline.

| Backend | Without tracing | With tracing |
| --- | --- | --- |
| scikit-learn | Vanilla | OpenTelemetry |
| BAML | Telemetry off | Collector (per-stage) |

We compare numerical parity, training time, inference latency, throughput and trace volume. Both the scikit-learn and BAML benchmark runners are implemented; publishing the final comparison results is still pending.

## Run it

Requires Python 3.12 and Node.js 22. The BAML CLI and runtime are pinned to 0.226.3.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
npm ci
npx baml --features beta generate

.venv/bin/python -m pytest tests/
.venv/bin/python examples/iris_example.py
.venv/bin/python examples/diabetes_example.py
```

Run the benchmarks:

```bash
.venv/bin/python benchmarks/sklearn_benchmark.py
.venv/bin/python benchmarks/baml_benchmark.py
```

## Next

- [MVP roadmap](https://github.com/mmmaillot/baml-ml/issues/1)
- [Implementation notes](docs/implementation.md), including numerical methods and BAML limitations.
- [NumBam](numbam/README.md): minimal Rust numerical backend in this repository; BAML integration is still experimental.
- [Native backend and embedded runtime design](docs/architecture/numerical-backends.md).

Experimental software, not a production-ready replacement for scikit-learn.

[Apache 2.0](LICENSE).
