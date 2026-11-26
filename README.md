# baml-ml

**Classical machine learning. Observable by design.**

An experimental library for classical machine learning in [BAML](https://github.com/BoundaryML/baml). The goal is to evaluate the usefulness and cost of runtime-native observability compared with manually instrumented scikit-learn pipelines.

## MVP

Implement training and inference in BAML, with a small scikit-learn-inspired API:

- `SimpleImputer` (mean strategy)
- `StandardScaler`
- `LinearRegression` (ordinary least squares)
- `LogisticRegression` (multiclass)
- `Pipeline`

Two built-in scikit-learn datasets: Iris for classification (`load_iris`, 150 samples, 4 features) and Diabetes for regression (`load_diabetes`, 442 samples, 10 features). Use `load_diabetes(scaled=False)` to exercise our own preprocessing without the dataset loader's default scaling.

## Experiment

| Configuration | Purpose |
| --- | --- |
| scikit-learn | Uninstrumented baseline |
| scikit-learn + OpenTelemetry | Explicitly instrumented baseline |
| BAML, telemetry off | Runtime baseline |
| BAML, telemetry on | Native observability |

Measure training time, inference latency (p50/p99), throughput, numerical parity, and telemetry overhead. Use identical inputs and train/test splits. Compare equivalent trace granularity and payloads; distinguish default profiling from detailed traces. Record runtime versions and hardware.

## Repository layout

- `baml_src/`: native ML implementations
- `benchmarks/`: comparison harness
- `tests/`: unit and numerical parity tests
- `examples/`: Iris and Diabetes examples

## Status

Early experimental scaffold. No BAML estimators are implemented yet. See the repository issues for the MVP roadmap.

No LLM calls, tree models, model exporters, or dashboards are in scope for v0.1.
