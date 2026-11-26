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

Two datasets: Iris for classification and Boston Housing for regression. Boston Housing is a legacy benchmark removed from scikit-learn due to ethical concerns about its variables; use it only for historical numerical and runtime comparison, not for real-world housing decisions.

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
- `examples/`: Iris and Boston Housing examples

## Status

Early experimental scaffold. No BAML estimators are implemented yet. See the repository issues for the MVP roadmap.

No LLM calls, tree models, model exporters, or dashboards are in scope for v0.1.
