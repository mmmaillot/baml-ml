# baml-ml

What would scikit-learn look like if observability were built into the language?

An experimental classical ML library written in [BAML](https://github.com/BoundaryML/baml), targeting native execution and built-in observability. No LLMs or API keys required.

## Implemented

- Mean imputation and standard scaling
- Linear regression (OLS) and multinomial logistic regression (L2)
- Classification and regression pipelines
- Native profiling and selective semantic tracing, validated on an Iris pipeline

Numerical parity against scikit-learn is tested on Iris and Diabetes, including missing values. The current implementation is in [migration/](migration/README.md), pinned to BAML 0.26.0.

## Native observability

Four declarative span directives expose the actual input, imputed values, standardized values, model parameters and predicted probabilities, together with the call hierarchy and durations. Default profiling aggregates function invocations without recording each intermediate value.

See [native ML observability](experiments/native_ml_observability/README.md) for verified payloads, typed NaN, structured errors and storage measurements.

## Performance baseline

Native Iris workloads, medians of four trials on a shared CI runner ([raw evidence and protocol](experiments/native_benchmark/README.md)):

| Workload | Off | Default | Empty span | Selective span |
|---|---:|---:|---:|---:|
| Single prediction | 16.0 µs | 23.0 µs | 24.8 µs | 35.6 µs |
| Batch of 30 | 227 µs | 308 µs | 313 µs | 388 µs |
| Training (120 rows) | 238 ms | 299 ms | 289 ms | 291 ms |

Single inference throughput ranges from 62,522 predictions/s without tracing to 28,123/s with selective spans. The latter adds approximately 373 bytes of BTEL and 1,247 bytes of SQL index per repeated prediction. CAS marginal growth was zero for repeated identical inputs. The SQL index is created during querying.

These are aggregate timings for small, repeated workloads, not production latency percentiles or a controlled comparison with scikit-learn.

## Run

Follow the [pinned CLI installation and native test instructions](migration/README.md). To reproduce the native experiments with the verified CLI:

```sh
export BAML_CLI=/absolute/path/to/baml-cli
bash experiments/native_ml_observability/run.sh /tmp/native-ml-observability
bash experiments/native_benchmark/run.sh /tmp/native-benchmark
```

## Validation policy

Native checks run automatically; full performance, scaling and historical differential campaigns are manual workflows. A reproducible Apple Silicon benchmark runner is deferred; see the scaling protocol.

## Next

- [Scaling results](experiments/scaling/README.md): near-linear feature scaling, batching plateau and ~8.9 kB marginal storage per distinct 32-feature prediction with spans and index.
- [Hotspot analysis](experiments/profiling/README.md): inference is dominated by checks and preprocessing; training has larger coarse-grained numerical kernel opportunities. Investigate redundant internal checks before broad Rust integration.
- [NumBam](numbam/README.md): experimental Rust numerical backend, deferred pending measured need ([#20](https://github.com/mmmaillot/baml-ml/issues/20)).
- [Native backend and embedded runtime design](docs/architecture/numerical-backends.md).

## Historical MVP

The original toolchain and Collector-based experiments are preserved at [v0.1.0](https://github.com/mmmaillot/baml-ml/tree/v0.1.0). See the [historical benchmark report](docs/benchmarks/report.md) and [implementation notes](docs/implementation.md). Those results are not the current native baseline.

Experimental software, not a production-ready replacement for scikit-learn.

[Apache 2.0](LICENSE).
