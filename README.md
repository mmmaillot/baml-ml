# baml-ml

A hyperobservable inference runtime for models trained with scikit-learn, built in [BAML](https://github.com/BoundaryML/baml).

Export fitted pipelines as portable, versioned data-only artifacts; load and execute them natively in BAML. Reconstruct individual predictions from preprocessing to output. Training stays outside the runtime. No LLMs or API keys required.

> **Observability should be a property of the runtime, not a responsibility of the developer.**

As generated code and model fleets grow, consistent runtime observation complements tests and code review. BAML-ML explores this thesis for classical ML: semantic traces are part of inference, not bespoke instrumentation per pipeline. Hybrid BAML/LLM workflows are a possible use, not a prerequisite.

## Supported scope

- Mean `SimpleImputer` and centered/scaled `StandardScaler`, in that order.
- Single-target dense float64 `LinearRegression`.
- `LogisticRegression`: binary sigmoid and multinomial softmax with ordered integer classes.
- Exact three-step pipeline, with validated model and feature dimensions and feature order.

See the [artifact contract](artifacts/README.md) for precise restrictions and export instructions. Scikit-learn 1.8.0 is the validated reference version. Other training frameworks and estimator families are deferred.

## Model lifecycle

Load or prepare a model once, then call its prediction methods for repeated single or batch requests in the same BAML execution. The loader owns and validates the parameter snapshot; every request still validates its inputs. See the [API examples](artifacts/README.md#export-and-run).

## Standalone-first delivery

The MVP is a native packaged executable for batch and edge inference. No inference SDK or serving framework. See the [standalone-first ADR](docs/architecture/standalone-first.md). Linux x86-64 packaging is [validated offline](https://github.com/mmmaillot/baml-ml/actions/runs/45958260690); the [CI binary artifact](https://github.com/mmmaillot/baml-ml/actions/runs/45958260690/artifacts/15179970292) is available. See [standalone batch usage and build](experiments/standalone/README.md).

## Batch and edge benchmark

Median latency on a MacBook Pro (Apple M1 Pro, 32 GB), using the same fitted Iris model and identical batches for `predict-proba`. Each configuration had 15 measured runs after warmup.

| Measurement | 1 row | 128 rows | 4,096 rows |
|---|---:|---:|---:|
| BAML process, telemetry off | 32.0 ms | 33.2 ms | 64.7 ms |
| BAML process, default semantic mode | 110.6 ms | 111.7 ms | 148.9 ms |
| Python process, scikit-learn | 898.3 ms | 895.0 ms | 895.4 ms |
| BAML loaded model, telemetry off | 0.009 ms | 0.739 ms | 23.388 ms |
| scikit-learn loaded model | 0.386 ms | 0.406 ms | 0.728 ms |

Process timings include startup, model loading, JSON input and output, and shutdown. Loaded model timings cover inference after model loading and input decoding. On this workload, BAML finished sooner when each batch launched a process. With both models loaded, BAML was faster for one row and scikit-learn was faster for the two larger batches.

The [full report](experiments/standalone_benchmark/m1-pro-2026-11-27.md) includes throughput, peak memory, recording sizes, probability parity and error handling. See the [reproduction instructions](experiments/standalone_benchmark/README.md).

## Run and validate

The native runtime is pinned to BAML 0.26.0. Follow the [installation and test instructions](docs/development.md).

```sh
export BAML_CLI=/absolute/path/to/baml-cli
"$BAML_CLI" test --max-concurrency 1 --test-timeout 5m
bash experiments/imported_inference/run.sh /tmp/imported-inference
bash experiments/native_ml_observability/run.sh /tmp/native-ml-observability
```

[Imported inference CI](https://github.com/mmmaillot/baml-ml/actions/runs/45940061227) validated the exporter and three independent fitted artifacts (Iris binary/multiclass and Diabetes), including missing inputs and semantic traces. The cleanup retains these checks.

## Native observability

The loaded-model runtime uses native BAML trace options with stable semantic boundaries and inherited model identity. Metadata-only semantic spans are the default; Full captures intermediate values and a one-time load artifact, Timing uses aggregate profiling, and the engine-wide off setting disables telemetry. See the [observation contract v1](docs/architecture/observation-contract.md) and [historical native ML observability evidence](experiments/native_ml_observability/README.md).

Adaptive replays only selected rows in Full. The packed `shadow` command compares same-family models and returns the champion predictions. On an Apple M1 Pro, a 4,096-row Iris batch with 52 anomalies used **84% fewer BTEL and CAS bytes combined** in Adaptive than in Full (172 kB vs 1,078 kB), with a median wall time of **238 ms vs 159 ms**. [Adaptive and shadow contract](docs/architecture/adaptive-shadow-observation.md) · [Benchmark and SQL evidence](experiments/adaptive_shadow/results/m1-pro-2026-11-27/report.md)

## Roadmap

1. Qualify native macOS ARM64 and Linux ARM64 packages ([#56](https://github.com/mmmaillot/baml-ml/issues/56)).
2. Establish a reproducible batch and edge performance baseline ([#57](https://github.com/mmmaillot/baml-ml/issues/57)). The first Apple Silicon results are [above](#batch-and-edge-benchmark).
3. Extend estimator coverage selectively, starting with decision trees after the baseline.
4. Improve sensitive-data controls and retention; evaluate deployment requirements.
5. Consider numerical acceleration only when representative inference profiles justify it ([#20](https://github.com/mmmaillot/baml-ml/issues/20)).

[Architecture decision](docs/architecture/inference-first.md).

## Historical checkpoint

The full pre-cleanup project (native training, old SDKs, numerical backends, benchmarks and evidence) is preserved at [pre-inference-cleanup](https://github.com/mmmaillot/baml-ml/tree/pre-inference-cleanup), commit `d87488f68d7ebc4025a9c8199131c4468f0f4982`. These are historical experiments, not active code or current imported-runtime performance claims.

- [Native baseline](https://github.com/mmmaillot/baml-ml/tree/pre-inference-cleanup/experiments/native_benchmark/README.md)
- [Scaling](https://github.com/mmmaillot/baml-ml/tree/pre-inference-cleanup/experiments/scaling/README.md)
- [Profiling](https://github.com/mmmaillot/baml-ml/tree/pre-inference-cleanup/experiments/profiling/README.md)
- [Validation optimization](https://github.com/mmmaillot/baml-ml/tree/pre-inference-cleanup/experiments/validation/README.md)
- [NumBam](https://github.com/mmmaillot/baml-ml/tree/pre-inference-cleanup/numbam/README.md)

Experimental software. [Apache 2.0](LICENSE).
