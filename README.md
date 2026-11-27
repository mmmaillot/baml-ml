# baml-ml

A hyperobservable inference runtime for models trained with scikit-learn, built in [BAML](https://github.com/BoundaryML/baml).

Export fitted pipelines as portable, versioned data-only artifacts; load and execute them natively in BAML. Reconstruct individual predictions from preprocessing to output. Training stays outside the runtime. No LLMs or API keys required.

## Supported scope

- Mean `SimpleImputer` and centered/scaled `StandardScaler`, in that order.
- Single-target dense float64 `LinearRegression`.
- `LogisticRegression`: binary sigmoid and multinomial softmax with ordered integer classes.
- Exact three-step pipeline, with validated model and feature dimensions and feature order.

See the [artifact contract](migration/artifacts/README.md) for precise restrictions and export instructions. Scikit-learn 1.8.0 is the validated reference version. Other training frameworks and estimator families are deferred.

## Run and validate

The native runtime is pinned to BAML 0.26.0. Follow the [installation and test instructions](migration/README.md).

```sh
export BAML_CLI=/absolute/path/to/baml-cli
(cd migration && "$BAML_CLI" test --max-concurrency 1 --test-timeout 5m)
bash experiments/imported_inference/run.sh /tmp/imported-inference
bash experiments/native_ml_observability/run.sh /tmp/native-ml-observability
```

[Imported inference CI](https://github.com/mmmaillot/baml-ml/actions/runs/45940061227) validated the exporter and three independent fitted artifacts (Iris binary/multiclass and Diabetes), including missing inputs and semantic traces. The cleanup retains these checks.

## Native observability

Selective semantic spans capture the model identity, inputs, imputed and standardized values, model parameters and predictions with parent IDs and durations. Default profiling aggregates calls; empty spans omit payloads. [Native ML observability evidence](experiments/native_ml_observability/README.md).

## Roadmap

1. Strengthen imported-model validation and reference coverage.
2. Define inference observation policies, sensitive-data controls and retention.
3. Establish standalone or embedded packaging.
4. Consider numerical acceleration only when representative inference profiles justify it ([#20](https://github.com/mmmaillot/baml-ml/issues/20)).

[Architecture decision](docs/architecture/inference-first.md).

## Historical checkpoint

The full pre-cleanup project (native training, old SDKs, numerical backends, benchmarks and evidence) is preserved at [pre-inference-cleanup](https://github.com/mmmaillot/baml-ml/tree/pre-inference-cleanup), commit `d87488f68d7ebc4025a9c8199131c4468f0f4982`. These are historical experiments, not active code or current imported-runtime performance claims.

- [Native baseline](https://github.com/mmmaillot/baml-ml/tree/pre-inference-cleanup/experiments/native_benchmark/README.md)
- [Scaling](https://github.com/mmmaillot/baml-ml/tree/pre-inference-cleanup/experiments/scaling/README.md)
- [Profiling](https://github.com/mmmaillot/baml-ml/tree/pre-inference-cleanup/experiments/profiling/README.md)
- [Validation optimization](https://github.com/mmmaillot/baml-ml/tree/pre-inference-cleanup/experiments/validation/README.md)
- [NumBam](https://github.com/mmmaillot/baml-ml/tree/pre-inference-cleanup/numbam/README.md)

Full performance campaigns remain manual; the old harnesses are archived. The Apple Silicon benchmark runner remains deferred.

Experimental software. [Apache 2.0](LICENSE).
