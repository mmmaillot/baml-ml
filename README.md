# baml-ml

A hyperobservable inference runtime for models trained with scikit-learn, built in [BAML](https://boundaryml.com) (*the hyper-observable programming language*).

Export fitted pipelines as portable, versioned, data-only JSON artifacts; load and execute them natively in a standalone BAML executable. Reconstruct individual predictions from preprocessing to output using native SQL queries over local BTEL recordings. Training stays outside the runtime. No Python at inference time, no pickle, no LLMs, and no API keys required.

> **Observability should be a property of the runtime, not a responsibility of the developer.**

As generated code and model fleets grow, consistent runtime observation complements tests and code review. `baml-ml` tests this thesis on classical ML: semantic traces, parameter redaction (`@trace.redact`), selective row-level replay, and champion/challenger shadow diffs are built into the runtime with zero telemetry SDK or collector.

## Try it in 30 seconds

Download a prebuilt standalone executable from [GitHub Releases](https://github.com/mmmaillot/baml-ml/releases/latest) (`linux-x86_64`, `linux-arm64`, or `macos-arm64`) and run it on the committed Iris fixture:

```sh
cat <<'EOF' > batch.json
{
  "features": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
  "rows": [[null, 3.0, 1.3, 0.2], [6.1, 3.0, 4.9, 1.8]]
}
EOF

# Adaptive mode: runs the batch in lightweight Semantic mode and replays only
# anomalous/uncertain rows (here row 0, with a missing sepal length) in Full.
./baml-ml predict \
  --model artifacts/fixtures/iris.json \
  --input batch.json \
  --operation predict-proba \
  --observation adaptive \
  --output-format json

# Shadow mode: compares two same-family models, returns champion predictions on
# stdout, and captures stage-by-stage Full traces for disagreements or anomalies.
./baml-ml shadow \
  --champion artifacts/fixtures/iris.json \
  --challenger artifacts/fixtures/iris.json \
  --input batch.json \
  --operation predict-proba \
  --output-format json
```

Real `shadow` output on stdout (champion predictions + compact audit summary):

```json
{
  "schema_version": 1,
  "operation": "predict-proba",
  "estimator_kind": "logistic_multinomial",
  "champion_model_id": "b353ccaf343fff703e6c7084d3e87d89f41693224caea74a690dc662a7889e89",
  "challenger_model_id": "b353ccaf343fff703e6c7084d3e87d89f41693224caea74a690dc662a7889e89",
  "classes": [0, 1, 2],
  "predictions": [
    [0.8217064033386522, 0.17829002427778534, 3.5723835624939424e-6],
    [0.0038088445397161125, 0.37263701903431495, 0.6235541364259689]
  ],
  "summary": {
    "total_rows": 2,
    "triggered_rows": 1,
    "escalated_rows": 1,
    "capped": false,
    "label_disagreements": 0,
    "max_score_delta": 0.0,
    "triggers": {
      "missing_input": 1,
      "extreme_zscore": 0,
      "low_margin": 0,
      "label_disagreement": 0,
      "score_delta": 0
    },
    "challenger_ok": true,
    "challenger_error": null
  }
}
```

Inspect stage-by-stage differences between champion and challenger directly from local BTEL recordings using [`semantic-diff.sql`](experiments/adaptive_shadow/semantic-diff.sql) with `baml-cli local recordings query`:

```sql
SELECT escalation.context_metadata['baml_ml.row_index'] AS row_index,
       stage.span_name AS stage,
       stage.context_metadata['baml_ml.shadow_role'] AS role,
       stage.context_metadata['baml_ml.model_id'] AS model_id,
       stage.input_args AS inputs,
       stage.output_value AS output
FROM spans stage
JOIN spans replay ON stage.parent_span_id = replay.span_id
JOIN spans escalation ON replay.parent_span_id = escalation.span_id
WHERE escalation.span_name = 'user.ShadowEscalatedRow'
  AND replay.span_name = 'user.RunSelectiveModel'
  AND stage.span_name IN ('user.ObservedImpute', 'user.ObservedScale',
      'user.ObservedLinear', 'user.ObservedDecisionScores', 'user.ObservedSoftmax',
      'user.ObservedLabels', 'user.ObservedTreeTraversal')
ORDER BY row_index, stage, role
```

## How native hyperobservability looks in BAML

Fitted model state is redacted declaratively on inference spans (`@trace.redact`) and captured once in `ModelLoaded` / `TreeModelLoaded` (lazily in `Adaptive` and `shadow`). Policies switch cleanly via native `trace.Options` ([`baml_src/observation.baml`](baml_src/observation.baml)):

```baml
class TracePrivate<T> {
  value: T @trace.redact
}

function ObservationOptions(mode: ObservationMode) -> trace.Options {
  if (mode == ObservationMode.Full) { return trace.span(); }
  if (mode == ObservationMode.Semantic || mode == ObservationMode.Adaptive) { return trace.empty_span(); }
  trace.timing()
}

function ObservedScale(x: float[][], state: TracePrivate<StandardScaler>) -> ScalerTransformResult throws InferenceFailure {
  let r = ScalerTransformCore(x, state.value, true, validated_state = true);
  if (!r.ok) { throw InferenceFailure { stage: "standardization", message: r.error }; }
  r
}
```

## Supported scope (`v0.2.0`)

Validated against **scikit-learn 1.8.0** for dense `float64` three-step `Pipeline`s (`SimpleImputer(strategy="mean")` → `StandardScaler` → estimator):

- **`LinearRegression`** (schema v1): single-target dense regression (`predict`).
- **`LogisticRegression`** (schema v1): binary sigmoid and multinomial softmax with sorted integer classes (`predict`, `predict-proba`).
- **`DecisionTreeClassifier`** (schema v2): single-output dense classification with sorted integer classes and exact IEEE-754 `binary32` ties-to-even threshold rounding (`predict`, `predict-proba`, visited paths and leaves in `Full`).
- **Observation modes**: `Semantic` (default metadata-only spans), `Timing` (aggregate profiler), `Full` (all stage payloads + one-time model snapshot), `Adaptive` (selective row-level `Full` replay with lazy snapshot and `max_escalated_rows` cap), and `BAML_TELEMETRY=off`.
- **Standalone CLI**: `predict` and same-family `shadow` (champion/challenger comparison with challenger inference-failure isolation).

See the [artifact contract](artifacts/README.md), [tree inference ADR](docs/architecture/tree-inference.md), [observation contract v1](docs/architecture/observation-contract.md), and [adaptive & shadow ADR](docs/architecture/adaptive-shadow-observation.md) for exact restrictions and export instructions. Unsupported variants are explicitly rejected rather than silently approximated.

## Benchmarks

### Batch and edge latency vs scikit-learn

Median latency on a MacBook Pro (Apple M1 Pro, 32 GB), using the same fitted Iris multinomial model and identical JSON batches for `predict-proba` (15 measured runs after warmup).

| Measurement | 1 row | 128 rows | 4,096 rows |
|---|---:|---:|---:|
| BAML process, telemetry off | 32.0 ms | 33.2 ms | 64.7 ms |
| BAML process, default semantic mode | 110.6 ms | 111.7 ms | 148.9 ms |
| Python process, scikit-learn | 898.3 ms | 895.0 ms | 895.4 ms |
| BAML loaded model, telemetry off | 0.009 ms | 0.739 ms | 23.388 ms |
| scikit-learn loaded model | 0.386 ms | 0.406 ms | 0.728 ms |

Process timings include startup, model loading, JSON I/O, and shutdown. Loaded timings cover inference after loading and decoding. When each batch launches a process, BAML finished **6× to 28× faster** than Python + scikit-learn with half the peak RSS. With both models already loaded in memory, BAML was faster at 1 row and scikit-learn was faster at 128 and 4,096 rows. See the [full standalone benchmark report](experiments/standalone_benchmark/m1-pro-2026-11-27.md).

### Selective observability storage vs latency trade-off

Median measurements on Apple M1 Pro for a packed 4,096-row Iris `predict-proba` batch (7 fresh processes per case after warmup; `max_escalated_rows = 64`):

| Batch | Mode | Replayed rows | BTEL bytes | CAS bytes | Peak RSS | Wall time |
|---|---|---:|---:|---:|---:|---:|
| healthy (0 anomalies) | `semantic` | 0 | 19,088 | 797 | 56.7 MiB | 151.7 ms |
| healthy (0 anomalies) | `adaptive` | 0 | 28,224 | 797 | 56.9 MiB | 159.1 ms |
| healthy (0 anomalies) | `full` | 0 | 19,395 | 833,271 | 70.4 MiB | 160.3 ms |
| sparse (52 NaNs, 1.27%) | `adaptive` | 52 | 55,672 | 116,621 | 58.3 MiB | 238.2 ms |
| sparse (52 NaNs, 1.27%) | `full` | 0 | 19,392 | 1,058,608 | 70.2 MiB | 159.1 ms |
| shifted (4,096 OOD, capped) | `adaptive` | 64 | 61,185 | 142,593 | 59.0 MiB | 258.2 ms |
| shifted (4,096 OOD, capped) | `shadow` | 64 | 101,987 | 252,158 | 64.9 MiB | 316.8 ms |

On the sparse 4,096-row batch, `Adaptive` used **84% fewer combined BTEL + CAS bytes** than `Full` (172 kB vs 1,078 kB) while keeping healthy batches payload-free (797 B CAS = `Semantic`). Replaying 52–64 individual rows adds ~80–100 ms because each replayed row emits its own stage spans; `max_escalated_rows` bounds both storage and latency. See the [full adaptive & shadow report](experiments/adaptive_shadow/results/m1-pro-2026-11-27/report.md) and [SQL evidence](experiments/adaptive_shadow/results/m1-pro-2026-11-27/sql-evidence.json).

## Standalone delivery

The standalone executable is packaged with `baml pack` (BAML 0.26.0) and offline-validated in CI across **Linux x86-64**, **Linux ARM64**, and **macOS ARM64**. Prebuilt checksummed archives are published on [GitHub Releases](https://github.com/mmmaillot/baml-ml/releases). See [standalone build and usage](experiments/standalone/README.md) and the [standalone-first ADR](docs/architecture/standalone-first.md).

## Run and validate from source

The native runtime is pinned to BAML 0.26.0. Follow the [installation and development instructions](docs/development.md):

```sh
export BAML_CLI=/absolute/path/to/baml-cli
"$BAML_CLI" check
"$BAML_CLI" test --max-concurrency 1 --test-timeout 5m
bash experiments/imported_inference/run.sh /tmp/imported-inference
bash experiments/native_ml_observability/run.sh /tmp/native-ml-observability
```

## Roadmap

1. `RandomForestClassifier`: native ensemble traversal, sklearn probability-averaging parity, and semantic traces ([#72](https://github.com/mmmaillot/baml-ml/issues/72)).
2. Arbitrary-length sequential pipeline contract (`schema_version = 3`) ([#73](https://github.com/mmmaillot/baml-ml/issues/73)).
3. Extended `SimpleImputer` strategies (`median`, `most_frequent`, `constant`) and dense `OneHotEncoder` ([#74](https://github.com/mmmaillot/baml-ml/issues/74)).
4. `ColumnTransformer` with heterogeneous numeric/categorical tabular inputs ([#75](https://github.com/mmmaillot/baml-ml/issues/75)).
5. End-to-end heterogeneous pipelines + `RandomForestClassifier` with `Adaptive` / `shadow`, standalone packaging, and ADR ([#76](https://github.com/mmmaillot/baml-ml/issues/76)).
6. Optional numerical acceleration only if measured inference profiles justify it ([#20](https://github.com/mmmaillot/baml-ml/issues/20), deferred).

[Architecture decision](docs/architecture/inference-first.md).

## Historical checkpoint

The pre-cleanup project (native training, old SDKs, numerical backends, and earlier experiments) is preserved at [pre-inference-cleanup](https://github.com/mmmaillot/baml-ml/tree/pre-inference-cleanup), commit `d87488f68d7ebc4025a9c8199131c4468f0f4982`.

Experimental software. [Apache 2.0](LICENSE).
