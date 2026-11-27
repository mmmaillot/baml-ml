# Adaptive observation and shadow inference

BAML 0.26.0 uses native trace options and local BTEL recordings for selective replay. A loaded model in Adaptive mode first executes the batch with Semantic capture, then replays selected rows with Full capture. The standalone `shadow` command applies the same policy to two compatible pipelines. It returns the champion predictions and an aggregate summary. [Observation contract v1](observation-contract.md)

## Policy

Pass `policy: AdaptivePolicy? = null` to the linear, logistic, tree or forest model loader or preparer. Adaptive mode uses `DefaultAdaptivePolicy()` when no policy is supplied. The loader validates and copies an explicit policy before decoding the model or batch JSON. The other modes reject a supplied policy. The packed `predict --observation adaptive` and `shadow` commands accept a partial JSON override with `--policy path.json`.

| Field | Default | Signal or constraint |
| --- | --- | --- |
| `enable_missing_input` | `true` | The raw row contains NaN, represented by `null` in batch JSON. |
| `enable_extreme_zscore` | `true` | The largest absolute scaled feature is at least `max_abs_z`. |
| `max_abs_z` | `4.0` | Finite and strictly positive. |
| `enable_low_margin` | `true` | The top two class probabilities differ by at most `min_proba_margin`. |
| `min_proba_margin` | `0.2` | Finite and in [0, 1]. |
| `enable_label_disagreement` | `true` | Champion and challenger predict different class labels. |
| `enable_score_delta` | `true` | At least one class probability differs by `max_proba_delta` or more, or regression predictions differ by `max_pred_delta` or more. |
| `max_proba_delta` | `0.15` | Finite and in [0, 1]. |
| `max_pred_delta` | `1.0` | Finite and nonnegative. |
| `max_escalated_rows` | `64` | Nonnegative JSON integer. |

In BAML 0.26.0 both omitted optional class fields and explicit JSON `null` decode to `null`. Both select the default in `AdaptivePolicyOverride`. The policy decoder checks unknown keys separately because native class decoding ignores them. Malformed JSON, incompatible field types and invalid values raise `AdaptivePolicyError`. Numeric thresholds accept JSON numbers, and the row cap requires an integer token.

Shadow combines the single-model signals from either model with label and score differences. Reasons are emitted in fixed order as `missing_input,extreme_zscore,low_margin,label_disagreement,score_delta`. A row contributes at most once to each trigger count. All triggered rows contribute to the counts. Full replay uses the first `min(K, max_escalated_rows)` qualifying rows in input order, with zero-based indices, and `capped` is true exactly when fewer than `K` rows are replayed. Probability thresholds use the model's ordered classes. Logistic label-only inference computes probabilities in memory for evaluation and replays its label stage.

## Standalone response and compatibility

`predict` keeps the existing `BatchOutput` v1 schema. `shadow` has its own v1 response with operation, estimator kind, both model IDs, ordered classes, champion predictions and `summary`. The summary contains total, triggered and escalated row counts, the cap flag, per-trigger counts, label disagreements, maximum score delta, and challenger status and error. Label disagreements and maximum delta are measured across the full successful batch even if the corresponding triggers are disabled. An absolute regression delta beyond float64 range still triggers and is reported as the largest finite float64 so stdout remains valid JSON.

Compatibility checks precede inference. Artifacts must have the same schema and estimator kind, including the binary versus multinomial logistic distinction, and identical ordered features and classes. Batch features must match both models. The ordinary batch shape infers class order from the matched artifacts. A shadow input may also supply an ordered `classes` array, which must match the artifacts when present. `predict-proba` requires a classifier.

## Native span hierarchy

- Adaptive load emits `ModelLoaded`, `TreeModelLoaded` or `ForestModelLoaded` as a metadata-only span. Each prediction call has a metadata-only `RunAdaptive` parent covering feature validation, the `RunSelectiveModel` main pass and any replay. A failed feature check records a failed parent with no numerical payload. The first call on a handle with a row to replay emits one Full load span tagged `baml_ml.lazy_snapshot = true`. Each `EscalatedRow` is a child of its prediction call and groups the Full stage spans for one input row.
- An `AdaptiveSummary` child appears when any row triggers, including with a zero cap. It carries `baml_ml.triggered_rows`, `baml_ml.escalated_rows`, `baml_ml.capped` and `baml_ml.triggers.<reason>` counters for the whole batch.
- `ShadowBatch` is the shadow operation span. It carries `baml_ml.champion_model_id`, `baml_ml.challenger_model_id`, `baml_ml.estimator_kind` and `baml_ml.operation`. It contains metadata-only model loads and main inference spans. The first scheduled replay adds one Full snapshot per model. Each `ShadowEscalatedRow` contains a pair of `RunSelectiveModel` spans with Full `Observed*` children. The two sides share a row index and can be joined by span IDs.
- Row scopes add `baml_ml.row_index`, `baml_ml.trigger_reasons` and `baml_ml.escalated = true`. Model-scoped children inherit the existing identity, schema, source and contract metadata. Shadow scopes also inherit `baml_ml.shadow_role` as `champion` or `challenger`. Row and model replay wrappers omit their own input and output payloads. Full stage spans record the stage data with fitted state redacted. The lazy load span captures the fitted artifact itself.

Healthy batches and batches with a zero cap take no Full snapshots and replay no rows. The challenger failure case below has its own snapshot rule. `BAML_TELEMETRY=off` disables native recording. [Stage-by-stage SQL](../../experiments/adaptive_shadow/semantic-diff.sql)

## Challenger failure isolation

When the champion succeeds and the challenger raises a typed `InferenceFailure` during the main inference pass, shadow returns the champion predictions with `challenger_ok = false` and a `challenger_error` carrying the stage and message. Both snapshots are captured even with a zero cap. A failing Full `ShadowChallengerFailed` span records the typed error with the challenger’s model metadata and `baml_ml.challenger_isolated = true`. The failed comparison returns zero trigger and comparison metrics. The failing batch ran in Semantic capture, so the offending input rows are absent from the replay. The caller can use the status flag to exclude it from comparison statistics.

Invalid policy, artifact, operation or compatibility, champion inference failures and unexpected engine failures propagate as command failures. Isolation is specific to the typed challenger failure after successful champion inference.

The contract is scoped to runtime spans. It provides no global privacy guarantee, automatic drift dashboard or collector, or cross-process retention and persistence engine. Inspect your root/caller tracing and local data handling before enabling replay on production inputs. [Reproduction and measured evidence](../../experiments/adaptive_shadow/README.md)
