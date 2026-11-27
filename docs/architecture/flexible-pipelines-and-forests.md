# ADR: flexible pipelines (schema v3) and random forests

Status: accepted, 2026-11-27. Implements #73, #74, #75 and #76 (re-landing
the reverted #72 RandomForestClassifier work as a v3 terminal estimator).
Supersedes the rigid three-step `SimpleImputer -> StandardScaler -> Estimator`
restriction of v1/v2, which remain 100% unchanged.

## Decision

Introduce a backward-compatible `schema_version = 3` data-only artifact for
arbitrary-length sequential pipelines:

```
features: string[]                 // ordered input columns
input_kinds: string[]              // per column "numeric" | "categorical"
steps: [                           // 0..N typed preprocessing steps, ordered
  { kind: "simple_imputer", n_features_in, n_features_out,
    imputer: { strategy, value_kind, statistics, fill_strings } }
  { kind: "standard_scaler", n_features_in, n_features_out, scaler: { means, scales } }
  { kind: "one_hot_encoder", n_features_in, n_features_out,
    one_hot: { columns: [{ kind: "string"|"integer", strings, ints }], handle_unknown, drop } }
  { kind: "column_transformer", n_features_in, n_features_out,
    column_transformer: { branches: [{ name, columns, mode, steps }], remainder, remainder_columns } }
]
estimator: { kind, n_features_in, weights, intercepts, classes, nodes, trees }
```

- Estimator kinds: `linear`, `logistic_binary`, `logistic_multinomial`,
  `decision_tree_classifier`, `random_forest_classifier`.
- Heterogeneous batch rows are `(float | string | null)[][]`, decoded into a
  unified cell matrix and validated per column kind at request time
  (`InferenceFailure { stage: "input_validation" }` on type mismatch,
  non-finite float, wrong order/count). Numeric-only JSON batches remain fully
  compatible with v1/v2 and all-numeric v3 pipelines.
- The per-step `n_features_in -> n_features_out` chain (including the terminal
  estimator) is validated once at prepare/load time; column kinds are tracked
  statically through the chain so a string can never reach an estimator.
- RandomForestClassifier reuses the validated binary32 tree core:
  features are rounded to float32 once per row, `predict_proba` is the
  arithmetic mean of per-tree leaf probabilities accumulated in estimator
  order (bit-exact sklearn 1.8.0 parity), `predict` is the argmax with
  first-class tie breaking.

## Strict rejection over approximation

Unsupported variants are rejected at export or load, never approximated:
extended imputer configurations outside `NaN`/`add_indicator=false`, sparse
one-hot/column-transformer outputs, `min_frequency`/`max_categories`/
`infrequent_if_exist` grouping, non-`error`/`ignore` `handle_unknown`, drop
other than `null`/`first`/`if_binary`, `transformer_weights`, nested column
transformers, callable/regex column selectors, multi-output forests,
non-integer classes, OvR/liblinear logistic, and any dimension-chain
mismatch.

## Observation contract for v3

Each executed preprocessing step emits its semantic span tagged with
`baml_ml.step_index` and `baml_ml.step_kind`, with fitted step state wrapped in
`@trace.redact`: `ObservedImputeStep`, `ObservedScaleStep`,
`ObservedOneHotEncode`, and `ObservedColumnTransformer` with per-branch child
spans `ObservedColumnTransformerBranch` carrying `baml_ml.branch_name` (branch
sub-steps keep their step-tagged spans and inherit the branch metadata).
Terminal estimators reuse `ObservedLinear`, `ObservedDecisionScores` /
`ObservedSoftmax` / `ObservedLabels`, `ObservedTreeTraversal` and the single
batch span `ObservedForestTraversal`. All modes (`Timing`, `Semantic`, `Full`,
`Adaptive`, `BAML_TELEMETRY=off`) and same-family `shadow` (matching
`estimator_kind`, ordered `features`, ordered `classes`, equal
`schema_version`) are supported. `Adaptive` fires `missing_input` on null
numeric or categorical cells and evaluates `extreme_zscore` on the output of
the most recent `standard_scaler` step or branch (cleanly no-op when no
scaler is present); `low_margin`, `label_disagreement` and `score_delta` work
unchanged on forest averaged probabilities. Failure stages:
`input_validation`, `imputation`, `standardization`, `one_hot_encoding`,
`column_transform`, `linear_prediction`, `decision_scores`, `softmax`,
`tree_traversal`, `forest_traversal`.

## Evidence

- sklearn 1.8.0 parity fixtures + oracle: `artifacts/fixtures/v3/` (21
  artifacts covering 0/1/multi-step pipelines, all imputer strategies, OHE
  drop/handle_unknown variants, integer categories, heterogeneous
  ColumnTransformer pipelines, forests), regenerated in CI through the real
  exporter (`PipelineV3OracleProbe`).
- BTEL SQL verification: `experiments/heterogeneous_pipelines/run.sh`.
- Standalone offline smoke (3 platforms): `experiments/standalone/run.sh`.
- Measured storage/latency campaign: `experiments/heterogeneous_pipelines/evidence/report.json`.

## Consequences

- v1/v2 artifacts, loaders, `BatchOutput` and `ShadowOutput` are unchanged;
  dispatch is on `schema_version`.
- `SelectiveModel.run_fn` now passes the cell matrix; v1/v2 selective paths
  convert to `float[][]` with strict rejection of string cells.
- `ShadowInput.rows` widened to `(float | string | null)[][]`; numeric-only
  shadow inputs decode unchanged.
