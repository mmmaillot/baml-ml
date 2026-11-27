# Fitted random forests (schema v4)
The v1 linear/logistic, v2 tree and future v3 (arbitrary sequential pipelines,
#73) artifact schemas and the batch output schema are unchanged. The forest
artifact uses the same mean imputer and standard scaler contract as v2, plus a
shared `n_features`, sorted integer `classes`, and an ordered `trees` array of
`n_estimators >= 1` validated `TreeClassifierState` trees. Every tree must
carry the same `n_features` and class order as the forest header; topologies,
dimensions, leaf probability sums and class orders are validated once at
load/prepare time (`ForestStateError` on top of the existing `TreeStateError`).
Malformed forests, multi-output trees (leaf width != class count) and
non-integer classes are rejected, never silently approximated.

Only dense single-output `RandomForestClassifier` is supported. Preprocessed
dense features are cast to binary32 once per row (`TreeRoundFloat32`), overflow
is rejected, and every tree is traversed with that shared rounded row. Following
`scikit-learn 1.8.0`, `predict_proba` is the arithmetic mean of the normalized
leaf class probabilities accumulated in estimator order and divided once by
`n_estimators`; `predict` is the argmax of that mean with first-class tie
breaking. The accumulation order matches sklearn's sequential
`_accumulate_prediction`, so parity is bit-exact, verified by the independent
oracle probe (`ForestOracleProbe`, exact `assert.equal` on labels, probabilities
and per-tree leaf IDs).

Semantic mode follows the existing metadata-only policy. Full records the
artifact once at load (`ForestModelLoaded`) and a single batch span
`ObservedForestTraversal` with averaged probabilities, labels and compact
per-row leaf IDs per tree; joining the leaf IDs with the load record and scaled
inputs reconstructs any tree path, split feature and threshold. No per-tree or
per-node spans are emitted. Timing and engine-off retain the existing contract.
`ObservationMode.Adaptive` (lazy snapshot + selective row replay), standalone
`predict` and same-family `shadow` (`random_forest_classifier` vs
`random_forest_classifier`) work through the shared `SelectiveModel` path. The
committed `forest-smoke` fixture is synthetic; actual sklearn pipelines and an
independent prediction oracle are generated in CI.
