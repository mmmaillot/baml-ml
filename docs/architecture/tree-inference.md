# Fitted decision trees (schema v2)
The v1 linear/logistic artifact and batch output schema are unchanged. The tree
artifact uses the same mean imputer and standard scaler, integer sorted classes,
and a flat nodes array indexed by sklearn node ID. Leaves store normalized class
probabilities; internal nodes store feature index, threshold and two child IDs.
Only single-output dense DecisionTreeClassifier is supported. Input is rounded
to binary32 after preprocessing. The schema tag is read for dispatch, then
the fitted tree is decoded and validated once per prepared model. The checked
public traversal remains available. No forests or training in the runtime.

Semantic mode follows the existing metadata-only policy. Full records the
artifact at load and one batch traversal span with ordered node IDs, leaf IDs,
labels and probabilities; split feature and threshold are reconstructible by
joining node IDs with the load record. It does not emit full per-node spans.
Timing and engine-off retain the existing contract. No production privacy or
retention guarantee. The committed tree-smoke fixture is synthetic; an actual
sklearn pipeline and independent prediction oracle are generated in CI.
