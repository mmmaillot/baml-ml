# First tree traversal experiment (#62)

This is the first step towards supporting DecisionTreeClassifier, not a
new supported CLI estimator. The core accepts fitted finite dense data
and integer classes. The data-only artifact schema, preprocessing pipeline,
prepared-state lifecycle and observation contract integration are deferred
to #63.

scikit-learn 1.8 converts input features to float32 before comparing them
with the fitted double-precision thresholds
([reference](https://scikit-learn.org/1.8/modules/generated/sklearn.tree.DecisionTreeClassifier.html)).
BAML floats are binary64. The reference implementation explicitly rounds
to binary32 using power-of-two scaling and nearest-even rounding, including
subnormal and overflow handling. A naïve binary64 threshold comparison
would choose the wrong leaf near a split. No native extension is necessary
for the initial correctness experiment; no speed claim.

Topology validation requires parent-before-child node IDs, one parent for
each nonroot node, valid split features and normalized finite leaf values.
This excludes cycles, shared nodes and unreachable nodes. The batch returns
labels, probabilities, leaves and node paths. The current checked function
validates each call; production preparation and capture-policy allocation
will be dealt with separately. No native missing-value branch support yet.

The automatic lightweight integration check generates actual sklearn 1.8
oracles for multiclass, weighted binary, a constant stump and a boundary
tree. It compares all labels, leaf IDs and visited node paths exactly, and
probabilities within 1e-12. NumPy provides independent binary32 rounding
cases including neighbors, ties, extrema and random exponents. These are
functional correctness checks, not performance benchmarks.
