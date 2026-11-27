# Scikit-learn fitted artifact v1 (#46)

The training environment exports data only; BAML performs inference. No
pickle/joblib execution, optimizer, or training dependency in the runtime.

## Supported scope

- Dense float64, single-target LinearRegression.
- LogisticRegression with sorted signed integer classes: binary sigmoid
  (one coefficient row) or multinomial softmax (one row per class).
  OvR/liblinear and legacy explicit binary multinomial semantics are rejected.
- Exact three-step pipeline: mean SimpleImputer (NaN, no indicators),
  StandardScaler (center and scale), then supported estimator. No feature
  expansion, dropped columns, sparse or multi-output support.
- Only sklearn 1.8.0 is presently used for independent fixtures. Other
  versions require compatibility validation. Unsupported variants must
  be rejected, not silently approximated.

The JSON contract has schema_version, source, source_version, model_id,
ordered feature names, preprocessing and estimator kind, means, scales,
weights, intercepts and ordered classes. All parameters must be finite.
For binary logits, map a single score z to [0,z] so softmax gives sigmoid.
Imported labels use decision scores, not rounded probabilities; ties select
the first class. Overflow is rejected. Numerical equivalence uses tolerances,
not bitwise equality across backends.

model_id is an exporter SHA256 of sorted compact JSON without model_id.
It is provenance, not a signature; the loader currently validates nonempty
identity, not the digest or origin. Feature names passed to the prediction
API must equal artifact order; the caller must correctly associate values
with names. For arrays trained without names, exporter names are caller
provided. Loading and validation are separate from inference. Prefer LoadLinearModel or
LoadLogisticModel: decode, validate and normalize once, then reuse the model's
callable prediction methods for each request. PrepareLinearModel and
PrepareLogisticModel accept raw JSON strings. Invalid artifacts fail during
preparation, not during the first prediction. Feature order, input shape,
non-finite inputs and arithmetic overflow remain checked per request.

BAML classes and arrays are mutable reference values. The factories decode
fresh data and return closures owning the validated state; no captured model
parameters, pipeline, or feature arrays are returned. The callable fields of
the returned handle can themselves be reassigned, but do not expose captured
arrays. This is API-level isolation, not a language-wide immutable type or
security boundary: internal trusted helpers are not supported public APIs.
The existing checked stateless PredictImported* wrappers remain available for
caller-owned mutable artifacts. Handles are live within a single BAML engine
execution; cross-process persistence and host SDK/packaging are not provided.

## Export and run

In the sklearn training environment, import export_sklearn.write_artifact,
then call write_artifact(fitted_pipeline, feature_names, output_path).

To regenerate independent references (manual):

```sh
python -m pip install scikit-learn==1.8.0
python migration/artifacts/generate_fixtures.py /tmp/generated-artifacts
python migration/artifacts/check_export.py /tmp/generated-artifacts
```

The committed fixtures use seed 42, 80/20 split (stratified for classification),
one missing training cell and two held-out rows (one missing, one complete).
The generator also creates clean-training artifacts, not yet committed.
References were independently computed on sklearn 1.8.0; CI regenerates
them through the actual exporter and compares numerical values. The native
tests read the committed JSON files and compare prediction values. The trace
probe checks intermediate values, model identity, hierarchy and predictions
for all three cases against independent references.

From migration, the preferred API is:

```baml
let model = LoadLogisticModel("artifacts/fixtures/iris.json");
let names = ["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"];
let single = model.predict_proba([[5.1,3.5,1.4,0.2]],names);
let batch = model.predict_proba([[5.1,3.5,1.4,0.2],[6.1,3.0,4.9,1.8]],names);
let labels = model.predict([[5.1,3.5,1.4,0.2]],names);
```

For linear regression use LoadLinearModel and model.predict(rows,names). Native example probes: ImportedIrisProbe, ImportedDiabetesProbe,
ImportedBinaryProbe.

```sh
cd migration
"$BAML_CLI" test --max-concurrency 1 --test-timeout 5m
cd ..
bash experiments/imported_inference/run.sh /tmp/imported-inference
```

[CI run 45940061227](https://github.com/mmmaillot/baml-ml/actions/runs/45940061227) validated all three imported reference models, the exporter and semantic trace reconstruction before cleanup. Full differential and performance campaigns remain manual. No packaging or imported-runtime benchmark claim.

The repeated-call profiler check requires one decode, artifact validation and conversion for three predictions in the same execution. This is a lifecycle test, not a performance benchmark.
