// Schema v3 fixtures were fitted independently with sklearn 1.8.0; the oracle
// holds the sklearn predict / predict_proba reference for each batch input.
class V3OracleEntry {
  input BatchInputV3
  is_classifier bool
  predict_labels int[]?
  predict_values float[]?
  predict_proba float[][]?
}

function V3Oracle() -> map<string, V3OracleEntry> {
  baml.json.decode<map<string, V3OracleEntry>>(baml.fs.read("artifacts/fixtures/v3/oracle.json"))
}

function V3Fixture(name: string) -> string {
  baml.fs.read("artifacts/fixtures/v3/" + name + ".json")
}

function V3Names() -> string[] {
  ["v3-linear-nosteps", "v3-logreg-scaler", "v3-logreg-impute-scale", "v3-tree-impute", "v3-linreg-impute"]
}

function Near(a: float, b: float) -> bool { Abs(a - b) < 1e-9 }

function NearMatrix(a: float[][], b: float[][]) -> bool {
  if (a.length() != b.length()) { return false; }
  for (let i = 0; i < a.length(); i += 1) {
    if (a[i].length() != b[i].length()) { return false; }
    for (let j = 0; j < a[i].length(); j += 1) {
      if (!Near(a[i][j], b[i][j])) { return false; }
    }
  }
  true
}

function V3CheckOne(name: string, mode: ObservationMode) -> void {
  let o = V3Oracle()[name];
  let raw_input = baml.json.stringify(o.input);
  let model = V3Fixture(name);
  if (o.is_classifier) {
    let labels = RunBatch(model, raw_input, "predict", mode);
    let probs = RunBatch(model, raw_input, "predict-proba", mode);
    let expected_labels = o.predict_labels;
    let expected_probs = o.predict_proba;
    assert.is_true(expected_labels != null && expected_probs != null)
    let got_labels = labels.predictions;
    let got_probs = probs.predictions;
    assert.is_true(got_labels is int[])
    assert.is_true(got_probs is float[][])
    if (expected_labels != null && got_labels is int[]) {
      assert.equal(got_labels, expected_labels)
    }
    if (expected_probs != null && got_probs is float[][]) {
      assert.is_true(NearMatrix(got_probs, expected_probs))
    }
  } else {
    let values = RunBatch(model, raw_input, "predict", mode);
    let expected = o.predict_values;
    assert.is_true(expected != null)
    let got = values.predictions;
    assert.is_true(got is float[])
    if (expected != null && got is float[]) {
      assert.equal(got.length(), expected.length())
      for (let i = 0; i < got.length(); i += 1) { assert.is_true(Near(got[i], expected[i])) }
    }
  }
}

test "v3_sklearn_parity_all_fixtures_and_modes" {
  for (let name in V3Names()) {
    V3CheckOne(name, ObservationMode.Timing);
    V3CheckOne(name, ObservationMode.Semantic);
    V3CheckOne(name, ObservationMode.Full);
  }
}

test "v3_loaded_handle_modes_agree_and_own_state" {
  let a = DecodePipelineArtifact(V3Fixture("v3-logreg-impute-scale"));
  let o = V3Oracle()["v3-logreg-impute-scale"];
  let cells = BatchCells(o.input);
  let full = LoadPipelineModel("artifacts/fixtures/v3/v3-logreg-impute-scale.json", observation = ObservationMode.Full);
  let semantic = PreparePipelineModel(baml.json.stringify(a), observation = ObservationMode.Semantic);
  let timing = PreparePipelineModel(baml.json.stringify(a), observation = ObservationMode.Timing);
  let r_full = full.predict_proba(cells, o.input.features);
  let r_sem = semantic.predict_proba(cells, o.input.features);
  let r_time = timing.predict_proba(cells, o.input.features);
  assert.is_true(r_full.ok && r_sem.ok && r_time.ok)
  assert.equal(r_full.probs, r_sem.probs)
  assert.equal(r_full.probs, r_time.probs)
  assert.equal(full.predict(cells, o.input.features).labels, semantic.predict(cells, o.input.features).labels)
  // Mutating the caller artifact after preparation must not change the handle.
  a.estimator.weights[0][0] = 1e9;
  a.features[0] = "mutated";
  let again = semantic.predict_proba(cells, o.input.features);
  assert.equal(again.probs, r_sem.probs)
  assert.equal(semantic.predict(cells, o.input.features).labels, r_full.labels)
  assert.is_true(!semantic.predict_proba(cells, ["wrong"]).ok)
}

function V3LoadFails(raw: string) -> bool {
  let m: LoadedPipelineModel? = PreparePipelineModel(raw) catch_all (e) { _ => null };
  m == null
}

test "v3_rejects_malformed_artifacts_at_load" {
  let base = V3Fixture("v3-logreg-impute-scale");
  assert.equal(PipelineArtifactError(DecodePipelineArtifact(base)), "");
  let a = DecodePipelineArtifact(base);
  a.schema_version = 4;
  assert.is_true(V3LoadFails(baml.json.stringify(a)));
  a = DecodePipelineArtifact(base);
  a.steps[0].kind = "mystery";
  assert.is_true(V3LoadFails(baml.json.stringify(a)));
  a = DecodePipelineArtifact(base);
  a.steps[1].n_features_in = 3;
  assert.is_true(V3LoadFails(baml.json.stringify(a)));
  a = DecodePipelineArtifact(base);
  a.estimator.n_features_in = 3;
  assert.is_true(V3LoadFails(baml.json.stringify(a)));
  a = DecodePipelineArtifact(base);
  let sc = a.steps[1].scaler;
  if (sc != null) { sc.scales[0] = 0.0; }
  assert.is_true(V3LoadFails(baml.json.stringify(a)));
  a = DecodePipelineArtifact(base);
  let imp = a.steps[0].imputer;
  if (imp != null) { imp.statistics[0] = float.nan(); }
  // NaN has no JSON form; validate the in-memory artifact directly.
  assert.is_true(PipelineArtifactError(a) != "");
  a = DecodePipelineArtifact(base);
  let imp2 = a.steps[0].imputer;
  if (imp2 != null) { imp2.strategy = "median"; }
  assert.is_true(V3LoadFails(baml.json.stringify(a)));
  a = DecodePipelineArtifact(base);
  a.features[1] = a.features[0];
  assert.is_true(V3LoadFails(baml.json.stringify(a)));
  a = DecodePipelineArtifact(base);
  a.features[1] = "";
  assert.is_true(V3LoadFails(baml.json.stringify(a)));
  a = DecodePipelineArtifact(base);
  a.estimator.kind = "random_forest_classifier";
  assert.is_true(V3LoadFails(baml.json.stringify(a)));
  a = DecodePipelineArtifact(base);
  a.input_kinds[0] = "categorical";
  assert.is_true(V3LoadFails(baml.json.stringify(a)));
  assert.is_true(V3LoadFails("{"))
}

function V3HandleError(m: LoadedPipelineModel, raw: string) -> string {
  let input = baml.json.decode<BatchInputV3>(raw);
  m.predict(BatchCells(input), input.features).error
}

test "v3_input_validation_rejections" {
  let m = LoadPipelineModel("artifacts/fixtures/v3/v3-logreg-impute-scale.json");
  let f = baml.json.stringify(["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]);
  assert.equal(V3HandleError(m, `{"features":${f},"rows":[["x",3.0,1.3,0.2]]}`), "string value in numeric column 0")
  assert.equal(V3HandleError(m, `{"features":${f},"rows":[[1.0,2.0]]}`), "feature count differs from artifact")
  assert.equal(V3HandleError(m, `{"features":${f},"rows":[[1.0,2.0,3.0,4.0],[1.0]]}`), "matrix is ragged: rows have different lengths")
  assert.equal(V3HandleError(m, `{"features":${f},"rows":[]}`), "matrix has no rows")
  assert.equal(V3HandleError(m, `{"features":["a","b","c","d"],"rows":[[null,3.0,1.3,0.2]]}`), "artifact: feature order mismatch")
  assert.equal(m.predict(NumericCells([[float.inf(),3.0,1.3,0.2]]), ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]).error, "non-finite value in numeric column 0")
  // A missing value without an imputer is rejected by the scaler step...
  let scaler_only = LoadPipelineModel("artifacts/fixtures/v3/v3-logreg-scaler.json");
  assert.equal(V3HandleError(scaler_only, `{"features":${f},"rows":[[null,3.0,1.3,0.2]]}`), "matrix contains NaN or infinite values")
  // ...or reaches the estimator directly in a 0-step pipeline and is rejected there.
  let lin0 = LoadPipelineModel("artifacts/fixtures/v3/v3-linear-nosteps.json");
  let lin0_features = baml.json.stringify(V3Oracle()["v3-linear-nosteps"].input.features);
  assert.equal(V3HandleError(lin0, `{"features":${lin0_features},"rows":[[null,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]]}`), "pipeline: missing value reached the estimator without imputation")
  // predict-proba on a linear v3 model is rejected.
  let lin = LoadPipelineModel("artifacts/fixtures/v3/v3-linear-nosteps.json");
  let lin_o = V3Oracle()["v3-linear-nosteps"];
  assert.equal(lin.predict_proba(BatchCells(lin_o.input), lin_o.input.features).error, "predict-proba requires a classifier estimator")
}

test "v3_adaptive_triggers_and_shadow" {
  let model = V3Fixture("v3-logreg-impute-scale");
  let f = baml.json.stringify(["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]);
  let input = `{"features":${f},"rows":[[null,3.0,1.3,0.2],[6.1,3.0,4.9,1.8]]}`;
  let out = RunShadow(model, model, input, "predict-proba", DefaultAdaptivePolicy());
  assert.equal(out.summary.total_rows, 2)
  assert.equal(out.summary.triggers.missing_input, 1)
  assert.equal(out.summary.triggered_rows, 1)
  assert.is_true(out.summary.challenger_ok)
  let adaptive = RunBatch(model, input, "predict-proba", ObservationMode.Adaptive);
  let semantic = RunBatch(model, input, "predict-proba", ObservationMode.Semantic);
  assert.equal(adaptive.predictions, semantic.predictions)
  // A scaler-less v3 pipeline never fires extreme_zscore, even for large inputs.
  let tree = V3Fixture("v3-tree-impute");
  let big = `{"features":${f},"rows":[[1e6,3.0,1.3,0.2]]}`;
  let tree_shadow = RunShadow(tree, tree, big, "predict", DefaultAdaptivePolicy());
  assert.equal(tree_shadow.summary.triggers.extreme_zscore, 0)
  assert.equal(tree_shadow.summary.triggers.missing_input, 0)
  // Same-family requirements: schema version, estimator kind, features, classes.
  let bad_kind: ShadowOutput? = RunShadow(model, V3Fixture("v3-tree-impute"), input, "predict", DefaultAdaptivePolicy()) catch_all (e) { _ => null };
  assert.equal(bad_kind, null)
  let bad_version: ShadowOutput? = RunShadow(model, baml.fs.read("artifacts/fixtures/iris.json"), input, "predict-proba", DefaultAdaptivePolicy()) catch_all (e) { _ => null };
  assert.equal(bad_version, null)
}

// CI oracle probe: regenerated fixtures are checked against the sklearn oracle
// in a fresh project (see .github/workflows/imported-inference.yml).
function PipelineV3OracleProbe() -> int {
  let count = 0;
  for (let name in V3Names()) {
    V3CheckOne(name, ObservationMode.Timing);
    count += 1;
  }
  count
}
