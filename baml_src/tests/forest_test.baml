// #76: end-to-end heterogeneous pipelines + RandomForestClassifier.
function ForestFixture(name: string) -> string {
  baml.fs.read("artifacts/fixtures/v3/" + name + ".json")
}

test "v3_forest_sklearn_parity_is_covered_by_oracle" {
  let o = PipelineOracle()["v3-ct-hetero-forest"];
  let raw_input = baml.json.stringify(o.input);
  let model = ForestFixture("v3-ct-hetero-forest");
  let probs = RunBatch(model, raw_input, "predict-proba", ObservationMode.Full);
  let labels = RunBatch(model, raw_input, "predict", ObservationMode.Timing);
  let expected = o.predict_proba;
  let expected_labels = o.predict_labels;
  assert.is_true(expected != null && expected_labels != null)
  let got = probs.predictions;
  assert.is_true(got is float[][])
  if (expected != null && got is float[][]) { assert.is_true(NearMatrix(got, expected)) }
  let got_labels = labels.predictions;
  assert.is_true(got_labels is int[])
  if (expected_labels != null && got_labels is int[]) { assert.equal(got_labels, expected_labels) }
}

test "v3_forest_rejects_malformed_estimators_at_load" {
  let base = ForestFixture("v3-forest-impute-scale");
  assert.equal(PipelineArtifactError(DecodePipelineArtifact(base)), "");
  let a = DecodePipelineArtifact(base);
  a.estimator.trees = [];
  assert.is_true(PipelineArtifactError(a) != "");
  a = DecodePipelineArtifact(base);
  a.estimator.trees[0].classes = [9, 9];
  assert.is_true(PipelineArtifactError(a) != "");
  a = DecodePipelineArtifact(base);
  a.estimator.trees[0].n_features = 3;
  assert.is_true(PipelineArtifactError(a) != "");
  a = DecodePipelineArtifact(base);
  a.estimator.kind = "gradient_boosting";
  assert.is_true(PipelineArtifactError(a) != "");
  assert.is_true(PipelineLoadFails(baml.json.stringify(a)));
}

test "v3_forest_adaptive_and_shadow_signals" {
  let champion = ForestFixture("v3-forest-impute-scale");
  let challenger = ForestFixture("v3-forest-impute-scale-b");
  let f = baml.json.stringify(["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]);
  let input = `{"features":${f},"rows":[[null,3.0,1.3,0.2],[6.1,3.0,4.9,1.8]]}`;
  let out = RunShadow(champion, challenger, input, "predict-proba", DefaultAdaptivePolicy());
  assert.equal(out.summary.total_rows, 2)
  assert.equal(out.summary.triggers.missing_input, 1)
  assert.is_true(out.summary.challenger_ok)
  // Champion predictions are returned; forest averaged probabilities match the oracle.
  let probs = out.predictions;
  let o = PipelineOracle()["v3-forest-impute-scale"];
  let expected = o.predict_proba;
  assert.is_true(probs is float[][] && expected != null)
  if (probs is float[][] && expected != null) { assert.is_true(NearMatrix(probs, expected)) }
  // Adaptive mode returns identical predictions to Semantic on the same batch.
  let adaptive = RunBatch(champion, input, "predict-proba", ObservationMode.Adaptive);
  let semantic = RunBatch(champion, input, "predict-proba", ObservationMode.Semantic);
  assert.equal(adaptive.predictions, semantic.predictions)
  // Heterogeneous forest: missing categorical and numeric cells both fire missing_input.
  let ct = ForestFixture("v3-ct-hetero-forest");
  let fc = baml.json.stringify(["age", "income", "city"]);
  let ct_input = `{"features":${fc},"rows":[[null,55000.0,"paris"],[30.0,null,null]]}`;
  let ct_shadow = RunShadow(ct, ct, ct_input, "predict", DefaultAdaptivePolicy());
  assert.equal(ct_shadow.summary.triggers.missing_input, 2)
  assert.equal(ct_shadow.summary.triggered_rows, 2)
}

