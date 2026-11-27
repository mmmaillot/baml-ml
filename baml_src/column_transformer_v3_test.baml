// #75: ColumnTransformer on heterogeneous tabular inputs.
function CTFixture(name: string) -> string {
  baml.fs.read("artifacts/fixtures/v3/" + name + ".json")
}

test "v3_ct_input_validation_on_heterogeneous_rows" {
  let m = LoadPipelineModel("artifacts/fixtures/v3/v3-ct-hetero-logreg.json");
  let f = baml.json.stringify(["age", "income", "city"]);
  let err = (raw: string) -> string {
    let input = baml.json.decode<BatchInputV3>(raw);
    m.predict(BatchCells(input), input.features).error
  };
  // String in a numeric column.
  assert.equal(err(`{"features":${f},"rows":[["old",50000.0,"paris"]]}`), "string value in numeric column 0")
  // Wrong column count and ragged rows.
  assert.equal(err(`{"features":${f},"rows":[[1.0,2.0]]}`), "feature count differs from artifact")
  assert.equal(err(`{"features":${f},"rows":[[1.0,2.0,"paris"],[1.0]]}`), "matrix is ragged: rows have different lengths")
  // Feature order mismatch.
  assert.equal(err(`{"features":["city","age","income"],"rows":[["paris",30.0,50000.0]]}`), "artifact: feature order mismatch")
  // Missing numeric and categorical cells are both accepted and imputed.
  let r = m.predict_proba(BatchCells(baml.json.decode<BatchInputV3>(`{"features":${f},"rows":[[null,null,null]]}`)), ["age", "income", "city"]);
  assert.is_true(r.ok)
}

test "v3_ct_rejects_malformed_transformers_at_load" {
  let base = CTFixture("v3-ct-hetero-logreg");
  assert.equal(PipelineArtifactError(DecodePipelineArtifact(base)), "");
  let a = DecodePipelineArtifact(base);
  let ct = a.steps[0].column_transformer;
  assert.is_true(ct != null)
  if (ct != null) {
    // Overlapping branch columns.
    ct.branches[1].columns = [1];
    assert.is_true(PipelineArtifactError(a) != "");
  }
  a = DecodePipelineArtifact(base);
  ct = a.steps[0].column_transformer;
  if (ct != null) { ct.branches[0].columns = [5]; }
  assert.is_true(PipelineArtifactError(a) != "");
  a = DecodePipelineArtifact(base);
  ct = a.steps[0].column_transformer;
  if (ct != null) { ct.branches[0].mode = "explode"; }
  assert.is_true(PipelineArtifactError(a) != "");
  a = DecodePipelineArtifact(base);
  ct = a.steps[0].column_transformer;
  if (ct != null) { ct.remainder = "passthrough"; ct.remainder_columns = [2]; }
  assert.is_true(PipelineArtifactError(a) != ""); // categorical passthrough remainder
  a = DecodePipelineArtifact(base);
  a.steps[0].n_features_out = 99;
  assert.is_true(PipelineArtifactError(a) != "");
  // Nested column transformer inside a branch is rejected.
  a = DecodePipelineArtifact(base);
  ct = a.steps[0].column_transformer;
  if (ct != null) {
    let nested = PipelineStep { kind: "column_transformer", n_features_in: 1, n_features_out: 1, imputer: null, scaler: null, one_hot: null, column_transformer: ct };
    ct.branches[0].steps.push(nested);
    ct.branches[0].steps[ct.branches[0].steps.length() - 1].n_features_out = 2;
  }
  assert.is_true(PipelineArtifactError(a) != "");
  // Passthrough branch on a categorical column is rejected.
  a = DecodePipelineArtifact(base);
  ct = a.steps[0].column_transformer;
  if (ct != null) {
    ct.branches[1].mode = "passthrough";
    ct.branches[1].steps = [];
  }
  assert.is_true(PipelineArtifactError(a) != "");
}

test "v3_ct_adaptive_missing_input_fires_on_numeric_and_categorical_nulls" {
  let model = CTFixture("v3-ct-hetero-logreg");
  let f = baml.json.stringify(["age", "income", "city"]);
  let input = `{"features":${f},"rows":[[null,55000.0,"paris"],[30.0,null,null],[45.0,60000.0,"berlin"]]}`;
  let out = RunShadow(model, model, input, "predict-proba", DefaultAdaptivePolicy());
  assert.equal(out.summary.triggers.missing_input, 2)
  assert.equal(out.summary.triggered_rows, 2)
  assert.is_true(out.summary.challenger_ok)
  let adaptive = RunBatch(model, input, "predict-proba", ObservationMode.Adaptive);
  let semantic = RunBatch(model, input, "predict-proba", ObservationMode.Semantic);
  assert.equal(adaptive.predictions, semantic.predictions)
  // extreme_zscore fires on the scaled numeric branch output.
  let big = `{"features":${f},"rows":[[1e9,55000.0,"paris"]]}`;
  let big_shadow = RunShadow(model, model, big, "predict", DefaultAdaptivePolicy());
  assert.equal(big_shadow.summary.triggers.extreme_zscore, 1)
}
