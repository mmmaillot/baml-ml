// #74: extended SimpleImputer strategies and native dense OneHotEncoder.
function V3CatFixture(name: string) -> string {
  baml.fs.read("artifacts/fixtures/v3/" + name + ".json")
}

function CatHandleError(m: LoadedPipelineModel, raw: string) -> string {
  let input = baml.json.decode<BatchInputV3>(raw);
  m.predict(BatchCells(input), input.features).error
}

test "v3_one_hot_error_on_unseen_category" {
  let m = LoadPipelineModel("artifacts/fixtures/v3/v3-ohe-error.json");
  let f = baml.json.stringify(["color", "size"]);
  assert.equal(CatHandleError(m, `{"features":${f},"rows":[["red","S"]]}`), "")
  let r = CatHandleError(m, `{"features":${f},"rows":[["purple","S"]]}`);
  assert.equal(r, "one_hot_encoding: unseen category in column 0")
  // Integer-coded categories compare numerically.
  let mi = LoadPipelineModel("artifacts/fixtures/v3/v3-ohe-int-categories.json");
  let fi = baml.json.stringify(["code_a", "code_b"]);
  assert.equal(CatHandleError(mi, `{"features":${fi},"rows":[[1,2]]}`), "")
  assert.equal(CatHandleError(mi, `{"features":${fi},"rows":[[9,2]]}`), "one_hot_encoding: unseen category in column 0")
  // A number in a string-category column is an unseen category under error.
  assert.equal(CatHandleError(m, `{"features":${f},"rows":[[5,"S"]]}`), "one_hot_encoding: unseen category in column 0")
  // A string cell in a numeric-only model is rejected at input validation.
  let mn = LoadPipelineModel("artifacts/fixtures/v3/v3-logreg-impute-scale.json");
  let fn = baml.json.stringify(["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]);
  assert.equal(CatHandleError(mn, `{"features":${fn},"rows":[["x",3.0,1.3,0.2]]}`), "string value in numeric column 0")
}

test "v3_string_imputer_replaces_null_cells" {
  let m = LoadPipelineModel("artifacts/fixtures/v3/v3-strimpute-ohe-logreg.json");
  let f = baml.json.stringify(["color", "size"]);
  // Null categorical cells are imputed with the fitted fill value and encoded.
  let r = m.predict_proba(BatchCells(baml.json.decode<BatchInputV3>(`{"features":${f},"rows":[[null,null]]}`)), ["color", "size"]);
  assert.is_true(r.ok)
  // A numeric cell in a string imputer fails in the imputation stage.
  assert.equal(CatHandleError(m, `{"features":${f},"rows":[[1,"S"]]}`), "imputer_transform: numeric value in string imputer")
  let a = DecodePipelineArtifact(V3CatFixture("v3-strimpute-ohe-logreg"));
  let imp = a.steps[0].imputer;
  if (imp != null) { imp.value_kind = "numeric"; }
  assert.is_true(PipelineArtifactError(a) != "")
}

test "v3_rejects_malformed_one_hot_payloads" {
  let base = V3CatFixture("v3-ohe-drop-first");
  assert.equal(PipelineArtifactError(DecodePipelineArtifact(base)), "");
  let a = DecodePipelineArtifact(base);
  let ohe = a.steps[0].one_hot;
  if (ohe != null) { ohe.handle_unknown = "warn"; }
  assert.is_true(PipelineArtifactError(a) != "");
  a = DecodePipelineArtifact(base);
  let ohe2 = a.steps[0].one_hot;
  if (ohe2 != null) { ohe2.drop = "last"; }
  assert.is_true(PipelineArtifactError(a) != "");
  a = DecodePipelineArtifact(base);
  let ohe3 = a.steps[0].one_hot;
  if (ohe3 != null) { ohe3.columns[0].strings = []; }
  assert.is_true(PipelineArtifactError(a) != "");
  a = DecodePipelineArtifact(base);
  let ohe4 = a.steps[0].one_hot;
  if (ohe4 != null) { ohe4.columns[0].strings = ["red", "red"]; }
  assert.is_true(PipelineArtifactError(a) != "");
  a = DecodePipelineArtifact(base);
  a.steps[0].n_features_out = 99;
  assert.is_true(PipelineArtifactError(a) != "");
  a = DecodePipelineArtifact(base);
  a.input_kinds[0] = "numeric";
  assert.is_true(PipelineArtifactError(a) != "");
  a = DecodePipelineArtifact(base);
  let ohe5 = a.steps[0].one_hot;
  if (ohe5 != null) { ohe5.columns[0].kind = "integer"; ohe5.columns[0].ints = [1.5]; ohe5.columns[0].strings = []; }
  assert.is_true(PipelineArtifactError(a) != "");
  assert.is_true(V3LoadFails("{"))
}

test "v3_ohe_ignore_matches_oracle_and_modes_agree" {
  let o = V3Oracle()["v3-ohe-ignore-unseen"];
  let raw_input = baml.json.stringify(o.input);
  let model = V3CatFixture("v3-ohe-ignore-unseen");
  let full = RunBatch(model, raw_input, "predict-proba", ObservationMode.Full);
  let semantic = RunBatch(model, raw_input, "predict-proba", ObservationMode.Semantic);
  assert.equal(full.predictions, semantic.predictions)
  let expected = o.predict_proba;
  let got = full.predictions;
  assert.is_true(expected != null && got is float[][])
  if (expected != null && got is float[][]) {
    assert.is_true(NearMatrix(got, expected))
  }
}
