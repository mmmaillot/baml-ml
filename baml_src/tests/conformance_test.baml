// Declarative negative conformance suite. Every mutation cataloged in
// artifacts/conformance/invalid_mutations.json must be rejected by the static
// validator with the cataloged message AND at load/prepare time. Non-finite
// floats have no JSON form (the decoder rejects NaN and out-of-range numbers),
// so those mutations are applied in memory at the bottom of this file.

class ConformanceBase {
  id string
  schema_version int
  file string
}

class ConformanceMutation {
  id string
  schema_version int
  file string
  expect_error_contains string
}

class ConformanceCatalog {
  schema int
  bases ConformanceBase[]
  mutations ConformanceMutation[]
}

function LoadConformanceCatalog() -> ConformanceCatalog {
  baml.json.decode<ConformanceCatalog>(baml.fs.read("artifacts/conformance/invalid_mutations.json"))
}

function ConformancePath(file: string) -> string {
  "artifacts/conformance/" + file
}

function Contains(haystack: string, needle: string) -> bool {
  haystack.split(needle).length() > 1
}

function ValidateBySchema(schema_version: int, raw: string) -> string {
  if (schema_version == 1) { return ArtifactError(DecodeFittedArtifact(raw)); }
  if (schema_version == 2) { return TreeArtifactError(DecodeTreeArtifact(raw)); }
  PipelineArtifactError(DecodePipelineArtifact(raw))
}

function LoadBySchemaOk(schema_version: int, raw: string) -> bool {
  if (schema_version == 1) {
    let a: FittedArtifact? = PrepareArtifact(raw, null) catch_all (e) { _ => null };
    return a != null;
  }
  if (schema_version == 2) {
    let a: FittedTreeArtifact? = PrepareTreeArtifact(raw) catch_all (e) { _ => null };
    return a != null;
  }
  let a: FittedPipelineArtifact? = PreparePipelineArtifact(raw) catch_all (e) { _ => null };
  a != null
}

test "conformance_bases_validate_clean_and_load" {
  let catalog = LoadConformanceCatalog();
  assert.equal(catalog.schema, 1)
  assert.is_true(catalog.bases.length() >= 6)
  let failures: string[] = [];
  for (let b in catalog.bases) {
    let raw = baml.fs.read(ConformancePath(b.file));
    let err = ValidateBySchema(b.schema_version, raw);
    if (err != "") { failures.push(b.id + " validator: " + err); }
    if (!LoadBySchemaOk(b.schema_version, raw)) { failures.push(b.id + " rejected at load"); }
  }
  assert.equal(failures, [])
}

test "conformance_invalid_mutations_rejected_at_validate_and_load" {
  let catalog = LoadConformanceCatalog();
  assert.is_true(catalog.mutations.length() >= 30)
  let failures: string[] = [];
  for (let m in catalog.mutations) {
    let raw = baml.fs.read(ConformancePath(m.file));
    let err = ValidateBySchema(m.schema_version, raw);
    if (!Contains(err, m.expect_error_contains)) { failures.push(m.id + " validator: " + err); }
    if (LoadBySchemaOk(m.schema_version, raw)) { failures.push(m.id + " loaded despite invalid mutation"); }
  }
  assert.equal(failures, [])
}

test "conformance_non_finite_parameters_rejected_in_memory" {
  // JSON has no non-finite float form, so these mutations are applied to
  // decoded artifacts directly; IsFinite must reject both NaN and +-inf.
  let v1 = DecodeFittedArtifact(baml.fs.read("artifacts/fixtures/iris.json"));
  v1.weights[0][0] = float.nan();
  assert.equal(ArtifactError(v1), "artifact: non-finite weight")
  let v1b = DecodeFittedArtifact(baml.fs.read("artifacts/fixtures/iris.json"));
  v1b.scaler_scales[0] = float.inf();
  assert.equal(ArtifactError(v1b), "artifact: invalid preprocessing parameters")
  let v1c = DecodeFittedArtifact(baml.fs.read("artifacts/fixtures/iris.json"));
  v1c.imputer_means[0] = float.nan();
  assert.equal(ArtifactError(v1c), "artifact: invalid preprocessing parameters")
  let v3 = DecodePipelineArtifact(baml.fs.read(ConformancePath("bases/v3-synthetic.json")));
  let ct = v3.steps[0].column_transformer;
  if (ct != null) {
    let sc = ct.branches[0].steps[0].scaler;
    if (sc != null) { sc.scales[0] = float.nan(); }
  }
  assert.equal(PipelineArtifactError(v3), "artifact: invalid scaler parameters")
  let v3b = DecodePipelineArtifact(baml.fs.read(ConformancePath("bases/v3-synthetic.json")));
  v3b.estimator.weights[0][0] = float.inf();
  assert.equal(PipelineArtifactError(v3b), "artifact: non-finite weight")
}

test "internal_enums_parse_once_at_wire_boundaries" {
  assert.equal(OperationName(ParseOperation("predict")), "predict")
  assert.equal(OperationName(ParseOperation("predict-proba")), "predict-proba")
  let bad: Operation? = ParseOperation("fit") catch_all (e) { _ => null };
  assert.equal(bad, null)
  assert.is_true(ColumnKindOf("numeric") == ColumnKind.Numeric)
  assert.is_true(ColumnKindOf("categorical") == ColumnKind.Categorical)
  assert.is_true(ColumnKindOf("ordinal") == null)
  assert.equal(NumericColumnKinds(2).length(), 2)
  // Wire JSON keeps plain strings end to end.
  let model = baml.fs.read("artifacts/fixtures/iris.json");
  let input = `{"features":["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"],"rows":[[5.1,3.5,1.4,0.2]]}`;
  assert.equal(RunBatch(model, input, "predict-proba", ObservationMode.Timing).operation, "predict-proba")
  assert.equal(RunBatch(model, input, "predict", ObservationMode.Timing).operation, "predict")
}
