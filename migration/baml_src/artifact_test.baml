// The committed JSON fixtures were fitted independently using sklearn 1.8.0.
// Read real artifact files, not BAML literals generated from model parameters.
function ImportedIris() -> FittedArtifact { ReadFittedArtifact("artifacts/fixtures/iris.json") }
function ImportedDiabetes() -> FittedArtifact { ReadFittedArtifact("artifacts/fixtures/diabetes.json") }
function ImportedBinary() -> FittedArtifact { ReadFittedArtifact("artifacts/fixtures/binary.json") }

function ImportedIrisRows() -> float[][] { [[float.nan(),3.0,1.3,0.2],[6.1,3.0,4.9,1.8]] }
function ImportedBinaryRows() -> float[][] { [[float.nan(),2.4,3.3,1.0],[6.7,3.0,5.0,1.7]] }
function ImportedDiabetesRows() -> float[][] {
  [[float.nan(),-0.044641636506989144,-0.006205954135807083,-0.015998975220305175,0.12501870313429186,0.1251981011367534,0.019186997017453092,0.03430885887772673,0.03243232415655107,-0.005219804415300423],
   [0.09256398319871433,-0.044641636506989144,0.0369065288194249,0.0218723855140367,-0.0249601584096303,-0.016658152053905938,0.0007788079970183853,-0.03949338287409329,-0.022516528376302174,-0.021788232074638245]]
}
function ImportedIrisProbe() -> ProbaResult {
  let a = ImportedIris();
  PredictImportedProba(ImportedIrisRows(),a.features,a)
}
function ImportedDiabetesProbe() -> NumArrayResult {
  let a = ImportedDiabetes();
  PredictImportedLinear(ImportedDiabetesRows(),a.features,a)
}
function ImportedBinaryProbe() -> ProbaResult {
  let a = ImportedBinary();
  PredictImportedProba(ImportedBinaryRows(),a.features,a)
}

test "imported_iris_multiclass" {
  let a = ImportedIris();
  assert.equal(ArtifactError(a),"")
  let p = ImportedIrisProbe();
  assert.is_true(p.ok)
  let expected = [[0.8217064033386521,0.17829002427778545,3.5723835624939457e-6],[0.0038088445397161117,0.37263701903431495,0.6235541364259689]];
  for (let i = 0; i < 2; i += 1) {
    for (let k = 0; k < 3; k += 1) { assert.is_true(Abs(p.probs[i][k]-expected[i][k])<1e-10) }
  }
  assert.equal(PredictImportedLabels(ImportedIrisRows(),a.features,a).value,[0,2])
}
test "imported_diabetes_linear" {
  let p = ImportedDiabetesProbe();
  assert.is_true(p.ok)
  assert.is_true(Abs(p.value[0]-137.8446766318574)<1e-9)
  assert.is_true(Abs(p.value[1]-179.97012889945012)<1e-9)
}
test "imported_binary_logistic" {
  let a = ImportedBinary();
  let p = ImportedBinaryProbe();
  assert.is_true(p.ok)
  let expected = [[0.06799970430610935,0.9320002956938906],[0.001255564699456535,0.9987444353005435]];
  for (let i = 0; i < 2; i += 1) {
    for (let k = 0; k < 2; k += 1) { assert.is_true(Abs(p.probs[i][k]-expected[i][k])<1e-10) }
  }
  assert.equal(PredictImportedLabels(ImportedBinaryRows(),a.features,a).value,[1,1])
}
test "imported_reject_wrong_feature_order_and_estimator" {
  let a = ImportedIris();
  assert.equal(PredictImportedProba(ImportedIrisRows(),["petal length (cm)","sepal width (cm)","sepal length (cm)","petal width (cm)"],a).error,"artifact: feature order mismatch")
  assert.equal(PredictImportedLinear(ImportedIrisRows(),a.features,a).error,"artifact: expected linear estimator")
  assert.equal(PredictImportedProba([[float.inf(),3.0,1.3,0.2]],a.features,a).error,"imputer_transform: input contains infinite values")
}
test "imported_reject_bad_version_dimensions_and_parameters" {
  let a = ImportedIris();
  a.schema_version = 2;
  assert.equal(ArtifactError(a),"artifact: unsupported schema version")
  a.schema_version = 1;
  a.scaler_scales[0] = 0.0;
  assert.equal(ArtifactError(a),"artifact: invalid preprocessing parameters")
  a.scaler_scales[0] = 1.0;
  a.weights[0][0] = float.nan();
  assert.equal(ArtifactError(a),"artifact: non-finite weight")
  a.weights[0][0] = 0.0;
  a.weights = [];
  assert.equal(ArtifactError(a),"artifact: invalid multinomial dimensions")
}

test "imported_binary_decision_boundary" {
  let a = ImportedBinary();
  a.classes = [-7,42];
  a.weights = [[0.0,0.0,0.0,0.0]];
  a.intercepts = [0.0];
  assert.equal(PredictImportedLabels(ImportedBinaryRows(),a.features,a).value,[-7,-7])
  a.intercepts = [1e-18];
  assert.equal(PredictImportedLabels(ImportedBinaryRows(),a.features,a).value,[42,42])
}
test "imported_reject_invalid_json" {
  let a: FittedArtifact? = DecodeFittedArtifact("{") catch_all (e) { _ => null };
  assert.equal(a,null)
}
