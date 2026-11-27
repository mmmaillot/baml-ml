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
  let m = LoadLogisticModel("artifacts/fixtures/iris.json");
  m.predict_proba(ImportedIrisRows(),["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"])
}
function ImportedDiabetesProbe() -> NumArrayResult {
  let m = LoadLinearModel("artifacts/fixtures/diabetes.json");
  m.predict(ImportedDiabetesRows(),["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"])
}
function ImportedBinaryProbe() -> ProbaResult {
  let m = LoadLogisticModel("artifacts/fixtures/binary.json");
  m.predict_proba(ImportedBinaryRows(),["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"])
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


function RepeatImportedIrisProbe() -> ProbaResult {
  let m = LoadLogisticModel("artifacts/fixtures/iris.json");
  let features = ["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"];
  let x = ImportedIrisRows();
  let first = m.predict_proba([x[0]],features);
  assert.is_true(first.ok)
  let second = m.predict_proba([x[1]],features);
  assert.is_true(second.ok)
  let batch = m.predict_proba(x,features);
  assert.is_true(batch.ok)
  assert.equal(first.probs[0],batch.probs[0])
  assert.equal(second.probs[0],batch.probs[1])
  batch
}

test "prepared_iris_repeated_single_and_batch" {
  assert.is_true(RepeatImportedIrisProbe().ok)
}

test "prepared_linear_repeat_and_request_validation" {
  let a = ImportedDiabetes();
  let m = LoadLinearModel("artifacts/fixtures/diabetes.json");
  let x = ImportedDiabetesRows();
  let p = m.predict(x,a.features);
  assert.is_true(p.ok)
  assert.equal(m.predict([x[0]],a.features).value,[p.value[0]])
  assert.equal(m.predict([x[1]],a.features).value,[p.value[1]])
  assert.is_true(!m.predict([],a.features).ok)
  assert.is_true(!m.predict([[1.0]],a.features).ok)
  assert.is_true(!m.predict([x[0],[1.0]],a.features).ok)
  assert.is_true(!m.predict(x,[]).ok)
  x[0][0] = float.inf();
  assert.is_true(!m.predict(x,a.features).ok)
  assert.equal(m.predict(ImportedDiabetesRows(),a.features).value,p.value)
}

test "prepared_models_own_state_and_outputs_do_not_alias" {
  let a = ImportedBinary();
  let names = a.features;
  let m1 = PrepareLogisticModel(baml.json.stringify(a));
  let x = ImportedBinaryRows();
  let r = m1.predict_proba(x,names);
  assert.is_true(r.ok)
  let original = r.probs[0][0];
  a.weights[0][0] = 100.0;
  a.intercepts[0] = 20.0;
  let m2 = PrepareLogisticModel(baml.json.stringify(a));
  a.weights[0][0] = float.nan();
  a.classes[0] = -999;
  a.features[0] = "changed";
  r.probs[0][0] = 123.0;
  x[0][0] = 6.0;
  let actual = ImportedBinaryRows();
  let correct = ["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"];
  assert.equal(m1.predict_proba(actual,correct).probs[0][0],original)
  assert.is_true(m2.predict_proba(actual,correct).probs[0][0] != original)
  assert.equal(m1.predict(actual,correct).value,[1,1])
  assert.equal(m1.predict_proba(actual,names).error,"artifact: feature order mismatch")
}

test "prepared_invalid_model_rejected_at_load" {
  let a = ImportedIris();
  a.scaler_scales[0] = 0.0;
  let failed: LoadedLogisticModel? = PrepareLogisticModel(baml.json.stringify(a)) catch_all (e) { _ => null };
  assert.equal(failed,null)
  let wrong: LoadedLinearModel? = LoadLinearModel("artifacts/fixtures/iris.json") catch_all (e) { _ => null };
  assert.equal(wrong,null)
  let malformed: LoadedLogisticModel? = PrepareLogisticModel("{") catch_all (e) { _ => null };
  assert.equal(malformed,null)
}

test "prepared_binary_tie_and_arithmetic_overflow" {
  let a = ImportedBinary();
  a.classes = [-7,42];
  a.weights = [[0.0,0.0,0.0,0.0]];
  a.intercepts = [0.0];
  let m = PrepareLogisticModel(baml.json.stringify(a));
  assert.equal(m.predict(ImportedBinaryRows(),a.features).value,[-7,-7])
  a.intercepts = [1e-18];
  let m2 = PrepareLogisticModel(baml.json.stringify(a));
  assert.equal(m2.predict(ImportedBinaryRows(),a.features).value,[42,42])
  a.weights[0][0] = 1e308;
  let huge = PrepareLogisticModel(baml.json.stringify(a));
  assert.is_true(!huge.predict_proba([[1e308,3.0,1.3,0.2]],a.features).ok)
  assert.is_true(!huge.predict([[1e308,3.0,1.3,0.2]],a.features).ok)
}
