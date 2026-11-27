test "batch_json_numbers_and_null" {
  let x = DecodeBatchInput(`{"features":["a","b"],"rows":[[1,null],[2.5,-3]]}`);
  let r = BatchRows(x);
  assert.equal(x.features,["a","b"])
  assert.equal(r[0][0],1.0)
  assert.is_true(IsNaN(r[0][1]))
  assert.equal(r[1],[2.5,-3.0])
}
test "batch_json_reject_string_values" {
  let bad: BatchInput? = DecodeBatchInput(`{"features":["a"],"rows":[["1"]]}`) catch_all (e) { _ => null };
  assert.equal(bad,null)
}
test "batch_dispatch_rejects_unsupported_operation" {
  let bad: BatchOutput? = RunBatch("{}", "{}", "fit", ObservationMode.Timing) catch_all (e) { _ => null };
  assert.equal(bad,null)
}
test "batch_iris_labels_and_probabilities" {
  let model = baml.fs.read("artifacts/fixtures/iris.json");
  let input = `{"features":["sepal length (cm)","sepal width (cm)","petal length (cm)","petal width (cm)"],"rows":[[null,3.0,1.3,0.2],[6.1,3.0,4.9,1.8]]}`;
  let labels = RunBatch(model,input,"predict",ObservationMode.Timing);
  let probs = RunBatch(model,input,"predict-proba",ObservationMode.Timing);
  assert.equal(labels.classes,[0,1,2])
  assert.equal(labels.predictions,[0,2])
  assert.equal(probs.estimator_kind,"logistic_multinomial")
  assert.equal(probs.operation,"predict-proba")
  assert.equal(probs.predictions,ImportedIrisProbe().probs)
}
