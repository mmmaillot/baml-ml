function TreeSmokeModel() -> string { baml.fs.read("artifacts/fixtures/tree-smoke.json") }
test "tree_artifact_batch_and_modes" {
  let input = `{"features":["x"],"rows":[[null],[-1],[1]]}`;
  let labels = RunBatch(TreeSmokeModel(),input,"predict",ObservationMode.Semantic);
  let probs = RunBatch(TreeSmokeModel(),input,"predict-proba",ObservationMode.Full);
  assert.equal(labels.predictions,[-3,-3,7])
  assert.equal(probs.predictions,[[0.5,0.5],[0.5,0.5],[0.0,1.0]])
  assert.equal(probs.predictions,RunBatch(TreeSmokeModel(),input,"predict-proba",ObservationMode.Timing).predictions)
}
test "tree_prepared_reuse_and_output_isolation" {
  let m = PrepareTreeModel(TreeSmokeModel());
  let x = [[-1.0],[1.0]];
  let r = m.predict_proba(x,["x"]);
  assert.is_true(r.ok)
  r.probs[0][0] = 123.0;
  assert.equal(m.predict_proba(x,["x"]).probs,[[0.5,0.5],[0.0,1.0]])
  assert.equal(m.predict(x,["x"]).value,[-3,7])
  assert.is_true(!m.predict(x,["wrong"]).ok)
  assert.is_true(!m.predict([[1e40]],["x"]).ok)
}
test "tree_malformed_rejected_on_load" {
  let a = DecodeTreeArtifact(TreeSmokeModel());
  a.nodes[0].left = 0;
  let failed: LoadedTreeModel? = PrepareTreeModel(baml.json.stringify(a)) catch_all (e) { _ => null };
  assert.equal(failed,null)
}
class TreePipelineOracle {
  input_json string
  labels int[]
  probs float[][]
}
function TreePipelineOracleProbe() -> int {
  let a = baml.fs.read("tree-pipeline.json");
  let oracle = baml.json.decode<TreePipelineOracle>(baml.fs.read("tree-pipeline-oracle.json"));
  let labels = RunBatch(a,oracle.input_json,"predict",ObservationMode.Timing);
  let probs = RunBatch(a,oracle.input_json,"predict-proba",ObservationMode.Timing);
  assert.equal(labels.predictions,oracle.labels)
  assert.equal(probs.predictions,oracle.probs)
  oracle.labels.length()
}
