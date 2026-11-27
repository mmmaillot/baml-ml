function ForestSmokeModel() -> string {
    baml.fs.read("artifacts/fixtures/forest-smoke.json")
}

test "forest_artifact_batch_and_modes" {
    let input = `{"features":["x"],"rows":[[null],[-1],[1],[2]]}`;
    let labels = RunBatch(
        ForestSmokeModel(),
        input,
        "predict",
        ObservationMode.Semantic,
    );
    let probs = RunBatch(
        ForestSmokeModel(),
        input,
        "predict-proba",
        ObservationMode.Full,
    );
    assert.equal(labels.estimator_kind, "random_forest_classifier");
    assert.equal(labels.classes, [-3, 7]);
    assert.equal(labels.predictions, [-3, -3, -3, 7]);
    assert.equal(probs.predictions, [
        [0.75, 0.25],
        [0.75, 0.25],
        [0.75, 0.25],
        [0.375, 0.625],
    ]);
    assert.equal(
        probs.predictions,
        RunBatch(
            ForestSmokeModel(),
            input,
            "predict-proba",
            ObservationMode.Timing,
        ).predictions,
    );
    let adaptive = RunBatch(
        ForestSmokeModel(),
        input,
        "predict-proba",
        ObservationMode.Adaptive,
        policy = DecodeAdaptivePolicy(`{"max_escalated_rows":1}`),
    );
    assert.equal(adaptive, probs)
}

test "forest_prepared_reuse_and_output_isolation" {
    let m = PrepareForestModel(ForestSmokeModel());
    let x = [[-1.0], [2.0]];
    let r = m.predict_proba(x, ["x"]);
    assert.is_true(r.ok);
    assert.equal(r.probs, [[0.75, 0.25], [0.375, 0.625]]);
    r.probs[0][0] = 123.0;
    assert.equal(m.predict_proba(x, ["x"]).probs, [
        [0.75, 0.25],
        [0.375, 0.625],
    ]);
    assert.equal(m.predict(x, ["x"]).value, [-3, 7]);
    assert.is_true(!m.predict(x, ["wrong"]).ok);
    assert.is_true(!m.predict([[1e40]], ["x"]).ok)
}

test "forest_malformed_rejected_on_load" {
    let self_loop = DecodeForestArtifact(ForestSmokeModel());
    self_loop.trees[0].nodes[0].left = 0;
    let no_trees = DecodeForestArtifact(ForestSmokeModel());
    no_trees.trees = [];
    let bad_width = DecodeForestArtifact(ForestSmokeModel());
    bad_width.n_features = 2;
    let bad_classes = DecodeForestArtifact(ForestSmokeModel());
    bad_classes.classes = [7, -3];
    let mismatched_tree = DecodeForestArtifact(ForestSmokeModel());
    mismatched_tree.trees[1].classes = [-3, 8];
    let bad_schema = DecodeForestArtifact(ForestSmokeModel());
    bad_schema.schema_version = 2;
    for (
        let a in [
            self_loop,
            no_trees,
            bad_width,
            bad_classes,
            mismatched_tree,
            bad_schema,
        ]
    ) {
        let failed: LoadedForestModel? = PrepareForestModel(baml.json.stringify(a)) catch_all (e) {
            _ => null,
        };
        assert.equal(failed, null)
    }
}

class ForestPipelineOracle {
    input_json: string,
    labels: int[],
    probs: float[][],
}

function ForestPipelineOracleProbe() -> int {
    let a = baml.fs.read("forest-pipeline.json");
    let oracle =
        baml.json.decode<ForestPipelineOracle>(baml.fs.read("forest-pipeline-oracle.json"));
    let labels = RunBatch(a, oracle.input_json, "predict", ObservationMode.Timing);
    let probs = RunBatch(
        a,
        oracle.input_json,
        "predict-proba",
        ObservationMode.Timing,
    );
    assert.equal(labels.predictions, oracle.labels);
    assert.equal(probs.predictions, oracle.probs);
    oracle.labels.length()
}
