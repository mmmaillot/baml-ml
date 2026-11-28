function ObservationFixture(kind: string) -> FittedArtifact {
    let fixture = if (kind == "linear") {
        "diabetes"
    } else if (kind == "logistic_binary") {
        "binary"
    } else {
        "iris"
    };
    let a = DecodeFittedArtifact(baml.fs.read("artifacts/fixtures/" + fixture + ".json"));
    a.model_id = "champion-" + kind;
    a.features = ["x"];
    a.imputer_means = [0.0];
    a.scaler_means = [0.0];
    a.scaler_scales = [1.0];
    if (kind == "logistic_multinomial") {
        a.weights = [[-2.0], [0.0], [2.0]];
        a.intercepts = [0.0, -2.0, 0.0];
    } else {
        a.weights = [[2.0]];
        a.intercepts = [0.0];
    }
    a
}

function ChallengerFixture(kind: string) -> FittedArtifact {
    let a = ObservationFixture(kind);
    a.model_id = "challenger-" + kind;
    a.imputer_means = [0.5];
    a.scaler_means = [0.5];
    a.scaler_scales = [0.5];
    if (kind == "logistic_multinomial") {
        a.weights = [[2.0], [0.0], [-2.0]];
    } else if (kind == "logistic_binary") {
        a.weights = [[-2.0]];
    } else {
        a.intercepts = [2.0];
    }
    a
}

test "adaptive_matches_semantic_and_full_for_each_operation" {
    let model_paths = [
        "artifacts/fixtures/diabetes.json",
        "artifacts/fixtures/binary.json",
        "artifacts/fixtures/iris.json",
        "artifacts/fixtures/tree-smoke.json",
    ];
    for (let i = 0; i < model_paths.length(); i += 1) {
        let model = baml.fs.read(model_paths[i]);
        let features = if (i == 3) {
            DecodeTreeArtifact(model).features
        } else {
            DecodeFittedArtifact(model).features
        };
        let row: (float?)[] = [];
        for (let name in features) {
            row.push(null);
        }
        let raw = baml.json.stringify(BatchInput {
            features: features,
            rows: [row, row],
        });
        let policy = DecodeAdaptivePolicy(`{"max_escalated_rows":1}`);
        let operations = if (i == 0) {
            ["predict"]
        } else {
            ["predict", "predict-proba"]
        };
        for (let operation in operations) {
            let semantic = RunBatch(model, raw, operation, ObservationMode.Semantic);
            let full = RunBatch(model, raw, operation, ObservationMode.Full);
            let adaptive = RunBatch(
                model,
                raw,
                operation,
                ObservationMode.Adaptive,
                policy = policy,
            );
            assert.equal(adaptive, semantic);
            assert.equal(adaptive, full)
        }
    }
}

test "loaded_adaptive_reuses_handle_and_rejects_misapplied_policy" {
    let policy = DefaultAdaptivePolicy();
    let model = LoadLogisticModel(
        "artifacts/fixtures/iris.json",
        observation = ObservationMode.Adaptive,
        policy = policy,
    );
    let a = DecodeFittedArtifact(baml.fs.read("artifacts/fixtures/iris.json"));
    let rows = [[float.nan(), 3.0, 1.3, 0.2]];
    assert.equal(model.predict_proba(rows, a.features).probs, [
        ImportedIrisProbe().probs[0],
    ]);
    assert.equal(model.predict(rows, a.features).value, [0]);
    assert.equal(model.predict_proba(rows, a.features).probs, [
        ImportedIrisProbe().probs[0],
    ]);
    assert.equal(
        model.predict(rows, ["wrong"]).error,
        "artifact: feature order mismatch",
    );
    let wrong = LoadLogisticModel("artifacts/fixtures/iris.json", policy = policy) catch (e) {
        let f: AdaptivePolicyError => null,
    };
    assert.equal(wrong, null);
    policy.max_escalated_rows = -1;
    let invalid = LoadLogisticModel(
        "artifacts/fixtures/iris.json",
        observation = ObservationMode.Adaptive,
        policy = policy,
    ) catch (e) {
        let f: AdaptivePolicyError => null,
    };
    assert.equal(invalid, null);
    assert.equal(model.predict(rows, a.features).value, [0])
}

test "shadow_linear_binary_multinomial_and_tree_preserve_champion" {
    let tree = baml.fs.read("artifacts/fixtures/tree-smoke.json");
    let tree_b = DecodeTreeArtifact(tree);
    tree_b.model_id = "tree-challenger";
    tree_b.imputer_means = [2.0];
    tree_b.scaler_means = [2.0];
    tree_b.scaler_scales = [0.5];
    let kinds = [
        "linear",
        "logistic_binary",
        "logistic_multinomial",
        "decision_tree_classifier",
    ];
    let healthy = `{"features":["x"],"rows":[[1],[1.5]]}`;
    let mixed = `{"features":["x"],"rows":[[null],[5],[1]]}`;
    let policy = DecodeAdaptivePolicy(`{"max_escalated_rows":1}`);
    for (let kind in kinds) {
        let is_tree = kind == "decision_tree_classifier";
        let a = if (is_tree) {
            tree
        } else {
            baml.json.stringify(ObservationFixture(kind))
        };
        let b = if (is_tree) {
            baml.json.stringify(tree_b)
        } else {
            baml.json.stringify(ChallengerFixture(kind))
        };
        let ops = if (kind == "linear") {
            ["predict"]
        } else {
            ["predict", "predict-proba"]
        };
        for (let op in ops) {
            let same = RunShadow(a, a, healthy, op, DefaultAdaptivePolicy());
            assert.is_true(same.summary.challenger_ok);
            assert.equal(same.summary.triggered_rows, 0);
            assert.equal(same.summary.label_disagreements, 0);
            assert.equal(same.summary.max_score_delta, 0.0);
            let diff = RunShadow(a, b, mixed, op, policy);
            let expected = RunBatch(a, mixed, op, ObservationMode.Semantic);
            assert.equal(diff.predictions, expected.predictions);
            assert.equal(diff.classes, expected.classes);
            assert.equal(diff.champion_model_id, expected.model_id);
            assert.is_true(diff.challenger_model_id != diff.champion_model_id);
            assert.equal(diff.schema_version, 1);
            assert.equal(diff.estimator_kind, kind);
            assert.equal(diff.operation, op);
            assert.equal(diff.summary.total_rows, 3);
            assert.is_true(diff.summary.triggered_rows >= 2);
            assert.equal(diff.summary.escalated_rows, 1);
            assert.is_true(diff.summary.capped);
            assert.equal(diff.summary.triggers.missing_input, 1);
            assert.is_true(diff.summary.triggers.extreme_zscore >= 1);
            assert.is_true(diff.summary.triggers.score_delta >= 1);
            if (kind != "linear") {
                assert.is_true(diff.summary.label_disagreements > 0)
            }
        }
    }
}

test "shadow_strict_model_input_and_operation_compatibility" {
    let a = ObservationFixture("logistic_binary");
    let input = `{"features":["x"],"classes":[0,1],"rows":[[1]]}`;
    let b = ChallengerFixture("logistic_binary");
    let p = DefaultAdaptivePolicy();
    let raw = baml.json.stringify(a);
    assert.is_true(
        RunShadow(raw, baml.json.stringify(b), input, "predict", p).summary.challenger_ok,
    );
    let invalid_pairs = [
        baml.json.stringify(ObservationFixture("linear")),
        baml.json.stringify(ObservationFixture("logistic_multinomial")),
        baml.fs.read("artifacts/fixtures/tree-smoke.json"),
        "{",
    ];
    b.features = ["z"];
    invalid_pairs.push(baml.json.stringify(b));
    b.features = ["x"];
    b.classes = [2, 3];
    invalid_pairs.push(baml.json.stringify(b));
    for (let other in invalid_pairs) {
        let output: ShadowOutput? = RunShadow(raw, other, input, "predict", p) catch_all (e) {
            _ => null,
        };
        assert.equal(output, null)
    }
    for (
        let wrong_input in [
            `{"features":["z"],"rows":[[1]]}`,
            `{"features":["x"],"classes":[1,0],"rows":[[1]]}`,
        ]
    ) {
        let output: ShadowOutput? = RunShadow(raw, raw, wrong_input, "predict", p) catch_all (e) {
            _ => null,
        };
        assert.equal(output, null)
    }
    let linear = baml.json.stringify(ObservationFixture("linear"));
    let linear_input = `{"features":["x"],"rows":[[1]]}`;
    for (let op in ["predict-proba", "fit"]) {
        let output: ShadowOutput? = RunShadow(linear, linear, linear_input, op, p) catch_all (e) {
            _ => null,
        };
        assert.equal(output, null)
    }
}

test "shadow_isolates_challenger_overflow_and_champion_failures_are_fatal" {
    let a = ObservationFixture("linear");
    let b = ChallengerFixture("linear");
    b.scaler_means = [0.0];
    b.scaler_scales = [1.0];
    b.weights = [[1e308]];
    let raw_a = baml.json.stringify(a);
    let raw_b = baml.json.stringify(b);
    let input = `{"features":["x"],"rows":[[2]]}`;
    let p = DecodeAdaptivePolicy(`{"max_escalated_rows":0}`);
    let r = RunShadow(raw_a, raw_b, input, "predict", p);
    assert.equal(r.predictions, [4.0]);
    assert.is_true(!r.summary.challenger_ok);
    assert.equal(r.summary.escalated_rows, 0);
    let error = r.summary.challenger_error;
    assert.is_true(error != null && error.stage == "linear_prediction");
    let output: ShadowOutput? = RunShadow(raw_b, raw_a, input, "predict", p) catch_all (e) {
        _ => null,
    };
    assert.equal(output, null);

    a.weights = [[0.0]];
    a.intercepts = [1e308];
    b.weights = [[0.0]];
    b.intercepts = [-1e308];
    let wide = RunShadow(
        baml.json.stringify(a),
        baml.json.stringify(b),
        input,
        "predict",
        p,
    );
    assert.is_true(wide.summary.challenger_ok);
    assert.equal(wide.summary.triggers.score_delta, 1);
    assert.is_true(IsFinite(wide.summary.max_score_delta));
    assert.is_true(baml.json.stringify(wide).length() > 0)
}

test "adaptive_snapshot_guard_stays_lazy_until_a_row_is_replayed" {
    let a = ObservationFixture("linear");
    let model = SelectiveLinear(a, ArtifactRegression(a));
    let policy = DefaultAdaptivePolicy();
    assert.equal(
        RunAdaptive(model, [[1.0]], ["x"], Operation.Predict, policy).values,
        [2.0],
    );
    assert.is_true(!model.guard.captured);
    policy.max_escalated_rows = 0;
    assert.equal(
        RunAdaptive(model, [[float.nan()]], ["x"], Operation.Predict, policy).values,
        [0.0],
    );
    assert.is_true(!model.guard.captured);
    policy.max_escalated_rows = 1;
    assert.equal(
        RunAdaptive(model, [[float.nan()]], ["x"], Operation.Predict, policy).values,
        [0.0],
    );
    assert.is_true(model.guard.captured);
    assert.equal(
        RunAdaptive(model, [[5.0]], ["x"], Operation.Predict, policy).values,
        [10.0],
    );
    assert.is_true(model.guard.captured)
}

