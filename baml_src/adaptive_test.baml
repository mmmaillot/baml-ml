test "adaptive_policy_defaults_and_optional_json_fields" {
    let d = DefaultAdaptivePolicy();
    assert.is_true(
        d.enable_missing_input
            && d.enable_extreme_zscore
            && d.enable_low_margin
            && d.enable_label_disagreement
            && d.enable_score_delta,
    );
    assert.equal(
        [
            d.max_abs_z,
            d.min_proba_margin,
            d.max_proba_delta,
            d.max_pred_delta,
        ],
        [4.0, 0.2, 0.15, 1.0],
    );
    assert.equal(d.max_escalated_rows, 64);
    let fields = DecodeAdaptivePolicyOverride(
        `{"enable_low_margin":false,"max_abs_z":null,"max_escalated_rows":0}`,
    );
    assert.equal(fields.enable_low_margin, false);
    assert.equal(fields.enable_missing_input, null);
    assert.equal(fields.max_abs_z, null);
    let p = DecodeAdaptivePolicy(
        `{"enable_low_margin":false,"max_abs_z":null,"max_escalated_rows":0}`,
    );
    assert.equal(p.enable_low_margin, false);
    assert.equal(p.enable_missing_input, true);
    assert.equal(p.max_abs_z, 4.0);
    assert.equal(p.max_escalated_rows, 0);
    let boundary = DecodeAdaptivePolicy(
        `{"min_proba_margin":0,"max_proba_delta":1,"max_pred_delta":0}`,
    );
    assert.equal(boundary.min_proba_margin, 0.0);
    assert.equal(boundary.max_proba_delta, 1.0);
    assert.equal(boundary.max_pred_delta, 0.0)
}

test "adaptive_policy_rejects_bad_json_and_thresholds" {
    let invalid = [
        "{",
        "null",
        "[]",
        `{"unknown":true}`,
        `{"enable_low_margin":1}`,
        `{"max_abs_z":"4"}`,
        `{"max_escalated_rows":1.5}`,
        `{"max_escalated_rows":1.0}`,
        `{"max_abs_z":0}`,
        `{"max_abs_z":-1}`,
        `{"max_abs_z":1e309}`,
        `{"min_proba_margin":-0.1}`,
        `{"min_proba_margin":1.1}`,
        `{"max_proba_delta":-0.1}`,
        `{"max_proba_delta":1.1}`,
        `{"max_pred_delta":-1}`,
        `{"max_escalated_rows":-1}`,
    ];
    for (let raw in invalid) {
        let valid = (DecodeAdaptivePolicy(raw) catch (e) {
            let f: AdaptivePolicyError => null,
        }) != null;
        assert.is_true(!valid)
    }
    for (let value in [float.nan(), 1e308 * 2.0, -1e308 * 2.0]) {
        let p = DefaultAdaptivePolicy();
        p.max_abs_z = value;
        assert.is_true(!PolicyIsValid(p));
        p.max_abs_z = 4.0;
        p.min_proba_margin = value;
        assert.is_true(!PolicyIsValid(p));
        p.min_proba_margin = 0.2;
        p.max_proba_delta = value;
        assert.is_true(!PolicyIsValid(p));
        p.max_proba_delta = 0.15;
        p.max_pred_delta = value;
        assert.is_true(!PolicyIsValid(p))
    }
}

function PolicyIsValid(p: AdaptivePolicy) -> bool {
    ValidateAdaptivePolicy(p) catch (e) {
        let f: AdaptivePolicyError => {
            return false;
        },
    };
    true
}

function Signal(
    missing: bool,
    z: float,
    probs: float[],
    label: int?,
    prediction: float,
) -> AdaptiveRowSignals {
    AdaptiveRowSignals {
        has_nan: missing,
        max_abs_z: z,
        probabilities: probs,
        label: label,
        prediction: prediction,
    }
}

test "adaptive_triggers_are_ordered_inclusive_and_capped" {
    let p = DecodeAdaptivePolicy(`{"min_proba_margin":0.25,"max_proba_delta":0.25}`);
    let a = [
        Signal(false, 0.0, [0.875, 0.125], 0, 0.0),
        Signal(true, 0.0, [0.875, 0.125], 0, 0.0),
        Signal(false, 4.0, [0.875, 0.125], 0, 0.0),
        Signal(false, 0.0, [0.625, 0.375], 0, 0.0),
        Signal(true, 4.0, [0.625, 0.375], 0, 0.0),
        Signal(false, 0.0, [0.875, 0.125], 0, 0.0),
        Signal(false, 0.0, [0.875, 0.125], 0, 0.0),
        Signal(false, 0.0, [0.875, 0.125], 0, 0.0),
    ];
    let b = [
        a[0],
        a[1],
        a[2],
        a[3],
        Signal(true, 4.0, [0.375, 0.625], 1, 0.0),
        Signal(false, 0.0, [0.625, 0.375], 0, 0.0),
        Signal(false, 0.0, [0.875, 0.125], 1, 0.0),
        Signal(false, 4.0, [0.875, 0.125], 0, 0.0),
    ];
    let single = EvaluateTriggers(p, a);
    assert.equal(single.triggered_rows, 4);
    assert.equal(single.rows[0].trigger_reasons, "missing_input");
    assert.equal(single.rows[1].trigger_reasons, "extreme_zscore");
    assert.equal(single.rows[2].trigger_reasons, "low_margin");
    assert.equal(
        single.rows[3].trigger_reasons,
        "missing_input,extreme_zscore,low_margin",
    );
    let all = EvaluateTriggers(p, a, challenger = b);
    assert.equal(all.triggered_rows, 7);
    assert.equal(all.escalated_rows, 7);
    assert.is_true(!all.capped);
    assert.equal(
        all.rows[3].trigger_reasons,
        "missing_input,extreme_zscore,low_margin,label_disagreement,score_delta",
    );
    assert.equal(all.rows[4].trigger_reasons, "low_margin,score_delta");
    assert.equal(all.rows[5].trigger_reasons, "label_disagreement");
    assert.equal(all.rows[6].trigger_reasons, "extreme_zscore");
    assert.equal(all.triggers.missing_input, 2);
    assert.equal(all.triggers.extreme_zscore, 3);
    assert.equal(all.triggers.low_margin, 3);
    assert.equal(all.triggers.label_disagreement, 2);
    assert.equal(all.triggers.score_delta, 2);
    assert.equal(all.label_disagreements, 2);
    assert.equal(all.max_score_delta, 0.25);
    for (let cap in [0, 2, 7, 8]) {
        p.max_escalated_rows = cap;
        let capped = EvaluateTriggers(p, a, challenger = b);
        assert.equal(capped.triggered_rows, 7);
        assert.equal(capped.triggers, all.triggers);
        assert.equal(capped.capped, cap < 7);
        assert.equal(capped.escalated_rows, if (cap < 7) { cap } else { 7 });
        for (let i = 0; i < capped.rows.length(); i += 1) {
            assert.equal(capped.rows[i].row_index, i + 1)
        }
    }
    let disabled = DecodeAdaptivePolicy(
        `{"enable_missing_input":false,"enable_extreme_zscore":false,"enable_low_margin":false,"enable_label_disagreement":false,"enable_score_delta":false}`,
    );
    let off = EvaluateTriggers(disabled, a, challenger = b);
    assert.equal(off.triggered_rows, 0);
    assert.equal(off.label_disagreements, 2);
    assert.equal(off.max_score_delta, 0.25)
}

test "adaptive_regression_score_boundary_and_snapshot_guard" {
    let a = [
        Signal(false, 0.0, [], null, 2.0),
        Signal(false, 0.0, [], null, 2.0),
    ];
    let b = [
        Signal(false, 0.0, [], null, 1.5),
        Signal(false, 0.0, [], null, 3.0),
    ];
    let r = EvaluateTriggers(DefaultAdaptivePolicy(), a, challenger = b);
    assert.equal(r.triggered_rows, 1);
    assert.equal(r.rows[0].row_index, 1);
    assert.equal(r.rows[0].trigger_reasons, "score_delta");
    assert.equal(r.max_score_delta, 1.0);
    assert.is_true(HasMissingInput([0.0, float.nan()]));
    assert.is_true(!HasMissingInput([0.0, 1.0]));
    assert.equal(MaxAbsoluteValue([-4.0, 3.0]), 4.0);
    let guard = SnapshotGuard { captured: false };
    assert.is_true(guard.claim());
    assert.is_true(!guard.claim())
}

test "adaptive_policy_rejected_before_decoding_model_or_batch" {
    let policy = DefaultAdaptivePolicy();
    let linear = PrepareLinearModel("{", policy = policy) catch (e) {
        let failure: AdaptivePolicyError => null,
    };
    let logistic = PrepareLogisticModel("{", policy = policy) catch (e) {
        let failure: AdaptivePolicyError => null,
    };
    let tree = PrepareTreeModel("{", policy = policy) catch (e) {
        let failure: AdaptivePolicyError => null,
    };
    let batch = RunBatch(
        "{",
        "{",
        "predict",
        ObservationMode.Semantic,
        policy = policy,
    ) catch (e) {
        let failure: AdaptivePolicyError => null,
    };
    assert.equal(linear, null);
    assert.equal(logistic, null);
    assert.equal(tree, null);
    assert.equal(batch, null);

    policy.max_escalated_rows = -1;
    let invalid = RunBatch(
        "{",
        "{",
        "predict",
        ObservationMode.Adaptive,
        policy = policy,
    ) catch (e) {
        let failure: AdaptivePolicyError => null,
    };
    assert.equal(invalid, null)
}
