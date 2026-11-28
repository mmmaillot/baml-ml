// Adaptive policy schema, decoding, validation and resolution.
// Policy is data only. Each loaded handle owns a validated copy.

class AdaptivePolicy {
    enable_missing_input: bool,
    enable_extreme_zscore: bool,
    max_abs_z: float,
    enable_low_margin: bool,
    min_proba_margin: float,
    enable_label_disagreement: bool,
    enable_score_delta: bool,
    max_proba_delta: float,
    max_pred_delta: float,
    max_escalated_rows: int,
}


class AdaptivePolicyOverride {
    enable_missing_input: bool?,
    enable_extreme_zscore: bool?,
    max_abs_z: float?,
    enable_low_margin: bool?,
    min_proba_margin: float?,
    enable_label_disagreement: bool?,
    enable_score_delta: bool?,
    max_proba_delta: float?,
    max_pred_delta: float?,
    max_escalated_rows: int?,
}


class AdaptivePolicyError {
    message: string,
}


function DefaultAdaptivePolicy() -> AdaptivePolicy {
    AdaptivePolicy {
        enable_missing_input: true,
        enable_extreme_zscore: true,
        max_abs_z: 4.0,
        enable_low_margin: true,
        min_proba_margin: 0.2,
        enable_label_disagreement: true,
        enable_score_delta: true,
        max_proba_delta: 0.15,
        max_pred_delta: 1.0,
        max_escalated_rows: 64,
    }
}


function ValidateAdaptivePolicy(p: AdaptivePolicy) -> void throws AdaptivePolicyError {
    if (!IsFinite(p.max_abs_z) || p.max_abs_z <= 0.0) {
        throw AdaptivePolicyError {
            message: "policy: max_abs_z must be finite and positive",
        };
    }
    if (!IsFinite(p.min_proba_margin) || p.min_proba_margin < 0.0 || p.min_proba_margin > 1.0) {
        throw AdaptivePolicyError {
            message: "policy: min_proba_margin must be in [0, 1]",
        };
    }
    if (!IsFinite(p.max_proba_delta) || p.max_proba_delta < 0.0 || p.max_proba_delta > 1.0) {
        throw AdaptivePolicyError {
            message: "policy: max_proba_delta must be in [0, 1]",
        };
    }
    if (!IsFinite(p.max_pred_delta) || p.max_pred_delta < 0.0) {
        throw AdaptivePolicyError {
            message: "policy: max_pred_delta must be finite and nonnegative",
        };
    }
    if (p.max_escalated_rows < 0) {
        throw AdaptivePolicyError {
            message: "policy: max_escalated_rows must be nonnegative",
        };
    }
}

// In BAML 0.26.0 omitted optional fields and explicit null both decode to null.
// Class decoding ignores unknown fields, so check the object keys separately.

function DecodeAdaptivePolicyOverride(raw: string) -> AdaptivePolicyOverride
    throws AdaptivePolicyError
{
    let fields = baml.json.decode<map<string, bool | int | float | null>>(raw) catch_all (e) {
        _ => {
            throw AdaptivePolicyError {
                message: "policy: expected a JSON object of overrides",
            };
        },
    };
    for (let key in fields.keys()) {
        if (
            key != "enable_missing_input"
                && key != "enable_extreme_zscore"
                && key != "max_abs_z"
                && key != "enable_low_margin"
                && key != "min_proba_margin"
                && key != "enable_label_disagreement"
                && key != "enable_score_delta"
                && key != "max_proba_delta"
                && key != "max_pred_delta"
                && key != "max_escalated_rows"
        ) {
            throw AdaptivePolicyError {
                message: "policy: unknown field " + key,
            };
        }
    }
    baml.json.decode<AdaptivePolicyOverride>(raw) catch_all (e) {
        _ => {
            throw AdaptivePolicyError {
                message: "policy: invalid JSON override field type",
            };
        },
    }
}


function DecodeAdaptivePolicy(raw: string) -> AdaptivePolicy throws AdaptivePolicyError {
    let overrides = DecodeAdaptivePolicyOverride(raw);
    let d = DefaultAdaptivePolicy();
    let policy = AdaptivePolicy {
        enable_missing_input: overrides.enable_missing_input ?? d.enable_missing_input,
        enable_extreme_zscore: overrides.enable_extreme_zscore ?? d.enable_extreme_zscore,
        max_abs_z: overrides.max_abs_z ?? d.max_abs_z,
        enable_low_margin: overrides.enable_low_margin ?? d.enable_low_margin,
        min_proba_margin: overrides.min_proba_margin ?? d.min_proba_margin,
        enable_label_disagreement: overrides.enable_label_disagreement
            ?? d.enable_label_disagreement,
        enable_score_delta: overrides.enable_score_delta ?? d.enable_score_delta,
        max_proba_delta: overrides.max_proba_delta ?? d.max_proba_delta,
        max_pred_delta: overrides.max_pred_delta ?? d.max_pred_delta,
        max_escalated_rows: overrides.max_escalated_rows ?? d.max_escalated_rows,
    };
    ValidateAdaptivePolicy(policy);
    policy
}


function ResolveAdaptivePolicy(mode: ObservationMode, policy: AdaptivePolicy?) -> AdaptivePolicy?
    throws AdaptivePolicyError
{
    if (mode != ObservationMode.Adaptive) {
        if (policy != null) {
            throw AdaptivePolicyError {
                message: "policy requires adaptive observation",
            };
        }
        return null;
    }
    if (policy == null) {
        return DefaultAdaptivePolicy();
    }
    // Copy the scalar fields so later caller mutations cannot change the handle.
    ValidateAdaptivePolicy(policy);
    AdaptivePolicy {
        enable_missing_input: policy.enable_missing_input,
        enable_extreme_zscore: policy.enable_extreme_zscore,
        max_abs_z: policy.max_abs_z,
        enable_low_margin: policy.enable_low_margin,
        min_proba_margin: policy.min_proba_margin,
        enable_label_disagreement: policy.enable_label_disagreement,
        enable_score_delta: policy.enable_score_delta,
        max_proba_delta: policy.max_proba_delta,
        max_pred_delta: policy.max_pred_delta,
        max_escalated_rows: policy.max_escalated_rows,
    }
}


function ReadAdaptivePolicy(path: string) -> AdaptivePolicy {
    if (path == "") {
        return DefaultAdaptivePolicy();
    }
    DecodeAdaptivePolicy(baml.fs.read(path))
}


