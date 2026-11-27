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

class SnapshotGuard {
    captured: bool,
    function claim(self) -> bool {
        if (self.captured) {
            return false;
        }
        self.captured = true;
        true
    }
}

class AdaptiveRowSignals {
    has_nan: bool,
    max_abs_z: float,
    probabilities: float[],
    label: int?,
    prediction: float,
}

class TriggerCounts {
    missing_input: int,
    extreme_zscore: int,
    low_margin: int,
    label_disagreement: int,
    score_delta: int,
}

function EmptyTriggerCounts() -> TriggerCounts {
    TriggerCounts {
        missing_input: 0,
        extreme_zscore: 0,
        low_margin: 0,
        label_disagreement: 0,
        score_delta: 0,
    }
}

class Escalation {
    row_index: int,
    trigger_reasons: string,
}

class TriggerEvaluation {
    triggered_rows: int,
    escalated_rows: int,
    capped: bool,
    triggers: TriggerCounts,
    rows: Escalation[],
    label_disagreements: int,
    max_score_delta: float,
}

function HasMissingInput(row: float[]) -> bool {
    for (let value in row) {
        if (IsNaN(value)) {
            return true;
        }
    }
    false
}

function MaxAbsoluteValue(row: float[]) -> float {
    let maximum = 0.0;
    for (let value in row) {
        let magnitude = Abs(value);
        if (magnitude > maximum) {
            maximum = magnitude;
        }
    }
    maximum
}

function HasLowMargin(row: AdaptiveRowSignals, threshold: float) -> bool {
    if (row.probabilities.length() < 2) {
        return false;
    }
    let first = 0.0;
    let second = 0.0;
    for (let p in row.probabilities) {
        if (p > first) {
            second = first;
            first = p;
        } else if (p > second) {
            second = p;
        }
    }
    first - second <= threshold
}

function ShadowScoreDelta(a: AdaptiveRowSignals, b: AdaptiveRowSignals) -> float {
    if (a.label == null) {
        return Abs(a.prediction - b.prediction);
    }
    let maximum = 0.0;
    for (let k = 0; k < a.probabilities.length(); k += 1) {
        let delta = Abs(a.probabilities[k] - b.probabilities[k]);
        if (delta > maximum) {
            maximum = delta;
        }
    }
    maximum
}

function AddTriggerReason(reasons: string, reason: string) -> string {
    if (reasons == "") {
        return reason;
    }
    reasons + "," + reason
}

// Challenger signals, when present, have the same length and class order as
// the champion. Count each reason at most once per row, even if both fire.
function EvaluateTriggers(
    policy: AdaptivePolicy,
    champion: AdaptiveRowSignals[],
    challenger: AdaptiveRowSignals[]? = null,
) -> TriggerEvaluation {
    let counts = EmptyTriggerCounts();
    let rows: Escalation[] = [];
    let triggered = 0;
    let disagreements = 0;
    let max_delta = 0.0;
    for (let i = 0; i < champion.length(); i += 1) {
        let a = champion[i];
        let missing = a.has_nan;
        let extreme = a.max_abs_z >= policy.max_abs_z;
        let low = policy.enable_low_margin && HasLowMargin(a, policy.min_proba_margin);
        let disagreement = false;
        let score_delta = false;
        if (challenger != null) {
            let b = challenger[i];
            missing = missing || b.has_nan;
            extreme = extreme || b.max_abs_z >= policy.max_abs_z;
            low = low || (policy.enable_low_margin && HasLowMargin(b, policy.min_proba_margin));
            disagreement = a.label != b.label;
            if (disagreement) {
                disagreements += 1;
            }
            let delta = ShadowScoreDelta(a, b);
            if (delta > max_delta) {
                max_delta = delta;
            }
            score_delta = if (a.label == null) {
                delta >= policy.max_pred_delta
            } else {
                delta >= policy.max_proba_delta
            };
        }

        let reasons = "";
        if (policy.enable_missing_input && missing) {
            counts.missing_input += 1;
            reasons = AddTriggerReason(reasons, "missing_input");
        }
        if (policy.enable_extreme_zscore && extreme) {
            counts.extreme_zscore += 1;
            reasons = AddTriggerReason(reasons, "extreme_zscore");
        }
        if (low) {
            counts.low_margin += 1;
            reasons = AddTriggerReason(reasons, "low_margin");
        }
        if (policy.enable_label_disagreement && disagreement) {
            counts.label_disagreement += 1;
            reasons = AddTriggerReason(reasons, "label_disagreement");
        }
        if (policy.enable_score_delta && score_delta) {
            counts.score_delta += 1;
            reasons = AddTriggerReason(reasons, "score_delta");
        }
        if (reasons != "") {
            triggered += 1;
            if (rows.length() < policy.max_escalated_rows) {
                rows.push(Escalation {
                    row_index: i,
                    trigger_reasons: reasons,
                });
            }
        }
    }
    // Two individually finite regression values can have an unrepresentable
    // difference. Preserve the trigger and keep the JSON summary numeric.
    if (!IsFinite(max_delta)) {
        max_delta = 1.7976931348623157e308;
    }
    TriggerEvaluation {
        triggered_rows: triggered,
        escalated_rows: rows.length(),
        capped: triggered > rows.length(),
        triggers: counts,
        rows: rows,
        label_disagreements: disagreements,
        max_score_delta: max_delta,
    }
}
