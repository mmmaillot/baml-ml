// Row signal collection and trigger evaluation (policy math, no tracing).
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

