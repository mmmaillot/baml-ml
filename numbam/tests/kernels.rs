use numbam::{dot, exp, log, LengthMismatch};

fn close(actual: f64, expected: f64) {
    let tolerance = 1e-12 * expected.abs().max(1.0);
    assert!((actual - expected).abs() <= tolerance, "{actual} != {expected}");
}

#[test]
fn dot_known_values() {
    assert_eq!(dot(&[1.0, 2.0, -3.0], &[4.0, -5.0, 6.0]), Ok(-24.0));
    assert_eq!(dot(&[], &[]), Ok(0.0));
}

#[test]
fn dot_rejects_mismatched_lengths() {
    assert_eq!(dot(&[1.0], &[]), Err(LengthMismatch { left: 1, right: 0 }));
    assert_eq!(dot(&[], &[1.0]), Err(LengthMismatch { left: 0, right: 1 }));
}

#[test]
fn dot_propagates_nonfinite_arithmetic() {
    assert!(dot(&[f64::NAN], &[1.0]).unwrap().is_nan());
    assert!(dot(&[f64::INFINITY], &[0.0]).unwrap().is_nan());
    assert_eq!(dot(&[f64::INFINITY], &[2.0]), Ok(f64::INFINITY));
}

#[test]
fn exp_known_values_and_order() {
    let values = exp(&[-1.0, 0.0, 1.0, 2.0]);
    for (&actual, expected) in values.iter().zip([
        0.36787944117144233,
        1.0,
        std::f64::consts::E,
        7.38905609893065,
    ]) {
        close(actual, expected);
    }
    assert!(exp(&[]).is_empty());
}

#[test]
fn exp_extreme_logits() {
    let values = exp(&[-1000.0, -745.0, 0.0, 1000.0, f64::NAN, f64::NEG_INFINITY]);
    assert_eq!(values[0], 0.0);
    assert!(values[1].is_finite() && values[1] >= 0.0);
    assert_eq!(values[2], 1.0);
    assert_eq!(values[3], f64::INFINITY);
    assert!(values[4].is_nan());
    assert_eq!(values[5], 0.0);
}

#[test]
fn log_known_values() {
    close(log(1.0), 0.0);
    close(log(2.0), std::f64::consts::LN_2);
    close(log(10.0), std::f64::consts::LN_10);
    close(log(0.5), -std::f64::consts::LN_2);
}

#[test]
fn log_domain_is_explicit() {
    assert_eq!(log(0.0), f64::NEG_INFINITY);
    assert_eq!(log(-0.0), f64::NEG_INFINITY);
    assert!(log(-1.0).is_nan());
    assert!(log(f64::NAN).is_nan());
    assert_eq!(log(f64::INFINITY), f64::INFINITY);
    assert!(log(f64::NEG_INFINITY).is_nan());
}

#[test]
fn exp_log_round_trip() {
    let xs = [-100.0, -2.0, 0.0, 2.0, 100.0];
    for (&x, e) in xs.iter().zip(exp(&xs)) {
        close(log(e), x);
    }
}
