//! Small, safe Rust kernels for the baml-ml interop experiment.
//!
//! No BAML binding is implemented here. Keep this crate independent of the
//! runtime so it can also be linked into a future embedded inference binary.

use std::{error::Error, fmt};

/// Input validation error for a dot product.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LengthMismatch {
    pub left: usize,
    pub right: usize,
}

impl fmt::Display for LengthMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "dot: vector lengths differ ({} != {})", self.left, self.right)
    }
}

impl Error for LengthMismatch {}

/// Dense float64 dot product in input order.
///
/// Empty vectors produce zero. Nonfinite values follow IEEE floating-point
/// arithmetic; validation of model inputs remains the caller's responsibility.
/// No fast-math or parallel reduction, so comparisons with the reference
/// implementation are easier to interpret.
pub fn dot(a: &[f64], b: &[f64]) -> Result<f64, LengthMismatch> {
    if a.len() != b.len() {
        return Err(LengthMismatch {
            left: a.len(),
            right: b.len(),
        });
    }
    Ok(a.iter().zip(b).fold(0.0, |sum, (&x, &y)| sum + x * y))
}

/// Elementwise natural exponential in one coarse-grained native call.
/// Preserves IEEE NaN, infinity and subnormal behavior of Rust f64::exp.
pub fn exp(values: &[f64]) -> Vec<f64> {
    values.iter().map(|&x| x.exp()).collect()
}

/// Natural logarithm, using standard IEEE domain semantics:
/// zero -> -infinity; negative -> NaN; +infinity -> +infinity.
/// The current BAML reference uses a finite negative sentinel for x <= 0.
/// Its production callers pass positive values; a future adapter must preserve
/// the reference contract where required rather than hide this difference.
pub fn log(value: f64) -> f64 {
    value.ln()
}
