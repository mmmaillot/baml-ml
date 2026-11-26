use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use baml_sdk::call_native_dot_sync;

fn main() {
    // A counter makes the test fail if BAML ever bypasses the Rust callback.
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&calls);
    let result = call_native_dot_sync(
        move |a: Vec<f64>, b: Vec<f64>| {
            seen.fetch_add(1, Ordering::SeqCst);
            numbam::dot(&a, &b)
        },
        vec![1.0, 2.0, -3.0],
        vec![4.0, -5.0, 6.0],
    )
    .expect("BAML must execute NumBam dot");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(result, -24.0);

    let result = call_native_dot_sync(
        |a: Vec<f64>, b: Vec<f64>| numbam::dot(&a, &b),
        vec![1.0],
        vec![2.0, 3.0],
    );
    assert!(result.is_err(), "length mismatch must propagate through BAML");
    println!("PASS: BAML invoked NumBam dot in-process; result=-24; length mismatch propagated");
}
