use baml_sdk::{bridge_probe_sync, call_native_dot_sync, pure_dot_sync};
use std::{hint::black_box, time::Instant};

const WARMUPS: usize = 5;
const SAMPLES: usize = 31;
const SIZES: [usize; 9] = [0, 4, 10, 64, 256, 1024, 4096, 16384, 65536];

fn close(value: f64, expected: f64) {
    assert!(
        (value - expected).abs() <= 1e-10 * expected.abs().max(1.0),
        "parity failed: {value} vs {expected}"
    );
}

fn percentile(values: &mut [f64], p: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    values[((values.len() as f64 * p).ceil() as usize).saturating_sub(1)]
}

fn timed(op: usize, a: &[f64], b: &[f64], expected: f64) -> f64 {
    if op == 0 {
        // Batch tiny direct calls to reduce timer overhead; black_box inputs
        // and outputs prevent eliminating repeated identical calculations.
        let reps = if a.len() <= 1024 { 256 } else { 16 };
        let start = Instant::now();
        for _ in 0..reps {
            black_box(numbam::dot(black_box(a), black_box(b)).unwrap());
        }
        return start.elapsed().as_secs_f64() * 1e9 / reps as f64;
    }
    // Time owned-vector preparation separately. Exclude it from the SDK calls
    // below, but include all conversion/dispatch inside the generated SDK.
    if op == 1 {
        let start = Instant::now();
        let owned = black_box((a.to_vec(), b.to_vec()));
        let elapsed = start.elapsed().as_secs_f64() * 1e9;
        black_box(owned);
        return elapsed;
    }
    let aa = a.to_vec();
    let bb = b.to_vec();
    let start = Instant::now();
    let result = match op {
        2 => bridge_probe_sync(aa, bb).expect("BAML probe"),
        3 => call_native_dot_sync(|_: Vec<f64>, _: Vec<f64>| 0.0, aa, bb)
            .expect("BAML callback probe"),
        4 => pure_dot_sync(aa, bb).expect("BAML dot"),
        5 => call_native_dot_sync(
            |x: Vec<f64>, y: Vec<f64>| numbam::dot(&x, &y),
            aa,
            bb,
        )
        .expect("BAML native dot"),
        _ => unreachable!(),
    };
    let elapsed = start.elapsed().as_secs_f64() * 1e9;
    close(result, if op <= 3 { 0.0 } else { expected });
    elapsed
}

fn main() {
    println!("size,operation,p50_ns,p95_ns,samples");
    let names = [
        "rust_dot",
        "host_vec_clone",
        "baml_probe",
        "callback_probe",
        "baml_dot",
        "callback_dot",
    ];
    for n in SIZES {
        // Exactly representable small fractional inputs avoid artificial
        // cancellation and permit deterministic numerical parity checks.
        let a: Vec<f64> = (0..n).map(|i| (i % 31) as f64 / 8.0).collect();
        let b: Vec<f64> = (0..n).map(|i| (i % 17) as f64 / 16.0).collect();
        let expected = numbam::dot(&a, &b).unwrap();
        let mut timings: [Vec<f64>; 6] = std::array::from_fn(|_| Vec::with_capacity(SAMPLES));
        for sample in 0..WARMUPS + SAMPLES {
            // Rotate order to limit systematic first/last measurement bias.
            for j in 0..names.len() {
                let op = (sample + j) % names.len();
                let ns = timed(op, &a, &b, expected);
                if sample >= WARMUPS {
                    timings[op].push(ns);
                }
            }
        }
        for (name, values) in names.iter().zip(timings.iter_mut()) {
            let p50 = percentile(values, 0.50);
            let p95 = percentile(values, 0.95);
            println!("{n},{name},{p50:.2},{p95:.2},{}", values.len());
        }
    }
}
