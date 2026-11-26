// The callback is supplied by the Rust host. BAML invokes it in-process.
function call_native_dot(callback: (float[], float[]) -> float, a: float[], b: float[]) -> float {
    callback(a, b)
}

// Matched argument signature: measures host-to-BAML dispatch and conversion.
function bridge_probe(a: float[], b: float[]) -> float {
    0.0
}

// Same ordered accumulation as the Rust kernel. Benchmark inputs have
// matching lengths, so no additional validation is timed in this baseline.
function pure_dot(a: float[], b: float[]) -> float {
    let sum = 0.0;
    for (let i = 0; i < a.length(); i += 1) {
        sum = sum + a[i] * b[i];
    }
    sum
}
