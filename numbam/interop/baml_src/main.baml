// The callback is supplied by the Rust host. BAML invokes it in-process.
function call_native_dot(callback: (float[], float[]) -> float, a: float[], b: float[]) -> float {
    callback(a, b)
}
