# BAML → Rust integration

Status: isolated host-callback proof of concept added; CI validation pending. Do not add a fake BAML adapter or silently route calculations through Python.

baml-ml pins the CLI and baml-py to 0.226.3. The newer BAML language documentation describes host callbacks, native Rust compilation and standalone packaging, but those capabilities have not been verified in this pinned toolchain.

## Isolated experiment

The current MVP remains on 0.226.3. This experiment uses language toolchain 0.25.0, whose [official Rust host-callable tests](https://github.com/BoundaryML/baml/blob/baml-language-0.25.0/baml_language/sdk_tests/crates/rust/function_calls/customizable/test_host_callables.rs) demonstrate typed callbacks invoked from BAML. The upstream [BAML fixture](https://github.com/BoundaryML/baml/blob/baml-language-0.25.0/baml_language/sdk_tests/fixtures/function_calls/baml_src/ns_host_callable_tests/main.baml) and [SDK verifier](https://github.com/BoundaryML/baml/blob/baml-language-0.25.0/.github/workflows/verify-rust-sdk.reusable.yaml) establish the SDK generation and loading pattern.

The BAML function receives a typed callback, invokes it with two vectors, and returns the result. The Rust host registers a closure invoking the actual NumBam library. The test checks the result, callback invocation count and length-mismatch error propagation. There is no Python or numerical subprocess.

With the new BAML wrapper installed and Rust available:

```bash
cd numbam/interop
BAML_VERSION=0.25.0 baml check
BAML_VERSION=0.25.0 baml generate
cargo run --manifest-path consumer/Cargo.toml
```

The generated Rust SDK uses BAML's bridge shared library; success here does not establish a single self-contained binary. The bridge dependency is confined to this consumer, not NumBam itself. Do not mark the packaging, trace or performance criteria complete.

## First proof of concept

1. Confirm a supported BAML-to-Rust call mechanism in the pinned toolchain, or pin a newer toolchain in an isolated experiment. Do not change the existing benchmark baseline.
2. Expose NumBam's dot through a Rust host callback or a supported native binding. Invoke it from a BAML deterministic function with two float arrays. No LLM, subprocess, Python numerical callback or network call.
3. Check success, length mismatch and nonfinite input behavior. Document how host errors map to the existing BAML result classes.
4. Capture an actual duration/status trace at the boundary, with telemetry both enabled and disabled.
5. Measure call and conversion overhead for small and large vectors and compare with the pure BAML dot. Include allocation cost.
6. Build a single executable that includes both the BAML program and the Rust crate. Run it in a clean target environment and inspect runtime dependencies.

Only after this proof succeeds should we integrate exp and log into logistic regression. Batch exponentials at the row or larger level where possible. Do not replace the entire optimizer or solver.

References: [runtime architecture](https://boundaryml.com/how-baml-works), [native compilation and host callbacks](https://boundaryml.com/blog/baml-0.24.0), [adoption and standalone packaging](https://boundaryml.com/adopt).

Record actual commands, versions and results here once tested.
