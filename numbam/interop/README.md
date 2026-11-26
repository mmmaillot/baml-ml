# BAML → Rust integration

Status: not implemented. Do not add a fake BAML adapter or silently route calculations through Python.

baml-ml pins the CLI and baml-py to 0.226.3. The newer BAML language documentation describes host callbacks, native Rust compilation and standalone packaging, but those capabilities have not been verified in this pinned toolchain.

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
