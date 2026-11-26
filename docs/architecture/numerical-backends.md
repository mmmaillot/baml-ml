# Numerical backends: pure BAML vs. native acceleration

Status: **Proposed / exploratory (2026-11-26)**. This note records an architectural discussion, not a decision to change the v0.1 implementation or roadmap.

## Context and motivation

baml-ml is an experiment in classical ML with runtime-native observability. The current implementation deliberately performs training and inference in BAML, with Python used for dataset preparation, tests, reference estimators, and benchmark orchestration.

The repository already implements scalar math (Sqrt, Exp, Log), dense vector and matrix primitives, preprocessing, QR-based ordinary least squares, and multinomial logistic regression. The pinned toolchain (BAML CLI and baml-py 0.226.3) uses beta expression functions; its documented lack of several numerical primitives explains why they were implemented in BAML.

This is useful for understanding the language and establishing a reference implementation, but it is not necessarily the most efficient architecture for a production ML library. NumPy/SciPy and scikit-learn similarly delegate much of their numerical work to compiled implementations, including BLAS and LAPACK where appropriate.

## Architectural principle

Observability should follow meaningful computational boundaries, not necessarily every arithmetic operation. An optimized native kernel can be a black box internally while exposing a BAML-level operation with a clear name, duration, status, shapes and selected diagnostics. Whether the BAML runtime captures these automatically at the chosen boundary must be demonstrated, not assumed.

For example, tracing a least-squares solve or a preprocessing stage can be more useful than recording every multiply or addition. Conversely, replacing an entire optimizer with a native call can hide convergence iterations; keep iteration-level orchestration in BAML if that diagnostic visibility is required, or explicitly expose appropriate diagnostics from the backend.

Proposed layering:

```text
baml-ml: estimators, pipeline, optimizer orchestration
                 |
      observable numerical operations
                 |
       +---------+----------+
       |                    |
  pure BAML backend    native backend (candidate)
  reference kernels    Rust/libm + BLAS/LAPACK
```

The native backend is optional and hypothetical. No dependency or production acceleration is being added by this note.

## Candidate operations

| Operation | Current / reference approach | Native candidate |
| --- | --- | --- |
| Sqrt, Exp, Log | Newton / series approximations | Rust f64 methods or libm |
| Dot | BAML loops | BLAS ddot |
| Matrix-vector multiplication | BAML loops | BLAS dgemv |
| Matrix-matrix multiplication (if needed) | BAML loops | BLAS dgemm |
| QR decomposition | Modified Gram-Schmidt | LAPACK dgeqrf (Householder QR) |
| Least-squares solve | QR + back substitution | LAPACK dgels or rank-aware dgelsd |
| Column statistics / elementwise operations | BAML loops | Rust vectorized kernels where justified |

BLAS does not provide generic exp/log/sqrt or all preprocessing operations. LAPACK solvers may have different rank handling and numerical properties from the current solver; preserve the existing public contracts or explicitly document intentional changes. A Rust-native linear algebra crate is also a candidate if it simplifies builds and portability. Pick implementations based on profiling rather than introducing all of them at once.

## Interoperability and open questions

BoundaryML documents an embedded runtime, host callbacks and a Rust compilation backend in its newer language/toolchain documentation. Host callback support in the 0.2x.y documentation is not proof of equivalent support in the pinned 0.226.3 package. Calling a BAML function from Rust is also not the same as calling a native function from BAML.

Before implementing acceleration:

1. Verify the supported native-call mechanism for the exact version in use (including direction of the call, packaging, callback lifetime and error semantics). If necessary evaluate a newer toolchain on a separate branch, preserving the reference baseline.
2. Run a minimal BAML-to-native numerical callback, without LLMs, networking or API keys. Verify that the native call boundary and error/duration can actually be observed. Do not silently substitute Python for numerical computation.
3. Investigate memory representation: the current row-major float[][] is not a guaranteed contiguous float64 buffer. Quantify conversion, allocation, copying and FFI overhead. Check layout/stride requirements and BLAS/LAPACK integer ABI. Prefer coarse-grained operations and contiguous buffers where the runtime supports them.
4. Evaluate platform support, dependency distribution, threading/SIMD, reproducibility, and licenses. Start with one platform and one kernel if needed.
5. Preserve shape validation, missing-value handling, finiteness checks and numerical parity. Reuse the current tests and add differential tests against both backends and scikit-learn.

Avoid assuming zero-copy interop, native Rust compilation automatically yielding BLAS-level performance, or traces continuing inside external kernels. All require validation. Record only appropriate inputs/outputs; prefer metadata or summaries over indiscriminately recording full matrices.

## Benchmark design

Complete the existing v0.1 benchmark first on Iris and Diabetes: sklearn vanilla, sklearn + OpenTelemetry, pure BAML without observations, and pure BAML with observations. Verify the actual runtime's observation controls and emitted data; a CLI analytics opt-out is not necessarily a runtime tracing switch.

If native acceleration proves feasible, add two more variants:

| Backend | Observation off | Observation on |
| --- | --- | --- |
| scikit-learn | yes | OpenTelemetry |
| Pure BAML | yes | BAML runtime |
| BAML + native kernels | yes | BAML runtime at the operation boundary |

Compare identical input data and split, model objectives, comparable trace granularity/payloads, warm-up, hardware, software versions and thread settings. Measure fit time, single-row and batch inference latency (p50/p99), throughput, allocations/copy cost where available and trace volume. Report within-backend observation overhead separately from cross-backend differences. For a fair kernel-level comparison, hold algorithms and convergence criteria fixed where practical; a different solver or toolchain is a separate experimental variable. If equivalent tracing cannot be achieved, report that limitation rather than presenting a misleading comparison.

The small MVP datasets establish feasibility and parity, not large-scale performance. Later, synthetic matrix sizes can expose the crossover at which native kernels outweigh bridge overhead.

## Potential standalone project: baml-numerics

A reusable numerical layer could serve classical ML, statistics, optimization, signal processing, time series and embedding operations beyond baml-ml.

A preliminary public search on 2026-11-26 for repositories named baml-numerics / baml-math and BAML numerical or linear algebra libraries did not identify a maintained NumPy-like library or ready-to-use BLAS/LAPACK binding for BoundaryML's language. This is not exhaustive proof of absence. Check upstream standard packages, the current language repository, package registries and ask BoundaryML maintainers before starting a separate project; prefer contributing to an existing library if one exists.

If a gap is confirmed, keep the first version narrow: a tested numerical API, pure BAML reference implementation and one accelerated native operation. Extract reusable primitives only after the baml-ml experiment establishes the need and the integration mechanism is proven.

## Recommended sequence

- Keep the current pure BAML implementation and finish the v0.1 roadmap (#1, #8–#11).
- Profile the actual workload and identify hot operations.
- Investigate native interoperability with a small proof of concept.
- Add an optional accelerated backend only if it preserves meaningful observation and materially improves end-to-end performance.
- Consider a separate baml-numerics project or upstream contribution after those results.

## References and evidence

- [Current pure BAML implementation](../../baml_src/primitives.baml) and [toolchain limitations](../../README.md).
- [BoundaryML: How BAML works](https://boundaryml.com/how-baml-works).
- [BoundaryML: BAML 0.24.0, Rust compilation and host-call ABI](https://boundaryml.com/blog/baml-0.24.0).
- [BAML v0.25.2 standard package: host callback API](https://developer.boundaryml.com/baml/packages/v0.25.2/baml/host/call_host_value).
- [BoundaryML standard packages and language reference](https://developer.boundaryml.com/baml).
- [BLAS](https://www.netlib.org/blas/) and [LAPACK](https://www.netlib.org/lapack/).

These upstream references describe possible directions, not verified compatibility with baml-ml's pinned toolchain.
