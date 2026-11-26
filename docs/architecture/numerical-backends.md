# Numerical backends: pure BAML vs. native acceleration

Status: **Proposed / exploratory (2026-11-26)**. Updated with the minimal NumBam scope, definition of done, and embedded inference follow-up. This note records an architectural discussion, not a decision to change the v0.1 implementation or roadmap.

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

## NumBam: native numerical backend inside baml-ml

NumBam initially lives in the same repository under [numbam/](../../numbam/README.md). Its sole current purpose is to support our ML experiments; a standalone project is not planned for v0.1. A reusable layer may eventually serve other scientific use cases, but those are not current requirements.

A preliminary public search on 2026-11-26 for repositories named baml-numerics / baml-math and BAML numerical or linear algebra libraries did not identify a maintained NumPy-like library or ready-to-use BLAS/LAPACK binding for BoundaryML's language. This is not exhaustive proof of absence. Check upstream standard packages, the current language repository, package registries and ask BoundaryML maintainers before starting a separate project; prefer contributing to an existing library if one exists.

Keep the first version narrow: a tested Rust numerical API, the existing pure BAML reference implementation, and a real native interoperability experiment.

### Positioning and identity

**NumBam** (working name): Numerical computing for BAML.

*Fast numerical computing. Observable by design.*

Proposed GitHub description: "The fundamental library for numerical computing in BAML, powered by Rust."

NumBam should be a NumPy-like foundation rather than a complete port of NumPy or SciPy. Its value is a small, efficient, BAML-facing numerical API built on existing native libraries, with meaningful operation boundaries for BAML observability. Rust is the preferred implementation language for the native layer because BAML's runtime is itself written in Rust, but this alone does not establish a supported extension mechanism.

The name NumBam was preferred over RayBam (an array/BAML pun) because the latter deliberately evokes the Ray-Ban brand. NumBam is a working name, not a legally cleared or reserved package name. Verify availability on GitHub, crates.io and npm and check relevant trademark databases before launch. Do not copy the identity of an existing brand.

### NumPy first, SciPy second

NumPy supplies the core array abstraction, shape/dtype conventions, broadcasting, elementwise operations, reductions and basic dense linear algebra. SciPy builds on NumPy arrays with specialized scientific algorithms, including more extensive linear algebra, optimization, sparse matrices, statistics and signal processing. Both can rely on native numerical kernels.

NumBam prioritizes only the numerical operations needed by the current baml-ml code. A broader SciPy-like layer, tentatively named **SciBam**, is explicitly deferred.

| Layer | Proposed project | Responsibility |
| --- | --- | --- |
| ML algorithms and pipelines | baml-ml | Training, inference, preprocessing, domain-level diagnostics |
| Scientific algorithms | SciBam (possible later project) | Optimization, convergence diagnostics, special functions and scientific workflows |
| Numerical foundation | NumBam | Dense arrays, elementwise functions, reductions and linear algebra |
| Native implementation | Rust and existing numerical libraries | Storage, vectorized kernels, decompositions and solves |

Dependency direction: baml-ml and any future SciBam use NumBam; NumBam must not depend on either. baml-ml can use NumBam directly without waiting for SciBam.

### Candidate Rust stack

| Library | Role | Decision |
| --- | --- | --- |
| [ndarray](https://github.com/rust-ndarray/ndarray) | N-dimensional arrays, shape/stride management, array operations | Candidate foundation |
| [ndarray-linalg](https://github.com/rust-ndarray/ndarray-linalg) | LAPACK-backed dense linear algebra for ndarray | Candidate if external BLAS/LAPACK packaging is acceptable |
| [faer](https://github.com/sarah-quinones/faer-rs) | High-performance Rust-native matrix algorithms and factorizations | Alternative linear algebra backend; evaluate conversions and portability |
| Rust f64 methods / [libm](https://crates.io/crates/libm) | Scalar mathematical functions | Prefer existing implementations over reimplementing series expansions |

Do not build another BLAS or LAPACK implementation. Compare the Rust-native route (e.g. faer) against a BLAS/LAPACK-backed route where useful; select the smallest viable dependency set, not every candidate at once. In particular, ndarray and faer have different array representations, so benchmark conversions before combining them.

### NumBam v0.1: scope and definition of done

Tracking: [issue #20](https://github.com/mmmaillot/baml-ml/issues/20).

The next priority is a minimal native backend, followed by an embedded inference runtime. Keep NumBam in this repository. Do not build a general-purpose NumPy replacement.

| Priority | Operation | Why |
| --- | --- | --- |
| P0 | dot | Both regressions and inference |
| P0 | vectorized exp | Logistic regression softmax and optimization |
| P0 | log | Logistic regression loss and log-sum-exp |
| Conditional | sqrt, matvec | Only if profiling demonstrates a need |
| Deferred | mean, variance, least squares | Keep current BAML implementations until evidence justifies replacing them |

Use Rust standard-library functions and slices first. No external BLAS/LAPACK installation, generic array framework, Python numerical callbacks, or SciBam. Existing optimized Rust crates are options only if justified by measured workloads. Preserve the current QR solver and logistic optimizer initially; do not confound the experiment with a change of algorithms.

The initial crate implements dot, vectorized exp and log as dependency-free Rust functions; the existing BAML estimators do not yet call them. Their tests and a Rust-only smoke example do not prove BAML interop or trace support.

**Definition of done (all six required):**

- [ ] BAML can call a Rust function using a supported native interop mechanism.
- [ ] Selected primitives pass differential numerical parity tests against BAML and/or NumPy.
- [ ] At least one existing baml-ml pipeline benefits measurably from acceleration and preserves numerical parity.
- [ ] The bridge overhead is measured, including conversion/allocation/dispatch.
- [ ] Native operations remain observable at their BAML boundary, demonstrated with actual trace or profiling evidence.
- [ ] A standalone executable containing BAML and the Rust kernels runs without Python, Cargo or separately installed numerical libraries on the target.

A direct Rust executable is a useful packaging smoke test, but it does not satisfy the last criterion. Use the supported BAML-to-host integration mechanism, not a subprocess or an unverified FFI shim. Keep the existing pure-BAML runtime pinned; if the host-call mechanism requires a newer toolchain, evaluate that separately.

### Embedded inference runtime (next phase)

After the six criteria are met, investigate exporting fitted classical ML pipelines to a portable, observable runtime. The first target should be inference of the existing linear/logistic regression pipelines; there is no need to bring training-only decompositions or optimizers into the deployment artifact. Prefer a self-contained executable per platform, no Python or separately installed numerical libraries on the target, offline execution and local observability. Compare with ONNX Runtime for the supported models; ONNX compatibility is a future option, not an MVP prerequisite. A combined BAML/Rust executable has not yet been demonstrated.

### Design boundary

The preferred architecture is a native NumPy-like foundation and a BAML-first, hyper-observable scientific/ML layer. A native QR or exp kernel need not expose its internal arithmetic. A BAML optimizer should retain meaningful iteration-level diagnostics rather than hide its entire execution in a native solver, unless an explicit diagnostic interface provides equivalent value. Native handles, zero-copy buffers, and automatic traces across the language boundary are design hypotheses to verify, not current features.

The pure BAML code in baml-ml remains an important reference and experimental control; NumBam must not invalidate the current MVP or its four-way benchmark.

## Recommended sequence

- Keep the current pure BAML implementation and finish the v0.1 roadmap (#1, #8–#11).
- Profile the actual workload and identify hot operations.
- Investigate native interoperability with a small proof of concept.
- Add an optional accelerated backend only if it preserves meaningful observation and materially improves end-to-end performance.
- Keep NumBam in this repo; move to an embedded inference runtime once issue #20 is complete. SciBam is explicitly deferred.

## References and evidence

- [Current pure BAML implementation](../../baml_src/primitives.baml) and [toolchain limitations](../../README.md).
- [BoundaryML: How BAML works](https://boundaryml.com/how-baml-works).
- [BoundaryML: BAML 0.24.0, Rust compilation and host-call ABI](https://boundaryml.com/blog/baml-0.24.0).
- [BAML v0.25.2 standard package: host callback API](https://developer.boundaryml.com/baml/packages/v0.25.2/baml/host/call_host_value).
- [BoundaryML standard packages and language reference](https://developer.boundaryml.com/baml).
- [BLAS](https://www.netlib.org/blas/) and [LAPACK](https://www.netlib.org/lapack/).

These upstream references describe possible directions, not verified compatibility with baml-ml's pinned toolchain.
