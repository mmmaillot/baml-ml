# NumBam

Fast numerical computing. Observable by design.

A minimal Rust numerical backend for baml-ml, not a NumPy clone. It lives in this repository until there is a concrete reason to extract it.

## Current code

- `dot(&[f64], &[f64])`: dense dot product with length validation.
- `exp(&[f64])`: vectorized exponential.
- `log(f64)`: natural logarithm.

No third-party dependencies. The existing pure BAML backend is unchanged. The native functions are not yet callable from BAML; tests cover Rust behavior only. In particular, native log follows IEEE domain semantics, whereas the current BAML implementation uses a finite sentinel for nonpositive inputs. Native exp may retain subnormals lost by the reference implementation. The BAML adapter must account for intentional differences.

## Run

From the repository root, with a Rust toolchain installed:

```bash
cargo test --manifest-path numbam/Cargo.toml --offline
cargo build --manifest-path numbam/Cargo.toml --release --example smoke --offline
./numbam/target/release/examples/smoke
```

The smoke executable exercises only the Rust library. It is not a packaged BAML model or proof of native BAML interop. Inspect dynamic system-library requirements when testing deployment.

## Definition of done

Tracked in [issue #20](https://github.com/mmmaillot/baml-ml/issues/20):

- [ ] BAML can call Rust.
- [ ] Numerical parity with the reference backend.
- [ ] Measurable acceleration of at least one existing pipeline.
- [ ] Measured dispatch, allocation and data conversion overhead.
- [ ] Actual observability at the native operation boundary.
- [ ] Standalone BAML + Rust executable, with no Python or separately installed numerical libraries on the target.

See [interop notes](interop/README.md) for the first unresolved gate and [architecture](../docs/architecture/numerical-backends.md) for the full plan. Finish NumBam before building the embedded inference runtime.
