# Native inference runtime

The repository root is the inference-only BAML 0.26.0 package. Historical training and numerical experiments are archived at [pre-inference-cleanup](https://github.com/mmmaillot/baml-ml/tree/pre-inference-cleanup).

For the trunk-based development workflow, Kanban issue lifecycle, and pre-push checks (for both humans and coding agents), see [AGENTS.md](../AGENTS.md).

## Quick start: one-command setup and verification

```sh
./scripts/install_baml.sh   # downloads + SHA-256-verifies the pinned BAML 0.26.0 baml-cli into .toolchain/
./scripts/verify.sh         # fast pre-PR checks: baml check + baml test
./scripts/verify.sh --full  # also runs the experiment harnesses under experiments/
```

The harnesses use only portable POSIX shell utilities (plus `jq` and `bash`), so `--full` also works with the stock macOS utilities on Apple Silicon.

`scripts/install_baml.sh` verifies the SHA-256 sidecar of the official [BAML 0.26.0 release](https://github.com/BoundaryML/baml/releases/tag/baml-language-0.26.0) archive for `linux-x86_64`, `linux-arm64` or `macos-arm64` before extraction (auto-detected, or forced with `--platform`). Useful flags: `--pack-host` also installs `baml-pack-host` (needed by `verify.sh --full` for the standalone package), `--dest DIR` changes the install location. The script prints the `BAML_CLI`/`BAML_PACK_HOST` paths; inside GitHub Actions it exports them automatically (see [setup-baml](../.github/actions/setup-baml/action.yml), used by every workflow). If you install the toolchain yourself elsewhere, point `BAML_CLI` at the absolute executable path instead:

```sh
export BAML_CLI=/absolute/path/to/baml-cli
BAML_TELEMETRY=off "$BAML_CLI" check
BAML_TELEMETRY=off "$BAML_CLI" test --max-concurrency 1 --test-timeout 5m
"$BAML_CLI" run ImportedIrisProbe --output-format json
"$BAML_CLI" run ImportedDiabetesProbe --output-format json
"$BAML_CLI" run ImportedBinaryProbe --output-format json
```

The example probes are defined in the native test file and use committed fitted JSON artifacts. See [artifact export and inference API](../artifacts/README.md).

## Repository overview

- `baml_src/`: the inference-only BAML runtime sources (see below); `baml.toml` is the package manifest.
- `artifacts/`: fitted JSON artifact fixtures, the sklearn export adapter and the independent oracle generators.
- `scripts/`: contributor tooling — `install_baml.sh` (verified toolchain install), `verify.sh` (local checks), `stage_runtime.sh` (copies non-test `baml_src/**/*.baml` into a staged project, recursively), `check_architecture.py` (baml_src layer linter, also in CI), `btel_jq.jq` (`named_rows` helper mapping BTEL SQL query results to objects keyed by column alias for `jq` assertions).
- `experiments/`: runnable evidence harnesses (`imported_inference`, `native_ml_observability`, `heterogeneous_pipelines`, `standalone`, plus manual benchmarks). They stage the runtime with `scripts/stage_runtime.sh` and assert on BTEL recordings with `scripts/btel_jq.jq`.
- `.github/actions/setup-baml/`: composite action wrapping `scripts/install_baml.sh`; all workflows use it instead of inline download/checksum blocks.

## Source layout

`baml_src/` is organized into five unidirectional layers plus a test folder. `scripts/check_architecture.py` (run by `verify.sh` and CI) checks source placement, keeps tracing out of the two lowest layers, bans `_vN` file-name suffixes, and limits runtime files to 300 lines. The packaged sources must contain the frozen BTEL span functions and CLI entry points with their `@contract(btel-v1)` and `@public-api` annotations.

- `kernels/`: pure math and tabular transforms, no tracing — `primitives.baml` (shape/finiteness checks, dot product, float predicates, and the shared `InferenceFailure` type), `tabular.baml` (`Cell[][]` container and conversions), `imputer.baml`, `scaler.baml`, `one_hot.baml`, `column_transformer.baml`, `linear.baml`, `logistic.baml` (stable softmax), `tree.baml`, `forest.baml`, `legacy_pipeline.baml` (frozen two-step schema 1 pipeline).
- `contracts/`: data-only JSON schemas, decoding and static artifact/type-chain validation, no tracing — `invariants.baml` (single-source-of-truth static invariants shared by all schema versions, plus the internal `Operation`/`ColumnKind` enums parsed once at wire boundaries), `legacy.baml` (frozen schema 1 & 2 artifacts) and `pipeline.baml` and `pipeline_validation.baml` (schema 3 arbitrary-length pipelines).
- `observability/`: `policy.baml` (`ObservationMode`, `TracePrivate<T>`) and the legacy and pipeline stage spans in `legacy_spans.baml` / `pipeline_spans.baml`.
- `runtime/`: the single execution engine `executable_model.baml` (`ExecutableModel` + per-family builders, shared by direct, Adaptive and shadow execution), direct BTEL entry spans (`direct_spans.baml`), selective orchestration `selective.baml`, `adaptive.baml` (policy), `triggers.baml`, `shadow.baml`.
- `surface/`: factory-owned public handles (`loaded_models.baml`, `adaptive_handles.baml`) and the standalone CLI entry points (`cli.baml`: `predict`, `shadow`, `RunBatch`, `RunShadow`).
- `tests/`: all `*_test.baml` suites and test/CI probes (`ImportedIrisProbe`, `TreeOracleProbe`, `PipelineOracleProbe`, ...), including `conformance_test.baml`, the native runner of the declarative negative conformance catalog `artifacts/conformance/invalid_mutations.json` (regenerate with `python3 artifacts/conformance/generate.py`).

`schema_version` is wire-level JSON data only; it never appears in module, type or file names. Wire JSON keeps plain strings (`operation`, `input_kinds`); the `Operation`/`ColumnKind` enums exist only inside the process, parsed once at CLI/JSON entry and rendered back with `OperationName` at the wire.

Native float comparisons use a total order; check NaN with IsNaN/IsFinite rather than x != x. Internal trusted finite-input paths are only called after preceding successful preprocessing; public transforms remain checked.

The active CI runs native checks and imported inference/trace validation. Full numerical and performance campaigns will be rebuilt for imported models and run manually.
