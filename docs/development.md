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
- `scripts/`: contributor tooling — `install_baml.sh` (verified toolchain install), `verify.sh` (local checks), `stage_runtime.sh` (copies non-test `baml_src/**/*.baml` into a staged project, recursively), `btel_jq.jq` (`named_rows` helper mapping BTEL SQL query results to objects keyed by column alias for `jq` assertions).
- `experiments/`: runnable evidence harnesses (`imported_inference`, `native_ml_observability`, `heterogeneous_pipelines`, `standalone`, plus manual benchmarks). They stage the runtime with `scripts/stage_runtime.sh` and assert on BTEL recordings with `scripts/btel_jq.jq`.
- `.github/actions/setup-baml/`: composite action wrapping `scripts/install_baml.sh`; all workflows use it instead of inline download/checksum blocks.

## Source layout

- artifact.baml: versioned data contract, JSON decoding, validation and imported prediction entry points.
- imputer.baml, scaler.baml: fitted preprocessing state and transforms.
- linreg.baml, logistic.baml: fitted inference, including stable softmax.
- pipeline.baml: checked preprocessing and prediction helpers.
- loaded_model.baml, observation.baml: prepared model handles and semantic trace boundaries.
- batch.baml: standalone JSON batch entry point.
- primitives.baml: shape and finite-value checks, dot product and native float predicates.
- *_test.baml: imported-model parity, validation, numerical and inference contract tests.

Native float comparisons use a total order; check NaN with IsNaN/IsFinite rather than x != x. Internal trusted finite-input paths are only called after preceding successful preprocessing; public transforms remain checked.

The active CI runs native checks and imported inference/trace validation. Full numerical and performance campaigns will be rebuilt for imported models and run manually.
