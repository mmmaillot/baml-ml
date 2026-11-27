# Native inference runtime

The repository root is the inference-only BAML 0.26.0 package. Historical training and numerical experiments are archived at [pre-inference-cleanup](https://github.com/mmmaillot/baml-ml/tree/pre-inference-cleanup).

For the trunk-based development workflow, Kanban issue lifecycle, and pre-push checks (for both humans and coding agents), see [AGENTS.md](../AGENTS.md).

## Install the pinned toolchain

Download the official [BAML 0.26.0 release](https://github.com/BoundaryML/baml/releases/tag/baml-language-0.26.0). Linux CI verifies the SHA-256 sidecar for the x86-64 archive before extraction; see [native checks](../.github/workflows/native-checks.yml). Set BAML_CLI to the absolute path of the extracted executable.

```sh
"$BAML_CLI" check
"$BAML_CLI" test --max-concurrency 1 --test-timeout 5m
"$BAML_CLI" run ImportedIrisProbe --output-format json
"$BAML_CLI" run ImportedDiabetesProbe --output-format json
"$BAML_CLI" run ImportedBinaryProbe --output-format json
```

The example probes are defined in the native test file and use committed fitted JSON artifacts. See [artifact export and inference API](../artifacts/README.md).

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
