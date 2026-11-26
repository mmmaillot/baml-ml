# baml-ml

**Classical machine learning. Observable by design.**

An experimental library for classical machine learning in [BAML](https://github.com/BoundaryML/baml). The goal is to evaluate the usefulness and cost of runtime-native observability compared with manually instrumented scikit-learn pipelines.

## Toolchain

Exact versions used and verified:

| Component | Version | Install |
| --- | --- | --- |
| BAML CLI (`baml`) | 0.226.3 | `npm install` in this repo (pinned in `package.json`), then use `npx baml ...`; or `npm install -g @boundaryml/baml@0.226.3` |
| BAML Python runtime (`baml-py`) | 0.226.3 | `pip install -r requirements.txt` |
| Node.js | 22.x | required by the BAML CLI |
| Python | 3.12 | required by `baml-py` and the test harness |

Setup from a clean checkout:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
npm install
```

Commands:

```bash
npx baml --features beta check      # type-check baml_src/ (no LLM, no API key)
npx baml --features beta generate   # regenerate the Python client into baml_client/ (gitignored)
.venv/bin/python -m pytest tests/   # run the Python test harness (auto-(re)generates baml_client/ when needed)
.venv/bin/python examples/dot_example.py  # run one deterministic BAML function from Python
```

The `--features beta` flag is required: expression functions with statements
(`let`, `if`, `for`) are a beta feature of the 0.226.3 CLI. The generated
client embeds the compiled artifact directly into `./baml_client/`, so the
Python side needs no flag and no LLM client is defined anywhere — no API key
is ever required for these tests.

## Data representation and conventions

Shared by all estimators (implemented in `baml_src/primitives.baml`):

- **Vector**: `float[]`. **Matrix**: `float[][]`, row-major (a list of rows).
- **Missing and finite values**: IEEE-754 `NaN` in a `float` cell represents a
  missing value, detected with `IsNaN(x)` (`x != x`). Finite values are checked
  in pure BAML with `IsFinite(x)` (`(x - x) == 0.0`, which rejects both `NaN`
  and `±Inf`). NaN round-trips through the `baml-py` bridge (verified for
  scalars, vectors and matrices in 0.226.3), so BAML optionals (`float?`) are
  not needed; note that 0.226.3 beta expression functions cannot inspect
  `null`, which is why NaN is the missing-value representation.
- **Numeric types**: `int` for loop indices and counts, `float` for all data.
  The beta expression language has strict typing: `int` and `float` never mix
  implicitly and there is no int→float conversion, so counters that get
  divided are kept as `float` (`n = n + 1.0`).
- **Shape and finiteness validation**: every public entry point validates
  shapes (non-empty, rectangular matrix; matching vector lengths; matching
  feature counts) via `CheckMatrix` / `CheckFiniteMatrix` and related guards.
- **Error convention & internal helpers**: BAML 0.226.3 expression functions
  cannot `throw`, `@assert` does not fire for computed (non-LLM) functions, and
  unions cannot be narrowed with `instanceof`. Public entry points therefore
  return a `*Result` class with non-nullable fields: `ok bool`, a payload, and
  `error string` (`""` on success). Python callers check `.ok` / `.error`; BAML
  callers check `ok`. Error messages are static strings (no string
  interpolation of numbers in the beta expression language). Because BAML has
  no `private` visibility modifier, internal BAML-to-BAML helpers (`IsNaN`,
  `IsFinite`, `*Unchecked`, `Empty*`) are also generated on `b` — they return
  raw values and are not part of the public Python API.

## Implemented so far

- `baml_src/primitives.baml`: `IsNaN`, `IsFinite`, `Sqrt` (Newton's method),
  `VecNorm`, `CheckMatrix`, `CheckFiniteMatrix`, `Dot`, `ColumnMeans`,
  `ColumnVariances` (population, ddof=0), `VecAdd`, `VecSub`, `VecScale`,
  `MatVec`, `Transpose`. All arithmetic runs in BAML.
- `baml_src/imputer.baml`: mean `SimpleImputer` — `ImputerFit`,
  `ImputerTransform`, `ImputerFitTransform` with an explicit `MeanImputer`
  fitted state (per-column means from finite non-missing training values only).
  Entirely missing training columns, `±Inf` inputs, non-finite means, and
  malformed `MeanImputer` states are rejected with documented errors (no silent
  feature dropping); transform preserves shape and finite non-missing entries
  and always uses the stored training statistics. Parity with
  `sklearn.impute.SimpleImputer(strategy="mean")` is tested within 1e-12 on
  supported fixtures (no all-missing columns, which sklearn drops by default).
- `baml_src/scaler.baml`: `StandardScaler` — `ScalerFit`, `ScalerTransform`,
  `ScalerFitTransform` with an explicit `StandardScaler` fitted state (per-feature
  mean and population variance, ddof=0; scale = `Sqrt(variance)`, and scale = 1
  for zero-variance columns). Input must be finite and already imputed; transform
  validates dimensions and uses the stored training statistics. Parity with
  `sklearn.preprocessing.StandardScaler` is tested within 1e-9.
- `baml_src/linreg.baml`: `LinearRegression` (ordinary least squares with
  intercept) — `LinearRegressionFit`, `LinearRegressionPredict` with an explicit
  `LinearRegressionModel` state (coefficients, intercept, feature count). Solved by
  QR decomposition of the design matrix `[1 | X]` via modified Gram-Schmidt plus
  back substitution; rank-deficient input (post-orthogonalization column norm
  <= 1e-10 times its original norm) is rejected explicitly. Parity with
  `sklearn.linear_model.LinearRegression` is tested within 1e-6 on well-conditioned
  fixtures.
- `baml_src/pipeline.baml`: minimal `Pipeline` (Imputer → Scaler → LinearRegression)
  with an explicit fitted `Pipeline` state — `PipelineFit`, `PipelinePreprocess`
  (named impute-then-scale step), `PipelinePredict`. Fitting is sequential on
  training data only; inference applies the stored states and never updates them.
  The classification variant (`predict_proba`) lands with issue #6
  (LogisticRegression). Parity with the equivalent sklearn `Pipeline` is tested
  within 1e-6.

## MVP

Implement training and inference in BAML, with a small scikit-learn-inspired API:

- `SimpleImputer` (mean strategy)
- `StandardScaler`
- `LinearRegression` (ordinary least squares)
- `LogisticRegression` (multiclass)
- `Pipeline`

Two built-in scikit-learn datasets: Iris for classification (`load_iris`, 150 samples, 4 features) and Diabetes for regression (`load_diabetes`, 442 samples, 10 features). Use `load_diabetes(scaled=False)` to exercise our own preprocessing without the dataset loader's default scaling.

## Experiment

| Configuration | Purpose |
| --- | --- |
| scikit-learn | Uninstrumented baseline |
| scikit-learn + OpenTelemetry | Explicitly instrumented baseline |
| BAML, telemetry off | Runtime baseline |
| BAML, telemetry on | Native observability |

Measure training time, inference latency (p50/p99), throughput, numerical parity, and telemetry overhead. Use identical inputs and train/test splits. Compare equivalent trace granularity and payloads; distinguish default profiling from detailed traces. Record runtime versions and hardware.

## Repository layout

- `baml_src/`: native ML implementations (BAML source)
- `baml_client/`: generated Python client (gitignored; produced by `npx baml --features beta generate`)
- `benchmarks/`: comparison harness
- `tests/`: unit and numerical parity tests (pytest; Python is the oracle/driver only, never a numerical backend for BAML functions)
- `examples/`: runnable examples

## Status

Issues #2, #3, #4 (StandardScaler), #5 (LinearRegression) and #7 (Pipeline) are implemented and tested: 76 pytest tests pass, including hand-computed fixtures, shape-mismatch and rank-deficiency rejection, `±Inf` and malformed-state rejection, all-missing-column rejection, held-out statistics reuse, and sklearn parity (SimpleImputer 1e-12, StandardScaler 1e-9, LinearRegression/Pipeline 1e-6).

Known limitations of BAML 0.226.3 beta expression functions (documented per issue #2 rather than worked around in Python):

- Statement-level expression functions require `--features beta`.
- No `throw`/exceptions and no runtime `@assert` in computed functions → the result-class error convention above.
- `float?`/`null` values cannot be inspected inside expression functions → NaN is used for missing values.
- No arithmetic mixing between `int` and `float`, and no int→float conversion.
- No float math methods (`sqrt`, `abs`, `pow`) and no math namespace in the beta surface — implemented in BAML instead (`Sqrt` via Newton's method); `exp`/`log` for later tickets (e.g. LogisticRegression) would need the same treatment (series expansions).
- No exponent float literals (`1e-10`, `1.0e308` are parse errors) — use decimal literals (`0.0000000001`).
- The beta compiler miscompiles if/else expressions nested inside branches that also contain statements; `baml_src` therefore uses flat statement-level validation plus a single tail if/else.

No LLM calls, tree models, model exporters, or dashboards are in scope for v0.1.
