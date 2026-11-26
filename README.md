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
.venv/bin/python examples/dot_example.py        # run one deterministic BAML function from Python
.venv/bin/python examples/iris_example.py       # issue #8: BAML ClassificationPipeline vs sklearn on Iris (clean + 10% NaN, seed 42)
.venv/bin/python examples/diabetes_example.py   # issue #8: BAML RegressionPipeline vs sklearn on Diabetes (clean + 10% NaN, seed 42)
.venv/bin/python benchmarks/sklearn_benchmark.py --out benchmarks/results/sklearn_benchmark.json  # issue #9: sklearn vanilla vs OpenTelemetry reference benchmark
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

- `baml_src/primitives.baml`: `IsNaN`, `IsFinite`, `Abs`, `Sqrt` (Newton's
  method), `VecNorm` (max-abs scaled), `CheckMatrix`, `CheckFiniteMatrix`,
  `Dot`, `ColumnMeans`, `ColumnVariances` (population, ddof=0), `VecAdd`,
  `VecSub`, `VecScale`, `MatVec`, `Transpose`. All arithmetic runs in BAML.
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
  mean and population variance, ddof=0; scale = `Sqrt(variance)`). Variances use
  Chan, Golub & LeVeque's (1983) corrected two-pass formula
  (`ColumnVariancesFromMeansUnchecked`), cancelling first-order mean rounding on
  constant columns (e.g. `0.1`, `log(1e-5)`, `1e100`) while keeping two-pass
  precision on large-offset features (`1e12 + N(0, 1)`). Near-constant threshold
  (issue #14): a column is treated as constant (`scale = 1.0`) when
  `var <= n * eps * var + (n * eps * mean)^2` (evaluated in scale space as
  `s * Sqrt(1 - n * eps) <= n * eps * |mean|` with `eps = 2^-52`), matching
  `sklearn`'s `_is_constant_feature` while preserving genuinely representable
  small scales. Input must be finite and already imputed; malformed scaler states
  (including non-positive or non-finite scales), dimension mismatches, and
  non-finite standardized outputs are rejected cleanly. Parity with
  `sklearn.preprocessing.StandardScaler` is tested within 1e-9, plus
  sklearn-inspired numerical hardening tests in `tests/test_numerical_hardening.py`.
- `baml_src/linreg.baml`: `LinearRegression` (ordinary least squares with
  intercept) — `LinearRegressionFit`, `LinearRegressionPredict` with an explicit
  `LinearRegressionModel` state (coefficients, intercept, feature count). Solved by
  QR decomposition of the design matrix `[1 | X]` via modified Gram-Schmidt plus
  back substitution; underdetermined (`n < p + 1`), rank-deficient
  (post-orthogonalization column norm <= 1e-10 times its original norm), and
  overflowing inputs, as well as malformed model states, are rejected explicitly.
  Parity with `sklearn.linear_model.LinearRegression` is tested within 1e-6 on
  well-conditioned fixtures.
- `baml_src/pipeline.baml`: minimal `Pipeline` (issue #7: Imputer → Scaler →
  LinearRegression or LogisticRegression) with explicit fitted states —
  `RegressionPipeline` (`RegressionPipelineFit`, `RegressionPipelinePreprocess`,
  `RegressionPipelinePredict`) and `ClassificationPipeline`
  (`ClassificationPipelineFit`, `ClassificationPipelinePreprocess`,
  `ClassificationPipelinePredict`, `ClassificationPipelinePredictProba`), plus
  the shared `PipelinePreprocessSteps` helper. Fitting is sequential on
  training data only; inference applies the stored states and never updates
  them, and downstream steps short-circuit when an upstream step fails.
  Parity with the equivalent sklearn `Pipeline` is tested within 1e-6
  (regression) and 1e-3 on probabilities with identical labels (classification
  on a missing-value Iris split).
- `baml_src/logistic.baml`: multiclass (multinomial) `LogisticRegression` with L2
  penalty (issue #6) — `LogRegFit`, `LogRegPredictProba`, `LogRegPredict` with an
  explicit `LogisticRegressionModel` state (sorted class order, per-class weights
  and intercepts, `n_iter`, `converged`). Objective: sum of NLL + ||W||²/(2C),
  intercepts unpenalized, optimized as the 1/n-scaled equivalent (same argmin)
  with C=1 by default usage; optimizer is gradient descent with Armijo
  backtracking line search (warm-started step), max_iter iterations, converged
  when the gradient inf-norm <= tol; non-convergence is reported as an error.
  Stable softmax (max subtraction) keeps rows finite and summing to 1 even for
  logits of ±1000. `Exp`/`Log` are implemented in BAML (`primitives.baml`):
  halving+squaring + Taylor for `Exp`, power-of-two scaling + atanh series for
  `Log`. Parity with a converged sklearn multinomial `LogisticRegression` on a
  deterministic standardized Iris split: probabilities within 1e-3 (measured
  ~2e-7) and identical labels; gradient checked against finite differences.

- `benchmarks/datasets.py` + `benchmarks/reference.py` (issue #8): shared deterministic fixtures — `load_iris` / `load_diabetes(scaled=False)`, fixed seed 42, 80/20 split (stratified for Iris only), optional fixed-seed 10% NaN variant — and the sklearn reference pipelines plus thin BAML wrappers used by both runtimes. Python only prepares data, runs reference models and evaluates; all BAML numerics stay in BAML.
- `examples/iris_example.py` and `examples/diabetes_example.py` (issue #8): end-to-end examples printing sklearn-reference and BAML metrics (accuracy/log loss/probability diffs on Iris; RMSE/R^2/prediction diffs on Diabetes) for the clean and the missing-value variant. Parity is asserted in `tests/test_examples.py` with explicit tolerances (proba 1e-3 and log loss 1e-6 for the solver-related GD-vs-lbfgs difference, measured ~1e-7; predictions/RMSE/R^2 1e-9 for OLS QR-vs-LAPACK, measured ~5e-13), plus a no-preprocessing-leakage check (fitted imputer/scaler stats equal the training-only numpy oracle).
- `benchmarks/sklearn_benchmark.py` (issue #9): Python reference benchmark runner, variants `sklearn` (uninstrumented) and `sklearn-otel` (OpenTelemetry spans around fit/predict and each named pipeline stage; in-memory exporter, `force_flush` outside timed regions, trace volume estimated from `Span.to_json()` bytes). Measures fit time, batch and single-row predict latency p50/p99, throughput, and trace volume per dataset with warm-ups, repetitions and a monotonic timer; dataset loading is excluded from timed regions; thread counts pinned to 1 and recorded. Writes a machine-readable JSON schema (schema_version 1: config, telemetry, environment, parity, results; non-applicable measurements are null) reusable by issue #10. Smoke-tested in `tests/test_benchmarks.py` (schema fields, timing units, vanilla emits zero spans, OTel emits named stage spans, instrumented/uninstrumented predictions match exactly).

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
- `benchmarks/`: comparison harness — shared dataset fixtures (`datasets.py`), sklearn reference pipelines + BAML wrappers (`reference.py`), and the sklearn vanilla/OpenTelemetry benchmark runner (`sklearn_benchmark.py`, writes JSON under the gitignored `benchmarks/results/`)
- `tests/`: unit and numerical parity tests (pytest; Python is the oracle/driver only, never a numerical backend for BAML functions)
- `examples/`: runnable examples (`dot_example.py`, and the issue #8 Iris/Diabetes parity examples)

## Status

Issues #2, #3, #4 (`StandardScaler`), #5 (`LinearRegression`), #6 (multiclass `LogisticRegression`), #7 (`Pipeline`: `RegressionPipeline` and `ClassificationPipeline`), #8 (Iris/Diabetes examples + end-to-end parity tests), #9 (sklearn vanilla/OpenTelemetry benchmark runner), and #14 (near-constant scaler threshold + sklearn-inspired numerical hardening tests) are implemented and tested: 175 pytest tests pass, including hand-computed fixtures, shape-mismatch, underdetermined and rank-deficiency rejection, `±Inf`/overflow and malformed-state rejection without VM crashes, all-missing-column rejection, held-out statistics reuse, constant/near-constant and large-offset column handling, extreme-logit softmax stability, gradient finite-difference checks, and sklearn parity (SimpleImputer 1e-12, StandardScaler 1e-9, LinearRegression/RegressionPipeline 1e-6, LogisticRegression and ClassificationPipeline proba 1e-3, end-to-end pipeline parity on Iris/Diabetes clean and 10%-NaN seed-42 splits: proba 1e-3, log loss 1e-6, predictions/RMSE/R^2 1e-9).

Known limitations of BAML 0.226.3 beta expression functions (documented per issue #2 rather than worked around in Python):

- Statement-level expression functions require `--features beta`.
- No `throw`/exceptions and no runtime `@assert` in computed functions → the result-class error convention above.
- Floating-point division by `0.0` or `-0.0` traps at runtime in the BAML VM (`DivisionByZero`) instead of producing IEEE-754 `±Inf`/`NaN` → divisors (including user-supplied model parameters) must be checked before `/`.
- `float?`/`null` values cannot be inspected inside expression functions → NaN is used for missing values.
- No arithmetic mixing between `int` and `float`, and no int→float conversion.
- No float math methods (`sqrt`, `abs`, `pow`) and no math namespace in the beta surface — implemented in BAML instead (`Abs`, `Sqrt` via Newton's method); `exp`/`log` for later tickets (e.g. LogisticRegression) would need the same treatment (series expansions).
- No exponent float literals (`1e-10`, `1.0e308` are parse errors) — use decimal literals (`0.0000000001`).
- No default parameter values and no `break`/`continue` — pass all arguments explicitly and drive loops with flags.
- `params` is a reserved keyword; a top-level `//` comment containing a `[...] =` pattern (e.g. `// logits[k] = ...`) breaks the beta parser — avoid that shape in top-level comments.
- The beta compiler miscompiles if/else expressions nested inside branches that also contain statements; `baml_src` therefore uses flat statement-level validation plus a single tail if/else.

No LLM calls, tree models, model exporters, or dashboards are in scope for v0.1.
