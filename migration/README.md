# BAML language migration (phase 1)

Isolated port to BAML language 0.25.0. The historical project and NumBam are unchanged. Baseline: [v0.1.0](https://github.com/mmmaillot/baml-ml/tree/v0.1.0).

Use the exact Linux CLI asset and SHA-256 from [migration CI](../.github/workflows/baml-migration.yml), extracted from the [official release](https://github.com/BoundaryML/baml/releases/tag/baml-language-0.25.0). The version manifest returned 404 during the interop POC, so CI verifies the release archive directly. With `BAML_CLI` set to the extracted executable's absolute path:

```bash
cd migration
"$BAML_CLI" check
"$BAML_CLI" test --list
"$BAML_CLI" test --max-concurrency 1
```

## Primitive compatibility

- Replace handwritten Exp, Log, Sqrt, Abs, IsNaN and IsFinite with [native float methods](https://github.com/BoundaryML/baml/blob/baml-language-0.25.0/baml_language/crates/baml_builtins2/baml_std/baml/float.baml).
- New language float equality is reflexive even for NaN, and comparisons have a total order. Use the predicate wrappers before ordering; never detect NaN with `x != x`.
- Retain the old finite negative Log sentinel for nonpositive values and Sqrt clamp to positive zero. NaN propagates.
- Native Exp intentionally improves subnormal behavior: e.g. exp(-720) is positive, whereas the old reciprocal implementation underflowed early. Platform math may differ in last bits.
- Retain matrix shape checks, result classes, ordered dot, scaled norm and corrected two-pass variance; no NumBam calls or solver changes.
- Port all 18 original primitive test cases to native BAML tests, plus Exp and Log checks against f64 reference values. No full estimator migration or SDK parity claim in this ticket.

Primitive CI: [20/20 passed](https://github.com/mmmaillot/baml-ml/actions/runs/45875548493).


## Estimators and examples (#26)

The four estimators and both pipeline variants are ported to the same project.
The original modified Gram-Schmidt QR and multinomial gradient descent with
Armijo line search remain unchanged. No NumBam calls or Python numerical callbacks.

Prepare clean and 10%-missing Iris and Diabetes examples using the same
historical dataset loader, seed 42, train/test split, and reference model
hyperparameters. Python generates a temporary BAML fixture file, containing
inputs and independent sklearn predictions / train-only means. Native BAML
functions execute fitting, inference and parity checks. No Python SDK is
needed for these examples.

```bash
# From the repository root, using Python 3.12:
python -m pip install -r migration/examples/requirements.txt
python migration/examples/generate_examples.py
cd migration
"$BAML_CLI" check
"$BAML_CLI" test --max-concurrency 1 --test-timeout 5m
"$BAML_CLI" run ExampleIrisClean
"$BAML_CLI" run ExampleIrisMissing
"$BAML_CLI" run ExampleDiabetesClean
"$BAML_CLI" run ExampleDiabetesMissing
```

The example results expose maximum absolute prediction and preprocessing mean
differences, and classification label equality. Tolerances are unchanged:
Iris 1e-3, Diabetes 1e-9, train-only means 1e-12. Native tests reject non-finite
predictions, wrong output shapes and error results. Full historical test
suite/SDK migration is #27; performance and telemetry comparisons are #28.


## Historical Python regression suite (#27)

Generate the new Python SDK from the same pinned CLI. Set
`BAML_BRIDGE_PATH` to the verified matching engine library and
`BAML_BRIDGE_DISABLE_DOWNLOAD=true` (see CI for exact checksums).
From the repository root:

```bash
python -m pip install -r migration/host_tests/requirements.txt
python migration/examples/generate_examples.py
(cd migration && "$BAML_CLI" generate --from "$PWD")
python migration/host_tests/prepare.py
export PYTHONPATH="$PWD/migration/gen:$PWD/migration/host_tests/generated"
(cd migration/host_tests/generated && python -m pytest tests)
```

The test mirror preserves the original numerical assertions and reference
functions; only the legacy client import and generation bootstrap change.
The new SDK exposes functions as `Name_sync` and generated classes directly.
The name adapter does not implement any numerical logic. Both SDKs must not
be loaded in one process. Python tests exercise host marshalling, NaN and
infinity, malformed fitted states, random differential checks, gradients,
convergence, end-to-end predictions and train-only preprocessing, alongside
the native BAML tests. The sklearn-only benchmark tests are also retained.

Explicit exclusions: old npm-client freshness/layout tests are specific to
the historical toolchain and are replaced by new SDK smoke tests. The old
Collector-based BAML benchmark and trace demo/report tests
(`test_baml_benchmark.py`, `test_report_and_trace_demo.py`) remain available
on the historical baseline; porting their runtime-specific behavior belongs
to #28 and the observability phase. No old assertions are weakened or deleted.
The old test suite and NumBam workflows are untouched.


## Runtime benchmark baseline (#28)

The new SDK runner reuses the historical staged fit/predict functions and seed-42
Iris/Diabetes splits. It records wall time for the trivial bridge probe, staged
fit, full test batch and each single-row prediction; per-call samples, p50/p99,
throughput and parity are saved in JSON. No runtime-only timing is inferred.
New trace metrics remain null until a supported capture path is demonstrated.

After the SDK and historical test mirror are generated as above:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BAML_TELEMETRY=off
python migration/benchmarks/run.py --out migration-benchmark-results/new-baml.json
```

On the same machine, in a separate checkout and Python environment:

```bash
git worktree add --detach /tmp/baml-mvp v0.1.0
cd /tmp/baml-mvp
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
npm ci
npx baml --features beta generate
.venv/bin/python benchmarks/baml_benchmark.py --repetitions 10 --fit-repetitions 3 --warmups 1 --out /absolute/path/migration-benchmark-results/legacy-baml.json
.venv/bin/python benchmarks/sklearn_benchmark.py --repetitions 10 --warmups 1 --out /absolute/path/migration-benchmark-results/sklearn.json
```

Back in the migration checkout:

```bash
python migration/benchmarks/report.py migration-benchmark-results
```

CI runs this protocol and uploads the raw JSON and report. Fit uses the historical
3 repetitions for both BAML versions and 10 for sklearn; all predict/probe
measurements use 10 repetitions and one warmup, except single-row inference
which is warmed by the batch. The benchmark is not an optimization target.
The old Collector is not a verified API of the new SDK. BAML_TELEMETRY=off
does not by itself prove runtime tracing is disabled. A native trace capture
probe, on/off overhead and Rust SDK comparison remain to be established.
