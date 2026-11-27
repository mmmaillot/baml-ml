# Standalone batch and edge benchmark

Manual baseline for [#57](https://github.com/mmmaillot/baml-ml/issues/57) on native
macOS ARM64. The [first MacBook Pro run](m1-pro-2026-11-27.md) records the results.
The benchmark uses the Iris multiclass `predict-proba` operation with batches of
1, 128 and 4,096 rows.

## Run

Use a native ARM64 Python 3.12 environment with scikit-learn 1.8.0. Set
`BAML_CLI` and `BAML_PACK_HOST` to the binaries extracted from the verified
BAML 0.26.0 ARM64 archive. The [standalone workflow](../../.github/workflows/standalone-package.yml)
has the archive name and SHA-256 verification command.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install scikit-learn==1.8.0
export BAML_CLI=/absolute/path/to/baml-cli
export BAML_PACK_HOST=/absolute/path/to/baml-pack-host
caffeinate -i .venv/bin/python experiments/standalone_benchmark/run.py /tmp/baml-bench
```

Choose a new output directory for each campaign. The default is 15 samples
per combination. Use `--samples 1` for a quick run. Run on AC power with the
machine otherwise idle for comparisons.

The output contains `metadata.json` with versions, checksums and hardware,
`samples.jsonl` with individual measurements, `summary.json` with medians and
wall time ranges, and `errors.json` with exit codes and stderr. The inputs,
fitted model in both formats, and both native binaries are kept for replay.
The Python worker loads the joblib file generated locally by this benchmark.

## Measurement

- **Workload.** Fit one scikit-learn pipeline (mean imputer, standard scaler and
  multinomial logistic regression) on a stratified split with seed 42. Export
  the same fitted instance to BAML JSON and joblib. Cycle the held out rows,
  with one JSON null every eight rows. Both jobs consume the same input file.
  Each measured invocation checks class order and metadata when available,
  and compares every probability against the reference within an absolute
  tolerance of 1e-9.
- **Process per batch.** Launch the shipped `predict` entry point and an
  equivalent Python job with the appropriate model file. Time from launch to
  exit, with JSON stdout redirected to a regular file. This includes file
  reads, parsing, model preparation, inference, output and runtime shutdown.
  Take peak RSS from Darwin `wait4`. The throughput divides batch size by the
  median wall time.
- **Launch and loaded model.** Time `--help` separately as a launch reference.
  This exercises each program's startup and argument parser. The second BAML
  binary loads the model once and reuses the same runtime prediction method
  with telemetry off. Both loaded jobs measure file reads, model preparation
  and input decoding separately. After three warmup calls, time a loop of
  500, 100 or 5 requests depending on batch size. Divide loop time by the
  call count. This includes input validation and preprocessing, and ends
  before host JSON output. BAML 0.26.0 uses a wall clock for these phases.
  Python uses a monotonic clock. Report the phase times independently of the
  standalone wall time.
- **Modes and storage.** The standalone job uses engine off, timing, semantic
  (the CLI default) and full. Each subprocess has a fresh home and uses
  the same observation level where applicable. Count logical bytes by BTEL,
  CAS and other files before any query builds an index. Discard that home after
  counting. Output bytes are tracked separately in the raw samples.
- **Repetition and errors.** Run jobs sequentially, shuffle the order in each
  round with a fixed seed, and discard the first round for each measurement
  group. Set numerical libraries to one thread and keep the filesystem cache
  warm through repeated reads. Separately exercise the seven standalone
  rejection cases with telemetry off and capture stderr and the exit code.

These measurements characterize one small model and this workload. The
process comparison uses the native binary's JSON artifact and Python's joblib
model format. The loaded model comparison isolates model execution after
each runtime has loaded the artifact and decoded the inputs.
