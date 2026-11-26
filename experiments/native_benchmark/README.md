# Native BAML observability overhead (#38)

BAML 0.26.0 CLI only; no SDK, Python, external numerical kernels, LLM or
estimator modifications. Bash, jq and GNU time are experiment tools.
The pretrained Iris fixture from #36 is reused. Training uses the historical
120-row clean stratified split (seed 42, exported once with sklearn 1.8.0).
The four observation modes are off, default, empty_span and selective span.
For the latter two, only the four semantic fit and four prediction functions
receive native directives, in isolated copies of the sources.

## Workloads

| Workload | Warmups | Measured calls | Rows per call |
|---|---:|---:|---:|
| Zero-call control | 10 | 0 | 1 |
| Single prediction | 10 | 100 / 1000 / 5000 | 1 |
| Batch prediction | 10 | 100 | 30 |
| Fit | 1 | 3 | 120 |

Single inference repeats the synthetic missing-value row. Batch inference
alternates two fixed rows. Fit reuses the same training set. This intentionally
measures repeated inputs/parameters, which may benefit from CAS deduplication;
distinct production inputs and larger dimensions are separate experiments.
Four independent trials rotate mode order; each runs in a fresh native process.

## Measurements

- An in-language timer surrounds the measured loop, excluding warmup, fixture
  construction, compilation and process startup. It includes checksum work
  and synchronous recording work. Instant.now/elapsed use a non-monotonic wall
  clock (see standard library docs); reject nonpositive samples. These are
  aggregate per-trial timings, not individual inference p50/p99.
- GNU time captures complete CLI process wall time (including startup, warmup,
  serialization and normal exit/flush), CPU and peak RSS. Wall resolution is
  centiseconds; do not subtract unrelated clocks to claim exact flush time.
- File inventories before execution, before SQL query and after query separate
  BTEL, named CAS paths, index and other files. Query time does not contribute to
  the runtime timing. Storage includes warmups and static metadata; the zero
  control and slope between 100 and 5000 calls distinguish fixed and marginal
  costs. Categories are path-based; preserve inventories for verification.
- Query counts and profiler invocation totals verify the effective capture mode;
  all trials must return identical checksums for the same workload. This is
  a consistency check, not a replacement for #27/#36 numerical parity tests.

Run with verified pinned CLI:

```sh
export BAML_CLI=/absolute/path/to/baml-cli
bash experiments/native_benchmark/run.sh /tmp/native-benchmark
```

CI archives 96 raw trial records, file inventories, query results, aggregate
report and representative BTEL trees. There is no performance threshold on
a shared runner. The first run must validate actual clock/JSON compatibility.
No measured results yet, and no claim about native-versus-sklearn latency,
production throughput, privacy controls or whether Rust is necessary.
