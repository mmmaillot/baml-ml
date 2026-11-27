# Native scaling (#40)

Extends the native benchmark protocol in #38. Four modes, four rotated trials,
13 workloads (208 total). Feature sweep: 4/32/128/512 at batch 1, 1000 calls.
Batch sweep: 1/32/256/1024 at 32 features (batch 1 also has a matched 100-call
control). Distinct vs repeated inputs: 32 features, 100 and 1000 calls.
Training sweep: 150/1000/10000 rows, 4 features, three classes, one warmup
and one timed fit, max 100 iterations and tolerance 1e-6. Synthetic data and
fixed model parameters, not a model-quality benchmark. No requirement that
the bounded optimizer converges; report convergence separately if needed.

All input batches are constructed before the timer. Distinct inputs have
unique first features; repeated inputs reuse a single batch. This isolates
inference time but peak RSS includes prebuilt input storage. Checksums must
match across modes and trials for each workload. The distinct-input storage
slope compares 100 and 1000 calls; compare with repeated workloads at matching
counts. SQL indexing is measured separately from recording. A 180-second
process timeout fails the trial and preserves partial evidence; do not treat
timeouts as measured latencies. Shared runner, aggregate loop timings, no
performance gate. See #38 for timer and storage caveats.

Run: `BAML_CLI=/absolute/path/to/baml-cli bash experiments/scaling/run.sh /tmp/scaling`.

Results pending CI; first run must validate the synthetic fixture and syntax.
