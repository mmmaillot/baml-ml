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

## Results

[Successful run 45934846303](https://github.com/mmmaillot/baml-ml/actions/runs/45934846303), 208 trials, four-trial medians. All checksums and capture checks passed.

| Features | Off µs/row | Default | Empty | Span |
|---:|---:|---:|---:|---:|
| 4 | 26.47 | 36.11 | 37.98 | 57.17 |
| 32 | 116.29 | 167.83 | 171.93 | 199.33 |
| 128 | 421.72 | 603.52 | 592.34 | 668.42 |
| 512 | 1617.98 | 2298.16 | 2361.24 | 2555.12 |

| Batch (32 features) | Off rows/s | Span rows/s |
|---:|---:|---:|
| 1 (100-call control) | 8469 | 4795 |
| 32 | 17919 | 11931 |
| 256 | 18573 | 12485 |
| 1024 | 18655 | 12413 |

| Training rows | Off ms | Default | Empty | Span |
|---:|---:|---:|---:|---:|
| 150 | 147.77 | 185.80 | 186.20 | 186.64 |
| 1000 | 552.63 | 692.09 | 686.03 | 694.88 |
| 10000 | 1196.37 | 1484.70 | 1483.51 | 1512.41 |

Iteration counts were not recorded; training complexity cannot be inferred.

At 32 features, distinct 1000-call span inference takes 330.04 ms vs
199.33 ms for repeated inputs. Between 100 and 1000 distinct calls,
BTEL grows by 333939 bytes, CAS by 6566280 bytes and SQL index by 1114112
bytes: approximately 371, 7296 and 1238 bytes per call respectively,
8905 bytes combined. Repeated inputs at matched counts grow by approximately
1609 bytes/call combined. Index is generated on query. Extrapolation to
one million distinct calls (~8.9 GB) is illustrative, not a production forecast.
Peak RSS for batch 1024 is 110200 KiB off and 260914 KiB span.

## Execution policy

Full scaling, performance and historical differential campaigns are manual
workflow dispatches. Native unit tests remain automatic. A future macOS
port should support Apple Silicon, verified CLI assets, BSD/GNU differences
(realpath, find, sed, time, timeout), RSS units, and capture OS/CPU/RAM,
commit, toolchain, power/thermal conditions and background load. Do not
compare timings across machines as a speedup. No macOS changes in this PR.
