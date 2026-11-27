# Adaptive and shadow experiment

The experiment packs `predict` and `shadow` using unmodified runtime sources,
BAML 0.26.0 and its matching native pack host. It uses only the Python 3.11+
standard library. The workload has 4,096 distinct four-feature rows and uses
the committed Iris multinomial model. A fresh working directory and home
isolate each process. The runner measures wall time, peak child RSS, BTEL
bytes and CAS bytes before any SQL indexing. One warmup per case precedes seven
measured runs with a fixed shuffled order.

```sh
export BAML_CLI=/absolute/path/to/baml-cli
export BAML_PACK_HOST=/absolute/path/to/baml-pack-host
python3 experiments/adaptive_shadow/run.py /tmp/adaptive-shadow-results
bash experiments/standalone/run.sh /tmp/standalone-results
```

A seven-run campaign on Apple M1 Pro is available with its raw samples and SQL results. [Measured report](results/m1-pro-2026-11-27/report.md)

The first command writes the report, toolchain and runtime hashes, raw samples
and SQL evidence. It asserts prediction parity and checks that zero, sparse
and capped anomalies replay exactly the expected indices. Dedicated native
queries verify the healthy and mixed plans, repeated calls on one loaded
handle, policy failures and an isolated challenger overflow.

The [Semantic Model Diff query](semantic-diff.sql) joins stage spans through
each model replay to its shared row span, then orders the pairs by row, stage
and role. The experiment checks exact input, imputed and scaled values plus
the final estimator details for linear, logistic and tree fixtures.
These small fixtures deliberately vary fitted parameters to make each expected
difference calculable independently. The storage benchmark uses the same
Iris artifact in both roles to keep the trigger rates fixed.

The second command verifies both packed subcommands and their BTEL output
in the existing offline standalone matrix. It runs with network denied
by the macOS sandbox or with Docker networking disabled on Linux.
