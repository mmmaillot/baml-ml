# Trusted internal validation (#44)

The inference pipeline previously scanned the imputed matrix for finite
values before scaling, then scanned the scaled matrix again before
classification. A successful ImputerTransform already guarantees finite
values, and a successful ScalerTransform checks every computed value.
Only these two cell-level rescans are removed.

Public ScalerTransform, LogRegPredictProba and LogRegPredict signatures
and full input checks are preserved. Internal core functions receive a
finite_input flag only from successful preceding steps; the flag is an
internal precondition, not a way for application callers to bypass checks.
Shape, feature counts, fitted-state parameters, overflow of standardized
values and non-finite probability checks remain unchanged. No fitted model
is assumed immutable or trusted across calls. Training is unchanged.
Regression preprocessing shares the imputer-to-scaler path, but the
linear-regression prediction validation is unchanged.

Semantic spans for scaler and logistic probability computation now target
ScalerTransformCore and LogRegPredictProbaCore, respectively, including
the original x/scaler or x/model payload and the extra finite_input flag.
The observability and benchmark harnesses follow the computation rather
than the public wrappers. Historical reports retain their original names.

Native regression tests exercise public rejection of NaN/Inf, malformed
inputs and states, error precedence, valid imputed predictions, class labels
and arithmetic overflow. The manual historical campaign is preserved.

## Focused comparison

The script compares the exact pre-change commit
40d58ba5de4e68a5a23182f6798f962339aa232e with the candidate on the same
runner, using identical 512-feature single-row and 32-feature batch-256
workloads from #40. Three trials in each variant and mode (off/default),
24 cases, with matched warmups and alternating variant order. Checksums
must match exactly; default profiler must show exactly two fewer full
finite-matrix scans per prediction, including warmups. Timing is descriptive,
not a performance gate. Full benchmark campaigns remain manual.

```sh
BAML_CLI=/absolute/path/to/baml-cli bash experiments/validation/run.sh /tmp/validation
```

Results pending validation. No Rust code or new estimator.
