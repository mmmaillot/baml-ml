# Native ML observability (#36)

Run a pretrained Iris pipeline through a single BAML CLI invocation, with no
Python SDK, Rust SDK, LLM or network inference. Bash and jq are external
experiment harnesses only. The original ML algorithm sources are copied into
an isolated project, and the only changes are optional native trace directives
on four semantic functions. No production source changes.

```sh
export BAML_CLI=/absolute/path/to/verified/baml-cli
bash experiments/native_ml_observability/run.sh /tmp/native-ml-observability
```

The fixture contains fitted parameters from an independent scikit-learn 1.8.0
reference, trained on the same stratified clean Iris split (seed 42, 120 training
rows). The two inputs are synthetic variants of one Iris row. The committed
oracle records the expected imputed and standardized values and predicted
probabilities. The native runner requires no Python installation or fitting.
Prediction equality across four observation modes is checked, alongside
independent probability parity to 1e-10.

## Modes and evidence

Eight isolated runs: off, default, empty_span and selective span, each with a
complete or missing feature. For each, save exact output, recording schema,
span/parent IDs, timing and payloads, profiler rows and all file sizes under
project .baml and the isolated home, before and after query indexing. Named
storage categories are provisional; retain the complete inventory to avoid
confusing BTEL bytes with total storage.

Three further native runs capture scalar arguments/results, a matrix of
10,000 distinct values, and an intentional out-of-bounds failure. Raw matrix
values are archived rather than dumped in logs. Querying the payload fields
is necessary but not sufficient to demonstrate their contents; inspect actual
rows and assert the transformation values and error payloads before closing
the issue. A JSON null does not on its own establish a recorded NaN.

No timing overhead claims: profiler durations and CLI process latency are
not a controlled performance benchmark. Native benchmarking and Rust
acceleration remain separate work.
