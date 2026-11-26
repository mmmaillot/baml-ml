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
values are archived rather than dumped in logs. The harness asserts the actual transformation values, explicit typed NaN, parent span IDs, empty-span payload absence, scalar values, every distinct matrix element, and structured panic payload. A JSON null does not on its own establish a recorded NaN.

No timing overhead claims: profiler durations and CLI process latency are
not a controlled performance benchmark. Native benchmarking and Rust
acceleration remain separate work.

## First observed results (BAML 0.26.0)

[Successful run 45901384969](https://github.com/mmmaillot/baml-ml/actions/runs/45901384969/job/140187722564).

Missing feature: sepal width, typed as {"$float":"NaN"}. Selective spans show
imputed row [5.1, 3.0483333333333342, 1.4, 0.2], scaled row
[-0.885662023181763, 0, -1.3457223087882542, -1.3232755820436295],
and probabilities [0.9262866982922007, 0.07371189601705161, 0.0000014056907476408886].
The four selected spans have parent IDs linking them to the pipeline/root.
Default mode records three spans but profiles all 32 calling-context nodes
(including runtime helpers); empty spans omit internal payloads. Off yields no
spans or profiler rows. The panic payload includes the actual index, array length,
function names and source lines. Full typed model parameters are visible in
selective spans; this is also a data confidentiality consideration.

| Missing-row mode | BTEL bytes | CAS-path bytes | Query index bytes |
|---|---:|---:|---:|
| Off | 0 | 0 | 0 |
| Default | 11314 | 48169 | 253952 |
| Empty spans | 11505 | 48301 | 253952 |
| Selective spans | 11666 | 52645 | 253952 |

These are per-isolated-process logical storage sizes, not per-prediction marginal
cost. Other files (~7.28 MB) are reported separately and exist even with tracing
off. CAS includes static/source data, not just numerical inputs. The distinct
100x100 matrix run recorded 2683 BTEL bytes and 94948 CAS-path bytes; the trace
query returned numerical input and output arrays. The additional full-matrix
assertion checks every element on subsequent CI runs. No extrapolation to
production throughput or retention cost is warranted.
