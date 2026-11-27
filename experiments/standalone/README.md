# Standalone batch executable (Linux x86-64)

Linux x86-64 validated in [CI run 45958260690](https://github.com/mmmaillot/baml-ml/actions/runs/45958260690): all five reference operations, four observation modes and seven invalid-input cases. See the [standalone-first ADR](../../docs/architecture/standalone-first.md).

## Build

Use the matching BAML CLI and pack host from the verified 0.26.0 release archive.
The CI script builds without test files or fixture data and runs the binary
inside an offline Ubuntu 24.04 container. A matching CLI is used outside the
container only to inspect the generated recordings.

```sh
export BAML_CLI=/absolute/path/to/baml-cli
export BAML_PACK_HOST=/absolute/path/to/baml-pack-host
bash experiments/standalone/run.sh /tmp/standalone-results
```

The CI artifact contains the binary and SHA-256 checksum. This is not a
public release. The OS/ABI dependencies are recorded by file and ldd.
Docker is a validation harness, not a deployment dependency.

## Usage

```sh
./baml-ml predict --model model.json --input data.json --output-format json > predictions.json
./baml-ml predict --model model.json --input data.json --operation predict-proba --observation full --output-format json
BAML_TELEMETRY=off ./baml-ml predict --model model.json --input data.json --output-format json
```

Input is a single JSON object:
```json
{"features":["f1","f2"],"rows":[[1.2,null],[3.4,5.6]]}
```
Names must exactly match artifact order. Rows must be nonempty and rectangular.
Numbers and null only; null means missing (NaN). Whole batch is in memory.
No arbitrary JSON strings, CSV, streaming or per-row process launch.

Output schema v1 includes schema_version, model_id, estimator_kind, operation,
classes and predictions. Predict returns floats for linear regression and
integer labels for logistic regression. Predict-proba is classification-only
and returns one row per sample, columns in classes order. Linear classes is [].
A failed job exits nonzero; stdout is only a usable result on exit zero.
Use shell redirection rather than a second file-writing API. No output file
atomicity guarantee from shell redirection; use a temporary file if needed.

Observation defaults to semantic, full and timing are explicit options.
Native global off is BAML_TELEMETRY=off. Packed runtime writes its local
recordings under the user home rather than the project source. No remote
credentials are embedded; avoid supplying Boundary credentials for local-only
use. Native root traces and caller/engine policy are distinct from semantic
span payload policies; see the [contract](../../docs/architecture/observation-contract.md).

macOS ARM64 and Linux ARM64 are not qualified yet. No inference SDK or serving
framework, and no benchmark claim.
