# Native observability discovery (BAML 0.26.0)

This isolated CLI experiment does not modify ML estimators or call an LLM.
Use the verified pinned CLI from the dedicated CI workflow:

```sh
export BAML_CLI=/absolute/path/to/baml-cli
python experiments/observability/run.py observability-results
```

The runner uses fresh project directories and homes, removes Boundary credentials,
disables CLI analytics with DO_NOT_TRACK, and uses synthetic data exclusively.
Default means BAML_TELEMETRY is unset; off means BAML_TELEMETRY=off.
Directives select span, timing or empty_span without instrumentation in function
bodies. The release-specific local SQL command is baml local recordings query.

Each case saves stdout/stderr, exit status, process wall time, discovered recording
files and sizes, schema, span count, grouped counts and sample rows. The wall time
includes process startup and is not an inference latency benchmark. Missing
recordings or failed queries are not interpreted as zero traces. Inspect the
actual schema and payloads before asserting profiling, hierarchy, redaction or
aggregation support. Results apply to CLI execution only; generated SDK capture
and the actual ML pipeline remain follow-up experiments.

References: https://boundaryml.com/blog/baml-0.26.0
