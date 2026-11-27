# ADR: standalone executable first

Status: accepted, 2026-11-27. Refines the packaging decision in
[inference-first](inference-first.md).

## Context

The runtime loads validated scikit-learn artifacts, performs native BAML
inference and records semantic observations. Its initial target is batch and
edge execution, not a long-lived request-serving integration. Supporting
a language SDK creates host integration, generated types, ABI/FFI packaging,
lifetime and compatibility obligations. A model-serving server introduces
network, concurrency, scaling and authentication responsibilities.

## Decision

Ship a standalone executable via BAML 0.26.0's native bytecode packaging
(`baml pack`), using the version-matched upstream pack host. No custom CLI
parser, host wrapper, Python inference SDK, HTTP server or AOT Rust build
in the MVP. Python is only needed in the existing training/export workflow.

The executable takes a versioned fitted artifact and a JSON batch of ordered
feature names and rows, automatically dispatches by estimator kind, loads
and validates the model once, executes a batch and returns JSON. Use
standard stdout and shell redirection for output; stderr for diagnostics
and nonzero exit on failure. Native auto-CLI flags and subcommands are
derived from the BAML entry point. The contract is independent of supported
estimator count; adding an estimator may extend operations or output
shapes, not require a separate SDK.

The MVP keeps full batches in memory. JSON null represents missing numeric
data and is mapped to native NaN; malformed inputs are rejected. No CSV,
Parquet, Arrow, streaming, daemon, model registry or serving framework.

Observation is native, defaults to Semantic, with explicit Full and Timing.
Engine-wide BAML_TELEMETRY=off disables recording; no remote telemetry
credentials are embedded. Local records are an operational concern; analysis
can use a separate matching BAML CLI, not an inference dependency. Semantic
mode is not a guarantee of application-wide privacy (see observation ADR).

## Acceptance and consequences

1. Linux x86-64: build from a verified upstream toolchain, test the native
   executable on independent reference artifacts, offline in a clean
   environment with no BAML CLI, SDK, model training library or sources.
   Inspect actual packed-runtime local traces with the matching developer CLI.
   Produce a checksummed CI binary artifact, not an automatic public release.
2. Qualify macOS ARM64 and Linux ARM64 separately on native hardware/runners;
   availability of an upstream pack host alone is insufficient. Windows
   deferred. Record the OS/ABI requirements.
3. Manual batch/edge benchmarks: separate startup/file I/O from inference,
   throughput, memory, binary size and recording storage. Compare identical
   fitted parameters with sklearn. Only optimize measured bottlenecks.

The runtime embeds compiled BAML bytecode and the upstream host engine, not
necessarily a fully statically linked executable. System libraries/ABI may
still be required. Batch lifetime is one process per job, not one process per
row. No requirement to preserve closures across processes.

Reconsider SDKs/serving only if actual demand warrants their maintenance.
