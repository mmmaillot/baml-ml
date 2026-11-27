# Architecture decision: inference-first ML runtime

Status: accepted, 2026-11-27. Supersedes the training-first expansion of the original MVP and the acceleration priorities in #42. Existing implementations and evidence are retained.

## Context and evidence

Native BAML recordings reconstruct an Iris prediction, including imputation,
scaling, model parameters and probabilities (#36). Full semantic spans have
a material cost, particularly with distinct inputs (#38, #40): roughly 8.9 kB
per additional distinct 32-feature prediction including the SQL index, on the
measured workload. Default profiling is more compact. Removing redundant
validation in #44 reduced measured inference time by 15–35% on two matched
workloads without changing checksums. The training profiler found numerical
hotspots, but accelerating them would not validate the main product thesis.

## Decision

Build a classical ML **inference runtime with native semantic observability**,
not a replacement for the entire scikit-learn training ecosystem.

- Training is external by default: existing tools fit the model and export
  its parameters and preprocessing state. No requirement to trace individual
  training operations. Training metadata, metrics and provenance may accompany
  the artifact; they do not need to be recorded by the runtime.
- Keep existing BAML training implementations as reference/experimental
  controls. No new solvers or training-performance work by default.
- Inference, preprocessing, validation, error semantics and trace boundaries
  remain explicit in BAML. No LLM dependency.
- Use a versioned, portable, data-only artifact contract. First support
  mean imputation, standard scaling, linear regression and multinomial
  logistic regression, with feature order and class order, dimensions,
  numeric precision and provenance explicitly defined. Reject unsupported
  variants rather than silently approximate them. Do not execute pickle or
  arbitrary model code in the runtime.
- Prefer existing numerical libraries if acceleration becomes necessary.
  NumBam is optional, limited to measured inference bottlenecks with
  coarse-grained native kernels, parity and packaging evidence. No per-dot
  FFI or new generic numerical framework. No commitment yet to ndarray,
  faer, SmartCore or ONNX.
- Treat tracing as a configurable operational policy: default aggregate
  profiling, selective semantic spans for investigation, disabled mode for
  comparison. Production sampling, sensitive-data controls and retention
  are requirements to evaluate, not features already implemented.

## Consequences and sequence

1. Specify the artifact schema and loader, export a trained pipeline from
   an existing training tool, and run it using native BAML with no training
   environment at inference time. Validate malformed artifacts, missing
   values, model identity, feature/class ordering and reference outputs.
2. Establish the inference observation contract: trace inputs and outputs
   at semantic step boundaries, model identity, failures and durations;
   evaluate capture policies, privacy and storage. Distinguish debugging
   traces from statistical production monitoring.
3. Establish an embeddable or standalone inference interface and packaging.
   Keep the existing CLI as an experimental harness.
4. Profile representative inference workloads. Consider NumBam only if
   the measured benefit justifies the conversion and integration costs.

The first milestone is portable imported-model inference and reproducible
traces, not training interoperability across every estimator. Existing
full numerical and performance campaigns remain manual; lightweight
native regression checks stay automatic. Apple Silicon benchmark runner
remains deferred.
