# Heterogeneous v3 pipelines + RandomForest (#76)

End-to-end evidence for `schema_version = 3` heterogeneous
`ColumnTransformer(SimpleImputer+StandardScaler, SimpleImputer+OneHotEncoder) -> RandomForestClassifier`
pipelines: BTEL SQL trace verification, standalone packed-binary smoke and a
reproducible storage/latency campaign.

## BTEL trace verification

```sh
bash experiments/heterogeneous_pipelines/run.sh /tmp/hetero-results
```

Runs unmodified runtime sources plus `probes.baml` in fresh isolated projects
and asserts, per probe (`full`, `semantic`, `timing`, `adaptive`, `shadow`,
`off`):

- identical predictions across modes, matching the committed sklearn 1.8.0
  oracle (`artifacts/fixtures/v3/oracle.json`);
- Full: exactly one `ObservedColumnTransformer` span tagged
  `baml_ml.step_index=0` / `baml_ml.step_kind=column_transformer`, two
  `ObservedColumnTransformerBranch` children carrying `baml_ml.branch_name`,
  step spans (`ObservedImputeStep`, `ObservedScaleStep`, `ObservedOneHotEncode`)
  tagged with step index/kind, one `ObservedForestTraversal`, one
  `PipelineModelLoaded`; all step payloads captured but fitted state redacted
  (`$redacted`, no `statistics`/`categories`/`trees`/`nodes`/`means`/`scales`);
- Semantic: same span inventory with null payloads; Timing/off: no semantic
  spans;
- Adaptive: `AdaptiveSummary` plus two `EscalatedRow` Full replays
  (`baml_ml.row_index`, `baml_ml.escalated`, `baml_ml.trigger_reasons=missing_input`)
  on the null numeric/categorical cells; the lazy `PipelineModelLoaded` snapshot
  fires exactly once after the semantic pass;
- Shadow: champion/challenger roles on every stage span, selective
  `ShadowEscalatedRow` Full replays, `missing_input=2`, predictions from the
  champion matching the oracle.

## Standalone smoke

`experiments/standalone/run.sh` packs the offline binary and exercises the
committed `v3-ct-hetero-forest` fixture in `predict` (full), `predict
--observation adaptive` and `shadow`, asserting oracle parity, the
step/branch span hierarchy and escalation metadata on Linux x86-64, Linux
ARM64 and macOS ARM64.

## Storage/latency campaign

```sh
bash experiments/heterogeneous_pipelines/campaign.sh /tmp/campaign
```

Runs a deterministic 500-row heterogeneous batch (5% null cells,
`generate_batch.py`) through the packed runtime probes in Semantic, Adaptive,
Full and shadow, best of 3 repetitions, measuring wall time, local recording
size and span count. The measured report committed under `evidence/` covers
the exact runtime sources by SHA-256; no performance claim is made beyond
those measured numbers on the recorded platform.
