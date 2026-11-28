// Probes for the heterogeneous v3 pipeline BTEL experiment. This file is
// copied into a fresh probe project by run.sh alongside the runtime sources.
class HeteroProbeEntry {
  input TabularBatchInput
  predict_labels int[]?
  predict_proba float[][]?
}

function HeteroOracle() -> map<string, HeteroProbeEntry> {
  baml.json.decode<map<string, HeteroProbeEntry>>(baml.fs.read("artifacts/fixtures/v3/oracle.json"))
}

function HeteroInput() -> TabularBatchInput {
  HeteroOracle()["v3-ct-hetero-forest"].input
}

function HeteroModel() -> string {
  baml.fs.read("artifacts/fixtures/v3/v3-ct-hetero-forest.json")
}

function HeteroProbe(mode: ObservationMode) -> PipelineResult {
  let m = LoadPipelineModel("artifacts/fixtures/v3/v3-ct-hetero-forest.json", observation = mode);
  m.predict_proba(BatchCells(HeteroInput()), HeteroInput().features)
}

function HeteroFullProbe() -> PipelineResult { HeteroProbe(ObservationMode.Full) }
function HeteroSemanticProbe() -> PipelineResult { HeteroProbe(ObservationMode.Semantic) }
function HeteroTimingProbe() -> PipelineResult { HeteroProbe(ObservationMode.Timing) }
function HeteroAdaptiveProbe() -> PipelineResult { HeteroProbe(ObservationMode.Adaptive) }

function HeteroShadowProbe() -> ShadowOutput {
  RunShadow(HeteroModel(), HeteroModel(), baml.json.stringify(HeteroInput()), "predict-proba", DefaultAdaptivePolicy())
}

// Campaign probe on the large generated batch (batch.json, copied next to the
// project by campaign.sh).
function HeteroCampaignInput() -> TabularBatchInput {
  baml.json.decode<TabularBatchInput>(baml.fs.read("batch.json"))
}

function HeteroCampaignProbe(mode: ObservationMode) -> PipelineResult {
  let m = LoadPipelineModel("artifacts/fixtures/v3/v3-ct-hetero-forest.json", observation = mode);
  m.predict_proba(BatchCells(HeteroCampaignInput()), HeteroCampaignInput().features)
}

function HeteroCampaignSemanticProbe() -> PipelineResult { HeteroCampaignProbe(ObservationMode.Semantic) }
function HeteroCampaignAdaptiveProbe() -> PipelineResult { HeteroCampaignProbe(ObservationMode.Adaptive) }
function HeteroCampaignFullProbe() -> PipelineResult { HeteroCampaignProbe(ObservationMode.Full) }

function HeteroCampaignShadowProbe() -> ShadowOutput {
  RunShadow(HeteroModel(), HeteroModel(), baml.json.stringify(HeteroCampaignInput()), "predict-proba", DefaultAdaptivePolicy())
}
