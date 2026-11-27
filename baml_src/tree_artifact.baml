// Version 2 is deliberately tree-only; the existing v1 schema remains unchanged.
class ArtifactHeader {
  schema_version int
}
class FittedTreeArtifact {
  schema_version int
  source string
  source_version string
  model_id string
  features string[]
  imputer_kind string
  scaler_kind string
  estimator_kind string
  imputer_means float[]
  scaler_means float[]
  scaler_scales float[]
  classes int[]
  nodes TreeNode[]
}
class PreparedTreePipeline {
  imputer MeanImputer
  scaler StandardScaler
  tree TreeClassifierState
}
class LoadedTreeModel {
  proba_fn: (float[][], string[]) -> ProbaResult throws never
  labels_fn: (float[][], string[]) -> IntArrayResult throws never
  function predict_proba(self, x: float[][], features: string[]) -> ProbaResult {
    let f = self.proba_fn;
    f(x,features)
  }
  function predict(self, x: float[][], features: string[]) -> IntArrayResult {
    let f = self.labels_fn;
    f(x,features)
  }
}
function ReadArtifactVersion(raw: string) -> int {
  let h = baml.json.decode<ArtifactHeader>(raw);
  h.schema_version
}
function DecodeTreeArtifact(raw: string) -> FittedTreeArtifact {
  baml.json.decode<FittedTreeArtifact>(raw)
}
function TreeArtifactError(a: FittedTreeArtifact) -> string {
  let n = a.features.length();
  if (a.schema_version != 2 || a.estimator_kind != "decision_tree_classifier") { return "artifact: unsupported tree schema or estimator"; }
  if (a.source != "scikit-learn" || a.source_version == "" || a.model_id == "") { return "artifact: invalid provenance"; }
  if (a.imputer_kind != "mean" || a.scaler_kind != "standard") { return "artifact: unsupported preprocessing"; }
  if (n == 0 || a.imputer_means.length() != n || a.scaler_means.length() != n || a.scaler_scales.length() != n) { return "artifact: invalid preprocessing dimensions"; }
  for (let j = 0; j < n; j += 1) {
    if (a.features[j] == "") { return "artifact: empty feature name"; }
    for (let k = 0; k < j; k += 1) {
      if (a.features[j] == a.features[k]) { return "artifact: duplicate feature name"; }
    }
    if (!IsFinite(a.imputer_means[j]) || !IsFinite(a.scaler_means[j]) || !IsFinite(a.scaler_scales[j]) || a.scaler_scales[j] <= 0.0) { return "artifact: invalid preprocessing parameters"; }
  }
  TreeStateError(TreeClassifierState { n_features: n, classes: a.classes, nodes: a.nodes })
}
function PrepareTreeArtifact(raw: string) -> FittedTreeArtifact {
  let a = DecodeTreeArtifact(raw);
  let err = TreeArtifactError(a);
  if (err != "") { throw baml.errors.InvalidArgument { message: err }; }
  a
}
function TreeModelLoaded(artifact: FittedTreeArtifact) -> void {}

// A single semantic batch span, not one full span for every visited node.
// Full mode exposes ordered node IDs and leaf IDs; together with the load
// record and scaled inputs these reconstruct split features and thresholds.
function ObservedTreeTraversal(x: float[][], state: TracePrivate<TreeClassifierState>, capture_path: bool) -> TreeBatchResult throws InferenceFailure {
  TreePredictCore(x,state.value,capture_path, $trace = trace.timing())
}
function PredictLoadedTree(x: float[][], features: string[], model_id: string, expected: string[], state: TracePrivate<PreparedTreePipeline>, observation: ObservationMode) -> TreeBatchResult throws InferenceFailure {
  let err = LoadedFeatureError(features,expected);
  if (err != "") { throw InferenceFailure { stage: "input_validation", message: err }; }
  let p = state.value;
  let opts = ObservationOptions(observation);
  let t = PreparedPreprocess(x,p.imputer,p.scaler,opts);
  ObservedTreeTraversal(t,TracePrivate<TreeClassifierState> { value: p.tree },observation == ObservationMode.Full, $trace = opts)
}
function PreparedTree(a: FittedTreeArtifact) -> PreparedTreePipeline {
  let n = a.features.length();
  PreparedTreePipeline {
    imputer: MeanImputer { means: a.imputer_means, n_features: n },
    scaler: StandardScaler { means: a.scaler_means, scales: a.scaler_scales, n_features: n },
    tree: TreeClassifierState { n_features: n, classes: a.classes, nodes: a.nodes }
  }
}
function BuildLoadedTree(a: FittedTreeArtifact, observation: ObservationMode, policy: AdaptivePolicy? = null) -> LoadedTreeModel {
  let adaptive = ResolveAdaptivePolicy(observation, policy);
  let names = a.features;
  let id = a.model_id;
  let model_opts = ModelObservation(observation,id,a.schema_version,a.source_version);
  TreeModelLoaded(a, $trace = model_opts);
  let p = PreparedTree(a);
  if (adaptive != null) {
    let classifier = AdaptiveClassifier(SelectiveTree(a, p), adaptive);
    return LoadedTreeModel { proba_fn: classifier.proba_fn, labels_fn: classifier.labels_fn };
  }
  let state = TracePrivate<PreparedTreePipeline> { value: p };
  LoadedTreeModel {
    proba_fn: (x: float[][], features: string[]) -> ProbaResult {
      TreeResultProba(PredictLoadedTree(x,features,id,names,state,observation, $trace = model_opts)) catch (e) {
        let f: InferenceFailure => ProbaResult { ok: false, probs: [], error: f.message }
      }
    },
    labels_fn: (x: float[][], features: string[]) -> IntArrayResult {
      TreeResultLabels(PredictLoadedTree(x,features,id,names,state,observation, $trace = model_opts)) catch (e) {
        let f: InferenceFailure => IntArrayResult { ok: false, value: [], error: f.message }
      }
    }
  }
}
function PrepareTreeModel(raw: string, observation: ObservationMode = ObservationMode.Semantic, policy: AdaptivePolicy? = null) -> LoadedTreeModel {
  let opts = ObservationOptions(observation);
  let a = PrepareTreeArtifact(raw, $trace = if (observation == ObservationMode.Full) { trace.span(inputs = false, output = false) } else { opts });
  BuildLoadedTree(a,observation, policy = policy)
}
function LoadTreeModel(path: string, observation: ObservationMode = ObservationMode.Semantic, policy: AdaptivePolicy? = null) -> LoadedTreeModel {
  PrepareTreeModel(baml.fs.read(path), observation = observation, policy = policy)
}
function RunTreeBatch(raw_model: string, raw_input: string, operation: string, mode: ObservationMode, policy: AdaptivePolicy? = null) -> BatchOutput {
  let opts = ObservationOptions(mode);
  let a = PrepareTreeArtifact(raw_model, $trace = if (mode == ObservationMode.Full) { trace.span(inputs = false, output = false) } else { opts });
  let input = DecodeBatchInput(raw_input);
  let rows = BatchRows(input);
  let m = BuildLoadedTree(a,mode, policy = policy);
  if (operation == "predict-proba") {
    let r = m.predict_proba(rows,input.features);
    if (!r.ok) { throw baml.errors.InvalidArgument { message: r.error }; }
    return BatchOutput { schema_version: 1, model_id: a.model_id, estimator_kind: a.estimator_kind, operation: operation, classes: a.classes, predictions: r.probs };
  }
  let r = m.predict(rows,input.features);
  if (!r.ok) { throw baml.errors.InvalidArgument { message: r.error }; }
  BatchOutput { schema_version: 1, model_id: a.model_id, estimator_kind: a.estimator_kind, operation: operation, classes: a.classes, predictions: r.value }
}

function TreeResultProba(r: TreeBatchResult) -> ProbaResult {
  ProbaResult { ok: true, probs: r.probs, error: "" }
}
function TreeResultLabels(r: TreeBatchResult) -> IntArrayResult {
  IntArrayResult { ok: true, value: r.labels, error: "" }
}
