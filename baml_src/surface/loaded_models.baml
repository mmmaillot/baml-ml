// Factory-owned public model handles (loaded_models) and their builders.
// Factory-owned state. A BAML class/array is mutable by reference; do not
// expose an artifact, parameter arrays or the pipeline captured by these closures.
// Accept JSON strings, not caller-owned mutable objects. Each decode owns its state.
// The callable fields themselves can be replaced by the caller, but this does not
// expose or change the factory's captured parameters.


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

function BuildLoadedTree(a: FittedTreeArtifact, observation: ObservationMode, adaptive: AdaptivePolicy?) -> LoadedTreeModel {
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
  let adaptive = ResolveAdaptivePolicy(observation, policy);
  let opts = ObservationOptions(observation);
  let a = PrepareTreeArtifact(raw, $trace = if (observation == ObservationMode.Full) { trace.span(inputs = false, output = false) } else { opts });
  BuildLoadedTree(a,observation,adaptive)
}

function LoadTreeModel(path: string, observation: ObservationMode = ObservationMode.Semantic, policy: AdaptivePolicy? = null) -> LoadedTreeModel {
  PrepareTreeModel(baml.fs.read(path), observation = observation, policy = policy)
}

function TreeResultProba(r: TreeBatchResult) -> ProbaResult {
  ProbaResult { ok: true, probs: r.probs, error: "" }
}

function TreeResultLabels(r: TreeBatchResult) -> IntArrayResult {
  IntArrayResult { ok: true, value: r.labels, error: "" }
}

class LoadedLinearModel {
  predict_fn: (float[][], string[]) -> NumArrayResult throws never
  function predict(self, x: float[][], features: string[]) -> NumArrayResult {
    let f = self.predict_fn;
    f(x, features)
  }
}


class LoadedLogisticModel {
  proba_fn: (float[][], string[]) -> ProbaResult throws never
  labels_fn: (float[][], string[]) -> IntArrayResult throws never
  function predict_proba(self, x: float[][], features: string[]) -> ProbaResult {
    let f = self.proba_fn;
    f(x, features)
  }
  function predict(self, x: float[][], features: string[]) -> IntArrayResult {
    let f = self.labels_fn;
    f(x, features)
  }
}


function PrepareLinearModel(raw: string, observation: ObservationMode = ObservationMode.Semantic, policy: AdaptivePolicy? = null) -> LoadedLinearModel {
  let adaptive = ResolveAdaptivePolicy(observation, policy);
  // Do not record raw artifact text or repeat the decoded output.
  let opts = ObservationOptions(observation);
  let a = PrepareArtifact(raw, true, $trace = if (observation == ObservationMode.Full) { trace.span(inputs = false, output = false) } else { opts });
  BuildLoadedLinear(a, observation, adaptive)
}

// Internal: caller owns the validated artifact and resolved policy.

function BuildLoadedLinear(a: FittedArtifact, observation: ObservationMode, adaptive: AdaptivePolicy?) -> LoadedLinearModel {
  let p = ArtifactRegression(a);
  let names = a.features;
  let id = a.model_id;
  let model_opts = ModelObservation(observation,id,a.schema_version,a.source_version);
  ModelLoaded(a, $trace = model_opts);
  if (adaptive != null) { return AdaptiveLinear(SelectiveLinear(a, p), adaptive); }
  let state = TracePrivate<RegressionPipeline> { value: p };
  LoadedLinearModel {
    predict_fn: (x: float[][], features: string[]) -> NumArrayResult {
      PredictLoadedLinear(x, features, id, names, state, observation, $trace = model_opts) catch (e) {
        let f: InferenceFailure => NumArrayResult { ok: false, value: [], error: f.message }
      }
    }
  }
}


function PrepareLogisticModel(raw: string, observation: ObservationMode = ObservationMode.Semantic, policy: AdaptivePolicy? = null) -> LoadedLogisticModel {
  let adaptive = ResolveAdaptivePolicy(observation, policy);
  let opts = ObservationOptions(observation);
  let a = PrepareArtifact(raw, false, $trace = if (observation == ObservationMode.Full) { trace.span(inputs = false, output = false) } else { opts });
  BuildLoadedLogistic(a, observation, adaptive)
}

// Internal: caller owns the validated artifact and resolved policy.

function BuildLoadedLogistic(a: FittedArtifact, observation: ObservationMode, adaptive: AdaptivePolicy?) -> LoadedLogisticModel {
  let p = ArtifactClassification(a);
  let names = a.features;
  let id = a.model_id;
  let model_opts = ModelObservation(observation,id,a.schema_version,a.source_version);
  ModelLoaded(a, $trace = model_opts);
  if (adaptive != null) { return AdaptiveClassifier(SelectiveLogistic(a, p), adaptive); }
  let state = TracePrivate<ClassificationPipeline> { value: p };
  LoadedLogisticModel {
    proba_fn: (x: float[][], features: string[]) -> ProbaResult {
      PredictLoadedProba(x, features, id, names, state, observation, $trace = model_opts) catch (e) {
        let f: InferenceFailure => ProbaResult { ok: false, probs: [], error: f.message }
      }
    },
    labels_fn: (x: float[][], features: string[]) -> IntArrayResult {
      PredictLoadedLabels(x, features, id, names, state, observation, $trace = model_opts) catch (e) {
        let f: InferenceFailure => IntArrayResult { ok: false, value: [], error: f.message }
      }
    }
  }
}


function LoadLinearModel(path: string, observation: ObservationMode = ObservationMode.Semantic, policy: AdaptivePolicy? = null) -> LoadedLinearModel {
  PrepareLinearModel(baml.fs.read(path), observation = observation, policy = policy)
}


function LoadLogisticModel(path: string, observation: ObservationMode = ObservationMode.Semantic, policy: AdaptivePolicy? = null) -> LoadedLogisticModel {
  PrepareLogisticModel(baml.fs.read(path), observation = observation, policy = policy)
}

// Internal execution helpers. Model ID is an explicit argument for tracing.
// These assume the factory has validated and privately captured the state.

class LoadedPipelineModel {
  estimator_kind string
  classes int[]
  predict_fn: (Cell[][], string[]) -> PipelineResult throws never
  proba_fn: (Cell[][], string[]) -> PipelineResult throws never
  function predict(self, cells: Cell[][], features: string[]) -> PipelineResult {
    let f = self.predict_fn;
    f(cells, features)
  }
  function predict_proba(self, cells: Cell[][], features: string[]) -> PipelineResult {
    let f = self.proba_fn;
    f(cells, features)
  }
}


function BuildLoadedPipeline(a: FittedPipelineArtifact, observation: ObservationMode, adaptive: AdaptivePolicy?) -> LoadedPipelineModel {
  let names = a.features;
  let id = a.model_id;
  let model_opts = ModelObservation(observation, id, a.schema_version, a.source_version);
  PipelineModelLoaded(a, $trace = model_opts);
  if (adaptive != null) { return AdaptivePipeline(SelectivePipeline(a), adaptive); }
  let state = TracePrivate<FittedPipelineArtifact> { value: a };
  LoadedPipelineModel {
    estimator_kind: a.estimator.kind,
    classes: a.estimator.classes,
    predict_fn: (cells: Cell[][], features: string[]) -> PipelineResult {
      PredictLoadedPipeline(cells, features, id, names, state, observation, Operation.Predict, $trace = model_opts) catch (e) {
        let f: InferenceFailure => PipelineResult { ok: false, values: [], labels: [], probs: [], error: f.message }
      }
    },
    proba_fn: (cells: Cell[][], features: string[]) -> PipelineResult {
      PredictLoadedPipeline(cells, features, id, names, state, observation, Operation.PredictProba, $trace = model_opts) catch (e) {
        let f: InferenceFailure => PipelineResult { ok: false, values: [], labels: [], probs: [], error: f.message }
      }
    }
  }
}


function PreparePipelineModel(raw: string, observation: ObservationMode = ObservationMode.Semantic, policy: AdaptivePolicy? = null) -> LoadedPipelineModel {
  let adaptive = ResolveAdaptivePolicy(observation, policy);
  let opts = ObservationOptions(observation);
  let a = PreparePipelineArtifact(raw, $trace = if (observation == ObservationMode.Full) { trace.span(inputs = false, output = false) } else { opts });
  BuildLoadedPipeline(a, observation, adaptive)
}


function LoadPipelineModel(path: string, observation: ObservationMode = ObservationMode.Semantic, policy: AdaptivePolicy? = null) -> LoadedPipelineModel {
  PreparePipelineModel(baml.fs.read(path), observation = observation, policy = policy)
}


