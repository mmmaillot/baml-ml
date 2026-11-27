// Version 4 is deliberately forest-only; the v1, v2 and future v3 (arbitrary
// sequential pipelines, issue #73) schemas remain unchanged.
class FittedForestArtifact {
    schema_version: int,
    source: string,
    source_version: string,
    model_id: string,
    features: string[],
    imputer_kind: string,
    scaler_kind: string,
    estimator_kind: string,
    imputer_means: float[],
    scaler_means: float[],
    scaler_scales: float[],
    classes: int[],
    n_features: int,
    trees: TreeClassifierState[],
}

class PreparedForestPipeline {
    imputer: MeanImputer,
    scaler: StandardScaler,
    forest: ForestClassifierState,
}

class LoadedForestModel {
    proba_fn: (float[][], string[]) -> ProbaResult throws never,
    labels_fn: (float[][], string[]) -> IntArrayResult throws never,
    function predict_proba(self, x: float[][], features: string[]) -> ProbaResult {
        let f = self.proba_fn;
        f(x, features)
    }
    function predict(self, x: float[][], features: string[]) -> IntArrayResult {
        let f = self.labels_fn;
        f(x, features)
    }
}

function DecodeForestArtifact(raw: string) -> FittedForestArtifact {
    baml.json.decode<FittedForestArtifact>(raw)
}

function ForestArtifactError(a: FittedForestArtifact) -> string {
    let n = a.features.length();
    if (a.schema_version != 4 || a.estimator_kind != "random_forest_classifier") {
        return "artifact: unsupported forest schema or estimator";
    }
    if (a.source != "scikit-learn" || a.source_version == "" || a.model_id == "") {
        return "artifact: invalid provenance";
    }
    if (a.imputer_kind != "mean" || a.scaler_kind != "standard") {
        return "artifact: unsupported preprocessing";
    }
    if (
        n == 0
            || a.n_features != n
            || a.imputer_means.length() != n
            || a.scaler_means.length() != n
            || a.scaler_scales.length() != n
    ) {
        return "artifact: invalid preprocessing dimensions";
    }
    for (let j = 0; j < n; j += 1) {
        if (a.features[j] == "") {
            return "artifact: empty feature name";
        }
        for (let k = 0; k < j; k += 1) {
            if (a.features[j] == a.features[k]) {
                return "artifact: duplicate feature name";
            }
        }
        if (
            !IsFinite(a.imputer_means[j])
                || !IsFinite(a.scaler_means[j])
                || !IsFinite(a.scaler_scales[j])
                || a.scaler_scales[j] <= 0.0
        ) {
            return "artifact: invalid preprocessing parameters";
        }
    }
    ForestStateError(ForestClassifierState {
        n_features: n,
        classes: a.classes,
        trees: a.trees,
    })
}

function PrepareForestArtifact(raw: string) -> FittedForestArtifact {
    let a = DecodeForestArtifact(raw);
    let err = ForestArtifactError(a);
    if (err != "") {
        throw baml.errors.InvalidArgument { message: err };
    }
    a
}

function ForestModelLoaded(artifact: FittedForestArtifact) -> void {}

// A single semantic batch span, not one full span per tree or per node.
// Full mode exposes averaged probabilities, labels and per-row leaf IDs per
// tree; together with the load record and scaled inputs these reconstruct any
// tree path, split feature and threshold.
function ObservedForestTraversal(
    x: float[][],
    state: TracePrivate<ForestClassifierState>,
    capture_leaves: bool,
) -> ForestBatchResult throws InferenceFailure {
    ForestPredictCore(x, state.value, capture_leaves, $trace = trace.timing())
}

function PredictLoadedForest(
    x: float[][],
    features: string[],
    model_id: string,
    expected: string[],
    state: TracePrivate<PreparedForestPipeline>,
    observation: ObservationMode,
) -> ForestBatchResult throws InferenceFailure {
    let err = LoadedFeatureError(features, expected);
    if (err != "") {
        throw InferenceFailure {
            stage: "input_validation",
            message: err,
        };
    }
    let p = state.value;
    let opts = ObservationOptions(observation);
    let t = PreparedPreprocess(x, p.imputer, p.scaler, opts);
    ObservedForestTraversal(
        t,
        TracePrivate<ForestClassifierState> { value: p.forest },
        observation == ObservationMode.Full,
        $trace = opts,
    )
}

function PreparedForest(a: FittedForestArtifact) -> PreparedForestPipeline {
    let n = a.features.length();
    PreparedForestPipeline {
        imputer: MeanImputer {
            means: a.imputer_means,
            n_features: n,
        },
        scaler: StandardScaler {
            means: a.scaler_means,
            scales: a.scaler_scales,
            n_features: n,
        },
        forest: ForestClassifierState {
            n_features: n,
            classes: a.classes,
            trees: a.trees,
        },
    }
}

// Internal: caller owns the validated artifact and resolved policy.
function BuildLoadedForest(
    a: FittedForestArtifact,
    observation: ObservationMode,
    adaptive: AdaptivePolicy?,
) -> LoadedForestModel {
    let names = a.features;
    let id = a.model_id;
    let model_opts = ModelObservation(observation, id, a.schema_version, a.source_version);
    ForestModelLoaded(a, $trace = model_opts);
    let p = PreparedForest(a);
    if (adaptive != null) {
        let classifier = AdaptiveClassifier(SelectiveForest(a, p), adaptive);
        return LoadedForestModel {
            proba_fn: classifier.proba_fn,
            labels_fn: classifier.labels_fn,
        };
    }
    let state = TracePrivate<PreparedForestPipeline> { value: p };
    LoadedForestModel {
        proba_fn: (x: float[][], features: string[]) -> ProbaResult {
            ForestResultProba(PredictLoadedForest(
                x,
                features,
                id,
                names,
                state,
                observation,
                $trace = model_opts,
            )) catch (e) {
                let f: InferenceFailure => ProbaResult {
                    ok: false,
                    probs: [],
                    error: f.message,
                },
            }
        },
        labels_fn: (x: float[][], features: string[]) -> IntArrayResult {
            ForestResultLabels(PredictLoadedForest(
                x,
                features,
                id,
                names,
                state,
                observation,
                $trace = model_opts,
            )) catch (e) {
                let f: InferenceFailure => IntArrayResult {
                    ok: false,
                    value: [],
                    error: f.message,
                },
            }
        },
    }
}

function PrepareForestModel(
    raw: string,
    observation: ObservationMode = ObservationMode.Semantic,
    policy: AdaptivePolicy? = null,
) -> LoadedForestModel {
    let adaptive = ResolveAdaptivePolicy(observation, policy);
    let opts = ObservationOptions(observation);
    let a = PrepareForestArtifact(raw, $trace = if (observation == ObservationMode.Full) {
        trace.span(inputs = false, output = false)
    } else {
        opts
    });
    BuildLoadedForest(a, observation, adaptive)
}

function LoadForestModel(
    path: string,
    observation: ObservationMode = ObservationMode.Semantic,
    policy: AdaptivePolicy? = null,
) -> LoadedForestModel {
    PrepareForestModel(
        baml.fs.read(path),
        observation = observation,
        policy = policy,
    )
}

function RunForestBatch(
    raw_model: string,
    raw_input: string,
    operation: string,
    mode: ObservationMode,
    adaptive: AdaptivePolicy?,
) -> BatchOutput {
    let opts = ObservationOptions(mode);
    let a = PrepareForestArtifact(raw_model, $trace = if (mode == ObservationMode.Full) {
        trace.span(inputs = false, output = false)
    } else {
        opts
    });
    let input = DecodeBatchInput(raw_input);
    let rows = BatchRows(input);
    let m = BuildLoadedForest(a, mode, adaptive);
    if (operation == "predict-proba") {
        let r = m.predict_proba(rows, input.features);
        if (!r.ok) {
            throw baml.errors.InvalidArgument { message: r.error };
        }
        return BatchOutput {
            schema_version: 1,
            model_id: a.model_id,
            estimator_kind: a.estimator_kind,
            operation: operation,
            classes: a.classes,
            predictions: r.probs,
        };
    }
    let r = m.predict(rows, input.features);
    if (!r.ok) {
        throw baml.errors.InvalidArgument { message: r.error };
    }
    BatchOutput {
        schema_version: 1,
        model_id: a.model_id,
        estimator_kind: a.estimator_kind,
        operation: operation,
        classes: a.classes,
        predictions: r.value,
    }
}

function ForestResultProba(r: ForestBatchResult) -> ProbaResult {
    ProbaResult {
        ok: true,
        probs: r.probs,
        error: "",
    }
}

function ForestResultLabels(r: ForestBatchResult) -> IntArrayResult {
    IntArrayResult {
        ok: true,
        value: r.labels,
        error: "",
    }
}
