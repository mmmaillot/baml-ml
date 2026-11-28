// Adaptive-mode handle wrappers around SelectiveModel.
function AdaptiveLinear(model: SelectiveModel, policy: AdaptivePolicy) -> LoadedLinearModel {
    let opts = SelectiveContext(model, trace.empty_span(), "");
    LoadedLinearModel {
        predict_fn: (x: float[][], features: string[]) -> NumArrayResult {
            let r = RunAdaptive(model, x, features, "predict", policy, $trace = opts) catch (e) {
                let f: InferenceFailure => {
                    return NumArrayResult {
                        ok: false,
                        value: [],
                        error: f.message,
                    };
                },
            };
            NumArrayResult {
                ok: true,
                value: r.values,
                error: "",
            }
        },
    }
}


function AdaptivePipeline(model: SelectiveModel, policy: AdaptivePolicy) -> LoadedPipelineModel {
    let opts = SelectiveContext(model, trace.empty_span(), "");
    LoadedPipelineModel {
        estimator_kind: model.estimator_kind,
        classes: model.classes,
        predict_fn: (cells: Cell[][], features: string[]) -> PipelineResult {
            let r = RunAdaptiveCells(model, cells, features, "predict", policy, $trace = opts) catch (e) {
                let f: InferenceFailure => {
                    return PipelineResult {
                        ok: false,
                        values: [],
                        labels: [],
                        probs: [],
                        error: f.message,
                    };
                },
            };
            PipelineResult {
                ok: true,
                values: r.values,
                labels: r.labels,
                probs: [],
                error: "",
            }
        },
        proba_fn: (cells: Cell[][], features: string[]) -> PipelineResult {
            let r = RunAdaptiveCells(
                model,
                cells,
                features,
                "predict-proba",
                policy,
                $trace = opts,
            ) catch (e) {
                let f: InferenceFailure => {
                    return PipelineResult {
                        ok: false,
                        values: [],
                        labels: [],
                        probs: [],
                        error: f.message,
                    };
                },
            };
            PipelineResult {
                ok: true,
                values: [],
                labels: [],
                probs: r.probabilities,
                error: "",
            }
        },
    }
}


function AdaptiveClassifier(model: SelectiveModel, policy: AdaptivePolicy) -> LoadedLogisticModel {
    let opts = SelectiveContext(model, trace.empty_span(), "");
    LoadedLogisticModel {
        labels_fn: (x: float[][], features: string[]) -> IntArrayResult {
            let r = RunAdaptive(model, x, features, "predict", policy, $trace = opts) catch (e) {
                let f: InferenceFailure => {
                    return IntArrayResult {
                        ok: false,
                        value: [],
                        error: f.message,
                    };
                },
            };
            IntArrayResult {
                ok: true,
                value: r.labels,
                error: "",
            }
        },
        proba_fn: (x: float[][], features: string[]) -> ProbaResult {
            let r = RunAdaptive(
                model,
                x,
                features,
                "predict-proba",
                policy,
                $trace = opts,
            ) catch (e) {
                let f: InferenceFailure => {
                    return ProbaResult {
                        ok: false,
                        probs: [],
                        error: f.message,
                    };
                },
            };
            ProbaResult {
                ok: true,
                probs: r.probabilities,
                error: "",
            }
        },
    }
}

