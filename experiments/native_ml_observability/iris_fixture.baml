// Frozen external oracle: sklearn 1.8.0, Iris, 80/20 stratified split,
// seed 42, mean imputer, StandardScaler, LogisticRegression(C=1,
// tol=1e-10, max_iter=10000). No Python needed at runtime.
function IrisFrozenPipeline() -> ClassificationPipeline {
  ClassificationPipeline {
    imputer: MeanImputer { means: [5.841666666666668, 3.0483333333333342, 3.769999999999999, 1.2049999999999987], n_features: 4 },
    scaler: StandardScaler { means: [5.841666666666668, 3.0483333333333342, 3.769999999999999, 1.2049999999999987], scales: [0.837415003978845, 0.44665111913239636, 1.7611359970201048, 0.7594789880788891], n_features: 4 },
    model: LogisticRegressionModel {
      classes: [0, 1, 2],
      weights: [[-1.0888755974042648, 1.023930224092558, -1.7988707919193383, -1.6851856429587624], [0.5361590297569796, -0.3600632080744804, -0.2045955001712917, -0.8083561894884974], [0.5527165676472843, -0.6638670160180775, 2.0034662920906294, 2.4935418324472605]],
      intercepts: [-0.3053113803154904, 1.9086393085791917, -1.6033279282637012],
      n_features: 4, n_iter: 29, converged: true
    }
  }
}

function IrisMissing() -> ProbaResult {
  ClassificationPipelinePredictProba([[5.1, float.nan(), 1.4, 0.2]], IrisFrozenPipeline())
}

function IrisComplete() -> ProbaResult {
  ClassificationPipelinePredictProba([[5.1, 3.5, 1.4, 0.2]], IrisFrozenPipeline())
}
