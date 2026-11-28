// Random forest batch inference core.
// Fitted random forest traversal reusing the validated binary32 tree core.
// sklearn RandomForestClassifier.predict_proba is the arithmetic mean of the
// normalized leaf class probabilities of every tree, accumulated in estimator
// order (bit-exact sklearn 1.8.0 parity); predict is the argmax of that mean
// with first-class tie breaking.

class ForestClassifierState {
  n_features int
  classes int[]
  trees TreeClassifierState[]
}


class ForestBatchResult {
  labels int[]
  probs float[][]
  leaves int[][]
}


function ForestStateError(f: ForestClassifierState) -> string {
  let n = f.n_features;
  let n_classes = f.classes.length();
  if (n <= 0 || n_classes == 0 || f.trees.length() == 0) {
    return "forest: invalid dimensions";
  }
  for (let k = 1; k < n_classes; k += 1) {
    if (f.classes[k] <= f.classes[k - 1]) { return "forest: invalid class order"; }
  }
  for (let t = 0; t < f.trees.length(); t += 1) {
    let tree = f.trees[t];
    if (tree.n_features != n || tree.classes != f.classes) {
      return "forest: tree " + t.to_string() + " dimensions or classes differ from the forest";
    }
    let err = TreeStateError(tree);
    if (err != "") { return "forest: tree " + t.to_string() + ": " + err; }
  }
  ""
}

// Checked public entry point.

function ForestPredictBatch(x: float[][], f: ForestClassifierState) -> ForestBatchResult {
  let err = ForestStateError(f);
  if (err != "") { throw baml.errors.InvalidArgument { message: err }; }
  ForestPredictCore(x, f, true) catch (e) {
    let fail: InferenceFailure => throw baml.errors.InvalidArgument { message: fail.message }
  }
}

// Internal: topology and state have already been validated. Preprocessed
// dense features are cast to binary32 once per row and shared by all trees.

function ForestPredictCore(x: float[][], f: ForestClassifierState, capture_leaves: bool) -> ForestBatchResult throws InferenceFailure {
  let d = CheckFiniteMatrix(x);
  if (!d.ok || d.cols != f.n_features) {
    throw InferenceFailure { stage: "forest_traversal", message: "forest: invalid finite input matrix" };
  }
  let n_trees = f.trees.length();
  let labels: int[] = [];
  let probs: float[][] = [];
  let leaves: int[][] = [];
  for (let i = 0; i < x.length(); i += 1) {
    let row: float[] = [];
    for (let j = 0; j < d.cols; j += 1) {
      let v = TreeRoundFloat32(x[i][j]);
      if (!IsFinite(v)) {
        throw InferenceFailure { stage: "forest_traversal", message: "forest: float32 input overflow" };
      }
      row.push(v);
    }
    let acc: float[] = [];
    for (let k = 0; k < f.classes.length(); k += 1) { acc.push(0.0); }
    let row_leaves: int[] = [];
    for (let t = 0; t < n_trees; t += 1) {
      let tree = f.trees[t];
      let id = TreeTraverse(row, tree, false).leaf;
      let node = tree.nodes[id];
      for (let k = 0; k < f.classes.length(); k += 1) {
        acc[k] = acc[k] + node.probabilities[k];
      }
      if (capture_leaves) { row_leaves.push(id); }
    }
    let p: float[] = [];
    let best = 0;
    for (let k = 0; k < f.classes.length(); k += 1) {
      p.push(acc[k] / n_trees);
      if (p[k] > p[best]) { best = k; }
    }
    labels.push(f.classes[best]);
    probs.push(p);
    if (capture_leaves) { leaves.push(row_leaves); }
  }
  ForestBatchResult { labels: labels, probs: probs, leaves: leaves }
}

// A single semantic batch span, not one full span per tree or per node.
// Full mode exposes averaged probabilities, labels and per-row leaf IDs per
// tree; together with the load record and preprocessed inputs these
// reconstruct any tree path, split feature and threshold.

