// Decision tree traversal and batch inference core.
// Fitted decision tree traversal with a checked public entry point.
// sklearn converts dense features to binary32 before comparing f64 thresholds.

class TreeNode {
  left int
  right int
  feature int
  threshold float
  probabilities float[]
}


class TreeClassifierState {
  n_features int
  classes int[]
  nodes TreeNode[]
}


class TreeBatchResult {
  labels int[]
  probs float[][]
  leaves int[]
  paths int[][]
}


class TreeRowTrace {
  leaf int
  path int[]
}

// Internal: state is validated and the row is already rounded to binary32.
// Returns the leaf ID, plus ordered visited node IDs when capture_path is set.

function TreeTraverse(row: float[], t: TreeClassifierState, capture_path: bool) -> TreeRowTrace {
  let id = 0;
  let path: int[] = [];
  if (capture_path) { path.push(0); }
  while (t.nodes[id].left != -1) {
    let node = t.nodes[id];
    id = if (row[node.feature] <= node.threshold) { node.left } else { node.right };
    if (capture_path) { path.push(id); }
  }
  TreeRowTrace { leaf: id, path: path }
}

// Round f64 to IEEE-754 binary32 (nearest, ties to even), returned as f64.
// Scale only by powers of two so no transcendental/log approximation enters
// threshold semantics. Not a general numerical library or optimized kernel.

function TreeRoundFloat32(x: float) -> float {
  if (!IsFinite(x) || x == 0.0) { return x; }
  let mag = x.abs();
  if (mag >= 3.4028235677973366e38) { return x.signum() * float.inf(); }
  let quantum = 1.401298464324817e-45;
  if (mag >= 1.1754943508222875e-38) {
    let normalized = mag;
    quantum = 1.1920928955078125e-7;
    while (normalized >= 2.0) {
      normalized = normalized / 2.0;
      quantum = quantum * 2.0;
    }
    while (normalized < 1.0) {
      normalized = normalized * 2.0;
      quantum = quantum / 2.0;
    }
  }
  let units = mag / quantum;
  let lower = units.floor();
  let remainder = units - lower;
  if (remainder > 0.5 || (remainder == 0.5 && (lower / 2.0).floor() * 2.0 != lower)) {
    lower = lower + 1.0;
  }
  x.signum() * (lower * quantum)
}

// Exported sklearn node IDs are parent-before-child. Requiring exactly one
// parent for every nonroot node plus forward-only edges proves reachability
// and excludes cycles without recursion on potentially malformed input.

function TreeStateError(t: TreeClassifierState) -> string {
  let n = t.nodes.length();
  if (t.n_features <= 0 || n == 0 || t.classes.length() == 0) { return "tree: invalid dimensions"; }
  for (let k = 1; k < t.classes.length(); k += 1) {
    if (t.classes[k] <= t.classes[k - 1]) { return "tree: invalid class order"; }
  }
  let parents: int[] = [];
  for (let i = 0; i < n; i += 1) { parents.push(0); }
  for (let i = 0; i < n; i += 1) {
    let node = t.nodes[i];
    if (node.left == -1 && node.right == -1) {
      if (node.feature != -2 || node.threshold != -2.0 || node.probabilities.length() != t.classes.length()) { return "tree: malformed leaf"; }
      let total = 0.0;
      for (let k = 0; k < node.probabilities.length(); k += 1) {
        let p = node.probabilities[k];
        if (!IsFinite(p) || p < 0.0 || p > 1.0) { return "tree: invalid leaf probability"; }
        total = total + p;
      }
      if ((total - 1.0).abs() > 1e-12) { return "tree: leaf probabilities do not sum to one"; }
    } else {
      if (node.left <= i || node.right <= i || node.left >= n || node.right >= n || node.left == node.right) { return "tree: invalid child edge"; }
      if (node.feature < 0 || node.feature >= t.n_features || !IsFinite(node.threshold) || node.probabilities.length() != 0) { return "tree: malformed split"; }
      parents[node.left] = parents[node.left] + 1;
      parents[node.right] = parents[node.right] + 1;
    }
  }
  for (let i = 1; i < n; i += 1) {
    if (parents[i] != 1) { return "tree: shared or unreachable node"; }
  }
  ""
}

// Checked experimental entry point. Returns paths as evidence for the next
// story; production capture policy and prepared state are not implemented yet.

function TreePredictBatch(x: float[][], t: TreeClassifierState) -> TreeBatchResult {
  let err = TreeStateError(t);
  if (err != "") { throw baml.errors.InvalidArgument { message: err }; }
  TreePredictCore(x,t,true) catch (e) {
    let f: InferenceFailure => throw baml.errors.InvalidArgument { message: f.message }
  }
}

// Internal: topology and state have already been validated.

function TreePredictCore(x: float[][], t: TreeClassifierState, capture_path: bool) -> TreeBatchResult throws InferenceFailure {
  let d = CheckFiniteMatrix(x);
  if (!d.ok || d.cols != t.n_features) { throw InferenceFailure { stage: "tree_traversal", message: "tree: invalid finite input matrix" }; }
  let labels: int[] = [];
  let probs: float[][] = [];
  let leaves: int[] = [];
  let paths: int[][] = [];
  for (let i = 0; i < x.length(); i += 1) {
    let row: float[] = [];
    for (let j = 0; j < d.cols; j += 1) {
      let v = TreeRoundFloat32(x[i][j]);
      if (!IsFinite(v)) { throw InferenceFailure { stage: "tree_traversal", message: "tree: float32 input overflow" }; }
      row.push(v);
    }
    let trace = TreeTraverse(row, t, capture_path);
    let id = trace.leaf;
    let path: int[] = [];
    if (capture_path) { path = trace.path; }
    let p: float[] = [];
    let best = 0;
    for (let k = 0; k < t.classes.length(); k += 1) {
      p.push(t.nodes[id].probabilities[k]);
      if (p[k] > p[best]) { best = k; }
    }
    labels.push(t.classes[best]);
    probs.push(p);
    leaves.push(id);
    if (capture_path) { paths.push(path); }
  }
  TreeBatchResult { labels: labels, probs: probs, leaves: leaves, paths: paths }
}

