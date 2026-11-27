"""Generate independent sklearn 1.8 tree/path and NumPy binary32 oracles."""
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

assert sklearn.__version__ == "1.8.0", sklearn.__version__
rng = np.random.default_rng(42)
x, y = load_iris(return_X_y=True)
cases = []

def add_case(name, x, y, **kwargs):
    fit_weights = kwargs.pop("sample_weight", None)
    model = DecisionTreeClassifier(random_state=42, **kwargs).fit(x, y, sample_weight=fit_weights)
    tree = model.tree_
    nodes = []
    extra = []
    for i in range(tree.node_count):
        left, right = int(tree.children_left[i]), int(tree.children_right[i])
        feat, threshold = int(tree.feature[i]), float(tree.threshold[i])
        assert left == -1 or (i < left and i < right)
        nodes.append(dict(left=left, right=right, feature=feat, threshold=threshold,
                          probabilities=tree.value[i, 0].tolist() if left == -1 else []))
        if left != -1:
            # Values at and immediately around the exact binary32 rounding boundary.
            f = np.float32(threshold)
            lower = f if float(f) <= threshold else np.nextafter(f, np.float32(-np.inf))
            upper = np.nextafter(lower, np.float32(np.inf))
            mid = (float(lower) + float(upper)) / 2
            # Include a training example reaching this node to exercise its branch.
            reaches = model.decision_path(x).getcol(i).toarray().ravel().astype(bool)
            row = x[np.flatnonzero(reaches)[0]].copy()
            for v in [float(lower), np.nextafter(mid, -np.inf), mid, np.nextafter(mid, np.inf), float(upper), threshold]:
                r = row.copy()
                r[feat] = v
                extra.append(r)
    rows = np.vstack([x, *extra]) if extra else x
    # CSR path stores nodes along parent-before-child IDs.
    paths = model.decision_path(rows)
    expected = dict(labels=model.predict(rows).tolist(), probs=model.predict_proba(rows).tolist(),
                    leaves=model.apply(rows).tolist(),
                    paths=[paths.indices[paths.indptr[i]:paths.indptr[i+1]].tolist() for i in range(len(rows))])
    state = dict(n_features=x.shape[1], classes=model.classes_.tolist(), nodes=nodes)
    cases.append(dict(name=name, state=state, rows=rows.tolist(), expected=expected))

add_case("iris-multiclass", x, y, max_depth=5)
add_case("weighted-binary", x[y != 2], np.where(y[y != 2] == 0, -3, 7),
         max_leaf_nodes=5, sample_weight=rng.uniform(0.1, 3.0, size=100))
add_case("constant-stump", np.zeros((4,1)), np.array([-3,7,-3,7]))
# A nonrepresentable threshold and two adjacent float32 values exercise
# rounding at an actual learned split, not merely a hand-written tree.
add_case("adjacent-floats", np.array([[1.0],[1.000000238418579],[1.0000004768371582],[1.0000007152557373]]),
         np.array([0,0,1,1]), max_depth=1)

values = [0.0, -0.0, 1.0, -1.0, np.finfo(np.float64).tiny, np.nextafter(0.,1.),
          2.**-150, 2.**-149, 2.**-126, float(np.finfo(np.float32).max),
          3.4028235677973366e38, 1e300, 1.0000000596046448, 1.0000001788139343]
for v in list(values):
    values.extend([float(np.nextafter(v,-np.inf)),float(np.nextafter(v,np.inf)),-float(v)])
values.extend(np.ldexp(rng.uniform(-2,2,512), rng.integers(-1070,1020,512)).tolist())
rounding = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    for v in values:
        if not np.isfinite(v):
            continue
        r = np.float32(v)
        rounding.append(dict(input=float(v),rounded=float(r) if np.isfinite(r) else None,
                             negative_zero=bool(r == 0 and np.signbit(r))))
Path(sys.argv[1]).write_text(json.dumps(dict(trees=cases, rounding=rounding),allow_nan=False),encoding="utf-8")
print(f"Generated {len(cases)} tree cases, {sum(len(c['rows']) for c in cases)} rows and {len(rounding)} binary32 cases")
