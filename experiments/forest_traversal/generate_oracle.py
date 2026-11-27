"""Generate independent sklearn 1.8 random forest oracles (labels, exact
probabilities and per-tree leaf IDs), including binary32 threshold boundaries
and an exact probability tie."""
import json
import sys
from pathlib import Path
import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

assert sklearn.__version__ == "1.8.0", sklearn.__version__
rng = np.random.default_rng(42)
x, y = load_iris(return_X_y=True)
cases = []


def forest_state(model, n_features):
    classes = [int(c) for c in model.classes_]
    trees = []
    for est in model.estimators_:
        tree = est.tree_
        nodes = []
        for i in range(tree.node_count):
            left, right = int(tree.children_left[i]), int(tree.children_right[i])
            feat, threshold = int(tree.feature[i]), float(tree.threshold[i])
            assert left == -1 or (i < left and i < right)
            nodes.append(dict(left=left, right=right, feature=feat, threshold=threshold,
                              probabilities=tree.value[i, 0].tolist() if left == -1 else []))
        trees.append(dict(n_features=n_features, classes=classes, nodes=nodes))
    return dict(n_features=n_features, classes=classes, trees=trees)


def boundary_rows(model, x):
    extra = []
    for est in model.estimators_:
        tree = est.tree_
        reaches_all = est.decision_path(x).toarray()
        for i in range(tree.node_count):
            left = int(tree.children_left[i])
            if left == -1:
                continue
            feat, threshold = int(tree.feature[i]), float(tree.threshold[i])
            # Values at and immediately around the exact binary32 rounding boundary.
            f = np.float32(threshold)
            lower = f if float(f) <= threshold else np.nextafter(f, np.float32(-np.inf))
            upper = np.nextafter(lower, np.float32(np.inf))
            mid = (float(lower) + float(upper)) / 2
            row = x[int(np.flatnonzero(reaches_all[:, i])[0])].copy()
            for v in [float(lower), np.nextafter(mid, -np.inf), mid, np.nextafter(mid, np.inf), float(upper), threshold]:
                r = row.copy()
                r[feat] = v
                extra.append(r)
    return extra


def add_case(name, x, y, **kwargs):
    fit_weights = kwargs.pop("sample_weight", None)
    model = RandomForestClassifier(random_state=42, **kwargs).fit(x, y, sample_weight=fit_weights)
    extra = boundary_rows(model, x)
    rows = np.vstack([x, *extra]) if extra else x
    expected = dict(labels=model.predict(rows).tolist(), probs=model.predict_proba(rows).tolist(),
                    leaves=model.apply(rows).tolist())
    cases.append(dict(name=name, state=forest_state(model, x.shape[1]), rows=rows.tolist(), expected=expected))


add_case("iris-multiclass", x, y, n_estimators=10, max_depth=5)
add_case("weighted-binary", x[y != 2], np.where(y[y != 2] == 0, -3, 7),
         n_estimators=7, max_leaf_nodes=5, sample_weight=rng.uniform(0.1, 3.0, size=100))
# bootstrap=False with unsplittable two-sample trees gives constant
# [0.5, 0.5] leaves: an exact averaged-probability tie that must resolve to
# the first class.
add_case("exact-tie", np.array([[0.0], [1.0]]), np.array([-3, 7]),
         n_estimators=2, min_samples_split=10, bootstrap=False)
# A nonrepresentable threshold and adjacent float32 values exercise rounding
# at actual learned splits of every tree, not merely hand-written forests.
add_case("adjacent-floats", np.array([[1.0], [1.000000238418579], [1.0000004768371582], [1.0000007152557373]]),
         np.array([0, 0, 1, 1]), n_estimators=3, max_depth=1)

Path(sys.argv[1]).write_text(json.dumps(dict(forests=cases), allow_nan=False), encoding="utf-8")
print(f"Generated {len(cases)} forest cases, {sum(len(c['rows']) for c in cases)} rows")
