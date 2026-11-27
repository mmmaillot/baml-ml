"""Export only the deliberately narrow, dense, float64 sklearn v1/v2/v4 contract."""
import hashlib
import json
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted


def _finite(value, shape):
    a = np.asarray(value)
    if a.shape != shape or a.dtype != np.float64 or not np.isfinite(a).all():
        raise ValueError("expected finite float64 parameters with exact dimensions")
    return a.tolist()


def _integer_classes(model):
    labels = np.asarray(model.classes_)
    if labels.ndim != 1 or labels.dtype.kind not in "iu" or not len(labels):
        raise ValueError("only integer classes supported")
    classes = [int(x) for x in labels]
    if any(x < -(2**63) or x >= 2**63 for x in classes) or classes != sorted(set(classes)):
        raise ValueError("invalid class order/range")
    return classes


def _tree_nodes(tree, n_classes):
    nodes = []
    for i in range(tree.node_count):
        left, right = int(tree.children_left[i]), int(tree.children_right[i])
        leaf = left == right == -1
        probs = _finite(tree.value[i, 0], (n_classes,)) if leaf else []
        if leaf and (min(probs) < 0 or max(probs) > 1 or abs(sum(probs)-1) > 1e-12):
            raise ValueError("unsupported leaf probabilities")
        threshold = float(tree.threshold[i])
        if not np.isfinite(threshold):
            raise ValueError("nonfinite split threshold")
        nodes.append(dict(left=left, right=right, feature=int(tree.feature[i]), threshold=threshold, probabilities=probs))
    return nodes


def _model_id(a):
    return hashlib.sha256(json.dumps(a, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def export_pipeline(pipeline, features):
    if type(pipeline) is not Pipeline or len(pipeline.steps) != 3:
        raise ValueError("expected exactly imputer, scaler, estimator")
    imp, scale, model = [step for _, step in pipeline.steps]
    if type(imp) is not SimpleImputer or imp.strategy != "mean" or imp.add_indicator or not np.isscalar(imp.missing_values) or not np.isnan(imp.missing_values):
        raise ValueError("unsupported imputer")
    if type(scale) is not StandardScaler or not scale.with_mean or not scale.with_std:
        raise ValueError("unsupported scaler")
    if type(model) not in (LinearRegression, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier):
        raise ValueError("unsupported estimator")
    for step in (imp, scale, model):
        check_is_fitted(step)
    features = list(features)
    n = len(features)
    if not n or any(not isinstance(f, str) or not f for f in features) or len(set(features)) != n:
        raise ValueError("invalid feature names")
    if hasattr(pipeline, "feature_names_in_") and list(pipeline.feature_names_in_) != features:
        raise ValueError("feature order differs from fitted pipeline")
    if any(step.n_features_in_ != n for step in (imp, scale, model)):
        raise ValueError("unsupported dimensionality change")
    means = _finite(imp.statistics_, (n,))
    sm = _finite(scale.mean_, (n,))
    ss = _finite(scale.scale_, (n,))
    if min(ss) <= 0:
        raise ValueError("nonpositive scale")
    classes = []
    if type(model) is LinearRegression:
        kind = "linear"
        weights = [_finite(model.coef_, (n,))]
        intercepts = [_finite(np.asarray(model.intercept_), ())]
    elif type(model) is DecisionTreeClassifier:
        if model.n_outputs_ != 1:
            raise ValueError("only single-output classification trees supported")
        classes = _integer_classes(model)
        nodes = _tree_nodes(model.tree_, len(classes))
        a = dict(schema_version=2, source="scikit-learn", source_version=sklearn.__version__,
                 features=features, imputer_kind="mean", scaler_kind="standard",
                 estimator_kind="decision_tree_classifier", imputer_means=means, scaler_means=sm,
                 scaler_scales=ss, classes=classes, nodes=nodes)
        a["model_id"] = _model_id(a)
        return a
    elif type(model) is RandomForestClassifier:
        if model.n_outputs_ != 1:
            raise ValueError("only single-output forests supported")
        classes = _integer_classes(model)
        if len(model.estimators_) < 1:
            raise ValueError("fitted forest has no estimators")
        trees = []
        for est in model.estimators_:
            if type(est) is not DecisionTreeClassifier or est.n_outputs_ != 1:
                raise ValueError("unsupported forest base estimator")
            if est.n_features_in_ != n or est.tree_.n_features != n:
                raise ValueError("unsupported dimensionality change")
            # sklearn fits forest trees on class indices 0..K-1, so every
            # leaf vector is already aligned with the sorted forest classes.
            if [int(c) for c in np.asarray(est.classes_)] != list(range(len(classes))):
                raise ValueError("base estimator classes differ from the forest")
            if est.tree_.value.shape != (est.tree_.node_count, 1, len(classes)):
                raise ValueError("base estimator leaf dimensions differ from the forest")
            trees.append(dict(n_features=n, classes=classes,
                              nodes=_tree_nodes(est.tree_, len(classes))))
        a = dict(schema_version=4, source="scikit-learn", source_version=sklearn.__version__,
                 features=features, imputer_kind="mean", scaler_kind="standard",
                 estimator_kind="random_forest_classifier", imputer_means=means, scaler_means=sm,
                 scaler_scales=ss, classes=classes, n_features=n, trees=trees)
        a["model_id"] = _model_id(a)
        return a
    else:
        if model.solver == "liblinear" or getattr(model, "multi_class", None) == "ovr":
            raise ValueError("unsupported OvR/logistic solver")
        classes = _integer_classes(model)
        if len(classes) < 2:
            raise ValueError("only integer classes supported")
        if np.any(np.asarray(model.n_iter_) >= model.max_iter):
            raise ValueError("fit may not have converged")
        kind = "logistic_binary" if len(classes) == 2 else "logistic_multinomial"
        k = 1 if len(classes) == 2 else len(classes)
        weights = _finite(model.coef_, (k, n))
        intercepts = _finite(model.intercept_, (k,))
        if len(classes) == 2 and getattr(model, "multi_class", None) == "multinomial":
            raise ValueError("legacy binary multinomial semantics unsupported")
    a = dict(schema_version=1, source="scikit-learn", source_version=sklearn.__version__,
             features=features, imputer_kind="mean", scaler_kind="standard",
             estimator_kind=kind, imputer_means=means, scaler_means=sm,
             scaler_scales=ss, weights=weights, intercepts=intercepts, classes=classes)
    a["model_id"] = _model_id(a)
    return a


def write_artifact(pipeline, features, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(export_pipeline(pipeline, features), f, indent=2, allow_nan=False)
        f.write("\n")
