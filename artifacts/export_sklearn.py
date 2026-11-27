"""Export only the deliberately narrow, dense, float64 sklearn v1 contract."""
import hashlib
import json
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.utils.validation import check_is_fitted
from sklearn.tree import DecisionTreeClassifier


def export_pipeline(pipeline, features):
    if type(pipeline) is not Pipeline or len(pipeline.steps) != 3:
        raise ValueError("expected exactly imputer, scaler, estimator")
    imp, scale, model = [step for _, step in pipeline.steps]
    if type(imp) is not SimpleImputer or imp.strategy != "mean" or imp.add_indicator or not np.isscalar(imp.missing_values) or not np.isnan(imp.missing_values):
        raise ValueError("unsupported imputer")
    if type(scale) is not StandardScaler or not scale.with_mean or not scale.with_std:
        raise ValueError("unsupported scaler")
    if type(model) not in (LinearRegression, LogisticRegression, DecisionTreeClassifier):
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
    def finite(value, shape):
        a = np.asarray(value)
        if a.shape != shape or a.dtype != np.float64 or not np.isfinite(a).all():
            raise ValueError("expected finite float64 parameters with exact dimensions")
        return a.tolist()
    means = finite(imp.statistics_, (n,))
    sm = finite(scale.mean_, (n,))
    ss = finite(scale.scale_, (n,))
    if min(ss) <= 0:
        raise ValueError("nonpositive scale")
    classes = []
    if type(model) is LinearRegression:
        kind = "linear"
        weights = [finite(model.coef_, (n,))]
        intercepts = [finite(np.asarray(model.intercept_), ())]
    elif type(model) is DecisionTreeClassifier:
        if model.n_outputs_ != 1:
            raise ValueError("only single-output classification trees supported")
        labels = np.asarray(model.classes_)
        if labels.ndim != 1 or labels.dtype.kind not in "iu" or not len(labels):
            raise ValueError("only integer classes supported")
        classes = [int(x) for x in labels]
        if any(x < -(2**63) or x >= 2**63 for x in classes) or classes != sorted(set(classes)):
            raise ValueError("invalid class order/range")
        t = model.tree_
        nodes = []
        for i in range(t.node_count):
            left, right = int(t.children_left[i]), int(t.children_right[i])
            leaf = left == right == -1
            probs = finite(t.value[i, 0], (len(classes),)) if leaf else []
            if leaf and (min(probs) < 0 or max(probs) > 1 or abs(sum(probs)-1) > 1e-12):
                raise ValueError("unsupported leaf probabilities")
            threshold = float(t.threshold[i])
            if not np.isfinite(threshold):
                raise ValueError("nonfinite split threshold")
            nodes.append(dict(left=left, right=right, feature=int(t.feature[i]), threshold=threshold, probabilities=probs))
        a = dict(schema_version=2, source="scikit-learn", source_version=sklearn.__version__,
                 features=features, imputer_kind="mean", scaler_kind="standard",
                 estimator_kind="decision_tree_classifier", imputer_means=means, scaler_means=sm,
                 scaler_scales=ss, classes=classes, nodes=nodes)
        a["model_id"] = hashlib.sha256(json.dumps(a, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        return a
    else:
        if model.solver == "liblinear" or getattr(model, "multi_class", None) == "ovr":
            raise ValueError("unsupported OvR/logistic solver")
        labels = np.asarray(model.classes_)
        if labels.ndim != 1 or labels.dtype.kind not in "iu" or len(labels) < 2:
            raise ValueError("only integer classes supported")
        classes = [int(x) for x in labels]
        if any(x < -(2**63) or x >= 2**63 for x in classes) or classes != sorted(set(classes)):
            raise ValueError("invalid class order/range")
        if np.any(np.asarray(model.n_iter_) >= model.max_iter):
            raise ValueError("fit may not have converged")
        kind = "logistic_binary" if len(classes) == 2 else "logistic_multinomial"
        k = 1 if len(classes) == 2 else len(classes)
        weights = finite(model.coef_, (k, n))
        intercepts = finite(model.intercept_, (k,))
        if len(classes) == 2 and getattr(model, "multi_class", None) == "multinomial":
            raise ValueError("legacy binary multinomial semantics unsupported")
    a = dict(schema_version=1, source="scikit-learn", source_version=sklearn.__version__,
             features=features, imputer_kind="mean", scaler_kind="standard",
             estimator_kind=kind, imputer_means=means, scaler_means=sm,
             scaler_scales=ss, weights=weights, intercepts=intercepts, classes=classes)
    a["model_id"] = hashlib.sha256(json.dumps(a, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return a


def write_artifact(pipeline, features, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(export_pipeline(pipeline, features), f, indent=2, allow_nan=False)
        f.write("\n")
