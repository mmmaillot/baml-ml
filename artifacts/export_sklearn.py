"""Export only the deliberately narrow, dense, float64 sklearn v1/v2/v3 contract."""
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


def _finite(value, shape):
    a = np.asarray(value)
    if a.shape != shape or a.dtype != np.float64 or not np.isfinite(a).all():
        raise ValueError("expected finite float64 parameters with exact dimensions")
    return a.tolist()


def _integer_classes(model, min_count=1):
    labels = np.asarray(model.classes_)
    if labels.ndim != 1 or labels.dtype.kind not in "iu" or len(labels) < min_count:
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


def _check_features(features):
    features = list(features)
    n = len(features)
    if not n or any(not isinstance(f, str) or not f for f in features) or len(set(features)) != n:
        raise ValueError("invalid feature names")
    return features, n


def _check_pipeline_names(pipeline, features):
    if hasattr(pipeline, "feature_names_in_") and list(pipeline.feature_names_in_) != features:
        raise ValueError("feature order differs from fitted pipeline")


def _export_linear(model, n):
    if np.asarray(model.coef_).ndim != 1:
        raise ValueError("only single-output linear regression supported")
    return dict(
        kind="linear",
        n_features_in=n,
        weights=[_finite(model.coef_, (n,))],
        intercepts=[_finite(np.asarray(model.intercept_, dtype=np.float64), ())],
        classes=[],
        nodes=[],
    )


def _export_logistic(model, n):
    if model.solver == "liblinear" or getattr(model, "multi_class", None) == "ovr":
        raise ValueError("unsupported OvR/logistic solver")
    classes = _integer_classes(model, min_count=2)
    if np.any(np.asarray(model.n_iter_) >= model.max_iter):
        raise ValueError("fit may not have converged")
    kind = "logistic_binary" if len(classes) == 2 else "logistic_multinomial"
    k = 1 if len(classes) == 2 else len(classes)
    if len(classes) == 2 and getattr(model, "multi_class", None) == "multinomial":
        raise ValueError("legacy binary multinomial semantics unsupported")
    return dict(
        kind=kind,
        n_features_in=n,
        weights=_finite(model.coef_, (k, n)),
        intercepts=_finite(model.intercept_, (k,)),
        classes=classes,
        nodes=[],
    )


def _export_decision_tree(model, n):
    if model.n_outputs_ != 1:
        raise ValueError("only single-output classification trees supported")
    classes = _integer_classes(model)
    return dict(
        kind="decision_tree_classifier",
        n_features_in=n,
        weights=[],
        intercepts=[],
        classes=classes,
        nodes=_tree_nodes(model.tree_, len(classes)),
    )


def export_estimator(model, n):
    """Export the terminal estimator of a schema v3 pipeline."""
    if model.n_features_in_ != n:
        raise ValueError("unsupported dimensionality change")
    if type(model) is LinearRegression:
        return _export_linear(model, n)
    if type(model) is LogisticRegression:
        return _export_logistic(model, n)
    if type(model) is DecisionTreeClassifier:
        return _export_decision_tree(model, n)
    raise ValueError("unsupported estimator")


def _export_simple_imputer(step, width, col_kinds):
    if step.strategy not in ("mean", "median", "most_frequent", "constant"):
        raise ValueError("unsupported imputer strategy")
    if step.add_indicator:
        raise ValueError("unsupported imputer indicators")
    if not np.isscalar(step.missing_values) or not np.isnan(step.missing_values):
        raise ValueError("unsupported imputer missing values")
    if step.n_features_in_ != width:
        raise ValueError("unsupported dimensionality change")
    stats = np.asarray(step.statistics_)
    if stats.shape != (width,):
        raise ValueError("unsupported imputer statistics shape")
    if stats.dtype == np.float64:
        if step.strategy in ("mean", "median", "most_frequent", "constant") and not np.isfinite(stats).all():
            raise ValueError("non-finite imputer statistics")
        if any(k != "numeric" for k in col_kinds):
            raise ValueError("numeric imputer on categorical input")
        return (
            dict(kind="simple_imputer", n_features_in=width, n_features_out=width,
                 imputer=dict(strategy=step.strategy, value_kind="numeric",
                              statistics=stats.tolist(), fill_strings=[])),
            width,
            list(col_kinds),
        )
    if stats.dtype.kind in "OUS":
        if step.strategy in ("mean", "median"):
            raise ValueError("unsupported imputer strategy for string statistics")
        fills = [s if isinstance(s, str) else None for s in stats.tolist()]
        if any(f is None or f == "" for f in fills):
            raise ValueError("unsupported imputer fill value")
        if any(k != "categorical" for k in col_kinds):
            raise ValueError("string imputer on numeric input")
        return (
            dict(kind="simple_imputer", n_features_in=width, n_features_out=width,
                 imputer=dict(strategy=step.strategy, value_kind="string",
                              statistics=[], fill_strings=fills)),
            width,
            list(col_kinds),
        )
    raise ValueError("unsupported imputer statistics dtype")


def _export_standard_scaler(step, width, col_kinds):
    if not step.with_mean or not step.with_std:
        raise ValueError("unsupported scaler")
    if step.n_features_in_ != width:
        raise ValueError("unsupported dimensionality change")
    sm = _finite(step.mean_, (width,))
    ss = _finite(step.scale_, (width,))
    if min(ss) <= 0:
        raise ValueError("nonpositive scale")
    if any(k != "numeric" for k in col_kinds):
        raise ValueError("scaler on categorical input")
    return (
        dict(kind="standard_scaler", n_features_in=width, n_features_out=width,
             scaler=dict(means=sm, scales=ss)),
        width,
        list(col_kinds),
    )


def export_preprocessor(step, width, col_kinds):
    """Export one preprocessing step of a schema v3 pipeline.

    Returns (step_dict, n_features_out, output_column_kinds).
    """
    if type(step) is SimpleImputer:
        return _export_simple_imputer(step, width, col_kinds)
    if type(step) is StandardScaler:
        return _export_standard_scaler(step, width, col_kinds)
    raise ValueError("unsupported pipeline step")


def _infer_input_kinds(pre, n):
    if not pre:
        return ["numeric"] * n
    first = pre[0]
    if type(first) is SimpleImputer and np.asarray(first.statistics_).dtype.kind in "OUS":
        return ["categorical"] * n
    return ["numeric"] * n


def export_pipeline_v3(pipeline, features, input_kinds=None):
    """Export an arbitrary-length sklearn Pipeline into the schema v3 contract.

    The last step must be a supported estimator; all preceding steps must be
    supported dense preprocessors. Dimensionality is validated along the
    whole chain and unsupported variants are rejected, never approximated.
    """
    if type(pipeline) is not Pipeline:
        raise ValueError("expected a sklearn Pipeline")
    steps = [step for _, step in pipeline.steps]
    if not steps:
        raise ValueError("empty pipeline")
    pre, est = steps[:-1], steps[-1]
    for step in steps:
        check_is_fitted(step)
    features, n = _check_features(features)
    _check_pipeline_names(pipeline, features)
    if input_kinds is None:
        input_kinds = _infer_input_kinds(pre, n)
    input_kinds = list(input_kinds)
    if len(input_kinds) != n or any(k not in ("numeric", "categorical") for k in input_kinds):
        raise ValueError("invalid input kinds")
    width = n
    col_kinds = list(input_kinds)
    step_dicts = []
    for step in pre:
        d, width, col_kinds = export_preprocessor(step, width, col_kinds)
        step_dicts.append(d)
    estimator = export_estimator(est, width)
    a = dict(schema_version=3, source="scikit-learn", source_version=sklearn.__version__,
             features=features, input_kinds=input_kinds, steps=step_dicts, estimator=estimator)
    a["model_id"] = _model_id(a)
    return a


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
    features, n = _check_features(features)
    _check_pipeline_names(pipeline, features)
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
    else:
        est = _export_logistic(model, n)
        kind = est["kind"]
        classes = est["classes"]
        weights = est["weights"]
        intercepts = est["intercepts"]
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


def write_artifact_v3(pipeline, features, path, input_kinds=None):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(export_pipeline_v3(pipeline, features, input_kinds), f, indent=2, allow_nan=False)
        f.write("\n")
