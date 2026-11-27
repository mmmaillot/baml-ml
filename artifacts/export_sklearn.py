"""Export only the deliberately narrow, dense, float64 sklearn v1/v2/v3 contract."""
import hashlib
import json
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
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
    vals = stats.tolist()
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
        arr = np.asarray(vals, dtype=np.float64)
        if not np.isfinite(arr).all():
            raise ValueError("non-finite imputer statistics")
        if any(k != "numeric" for k in col_kinds):
            raise ValueError("numeric imputer on categorical input")
        return (
            dict(kind="simple_imputer", n_features_in=width, n_features_out=width,
                 imputer=dict(strategy=step.strategy, value_kind="numeric",
                              statistics=arr.tolist(), fill_strings=[])),
            width,
            list(col_kinds),
        )
    if all(isinstance(v, str) for v in vals):
        if step.strategy in ("mean", "median"):
            raise ValueError("unsupported imputer strategy for string statistics")
        if any(v == "" for v in vals):
            raise ValueError("unsupported imputer fill value")
        if any(k != "categorical" for k in col_kinds):
            raise ValueError("string imputer on numeric input")
        return (
            dict(kind="simple_imputer", n_features_in=width, n_features_out=width,
                 imputer=dict(strategy=step.strategy, value_kind="string",
                              statistics=[], fill_strings=list(vals))),
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


def _export_one_hot_encoder(step, width, col_kinds):
    if step.n_features_in_ != width:
        raise ValueError("unsupported dimensionality change")
    if getattr(step, "sparse_output", getattr(step, "sparse", False)):
        raise ValueError("sparse one-hot output unsupported")
    if step.handle_unknown not in ("error", "ignore"):
        raise ValueError("unsupported one-hot handle_unknown")
    if step.drop not in (None, "first", "if_binary"):
        raise ValueError("unsupported one-hot drop")
    if getattr(step, "min_frequency", None) is not None or getattr(step, "max_categories", None) is not None:
        raise ValueError("infrequent category grouping unsupported")
    if getattr(step, "infrequent_categories", None) is not None:
        raise ValueError("infrequent category mapping unsupported")
    if step.dtype != np.float64:
        raise ValueError("unsupported one-hot dtype")
    if any(k != "categorical" for k in col_kinds):
        raise ValueError("one-hot encoder on numeric input")
    columns = []
    n_out = 0
    for cats in step.categories_:
        cats = np.asarray(cats)
        if cats.ndim != 1 or len(cats) == 0:
            raise ValueError("invalid fitted categories")
        if cats.dtype.kind in "iu":
            if np.any(np.abs(cats.astype(np.int64)) >= 2**53):
                raise ValueError("integer category outside exact float64 range")
            columns.append(dict(kind="integer", strings=[], ints=[float(v) for v in cats.tolist()]))
        elif cats.dtype.kind in "OUS":
            vals = cats.tolist()
            if any(not isinstance(v, str) for v in vals):
                raise ValueError("non-string category in string column")
            columns.append(dict(kind="string", strings=list(vals), ints=[]))
        else:
            raise ValueError("unsupported category dtype")
        n = len(cats)
        adj = 1 if (step.drop == "first" or (step.drop == "if_binary" and n == 2)) else 0
        n_out += n - adj
    fitted_out = getattr(step, "n_features_out_", None)
    if fitted_out is None:
        fitted_out = sum(int(v) for v in getattr(step, "_n_features_outs", [n_out]))
    if n_out < width or n_out != fitted_out:
        raise ValueError("one-hot output dimension mismatch")
    return (
        dict(kind="one_hot_encoder", n_features_in=width, n_features_out=n_out,
             one_hot=dict(columns=columns, handle_unknown=step.handle_unknown,
                          drop=step.drop)),
        n_out,
        ["numeric"] * n_out,
    )


def _resolve_columns(cols, width, features):
    """Resolve explicit integer or string column selectors to indices."""
    if isinstance(cols, (list, tuple, np.ndarray)) and not callable(cols):
        arr = list(cols)
        if arr and all(isinstance(c, (int, np.integer)) and not isinstance(c, bool) for c in arr):
            out = [int(c) for c in arr]
        elif arr and all(isinstance(c, str) for c in arr):
            if any(c not in features for c in arr):
                raise ValueError("unknown column name selector")
            out = [features.index(c) for c in arr]
        else:
            raise ValueError("unsupported column selector")
        if any(c < 0 or c >= width for c in out):
            raise ValueError("column index out of range")
        return out
    raise ValueError("runtime column selectors unsupported")


def _export_branch_pipeline(trans, cols_r, col_kinds):
    if type(trans) is ColumnTransformer:
        raise ValueError("nested column transformer unsupported")
    if not isinstance(trans, Pipeline):
        raise ValueError("unsupported branch transformer")
    sub = [s for _, s in trans.steps]
    for s in sub:
        check_is_fitted(s)
    sub_width = len(cols_r)
    sub_kinds = [col_kinds[c] for c in cols_r]
    step_dicts = []
    for s in sub:
        d, sub_width, sub_kinds = export_preprocessor(s, sub_width, sub_kinds, None)
        step_dicts.append(d)
    if any(k != "numeric" for k in sub_kinds):
        raise ValueError("branch pipeline output is not numeric")
    return step_dicts, sub_width


def _export_column_transformer(step, width, col_kinds, features):
    if step.transformer_weights is not None:
        raise ValueError("transformer_weights unsupported")
    if getattr(step, "sparse_output_", False):
        raise ValueError("sparse column transformer output unsupported")
    if step.n_features_in_ != width:
        raise ValueError("unsupported dimensionality change")
    for _name, _trans, cols in step.transformers:
        if callable(cols) or callable(_trans):
            raise ValueError("callable selectors/transformers unsupported")
    branches = []
    used = set()
    fitted_remainder = None
    for name, trans, cols in step.transformers_:
        if name == "remainder":
            fitted_remainder = _resolve_columns(cols, width, features)
            continue
        cols_r = _resolve_columns(cols, width, features)
        if any(c in used for c in cols_r):
            raise ValueError("overlapping branch columns")
        used.update(cols_r)
        if not isinstance(name, str) or not name:
            raise ValueError("invalid branch name")
        if trans == "drop":
            branches.append(dict(name=name, columns=cols_r, mode="drop", steps=[]))
        elif trans == "passthrough":
            if any(col_kinds[c] != "numeric" for c in cols_r):
                raise ValueError("passthrough branch on categorical input")
            branches.append(dict(name=name, columns=cols_r, mode="passthrough", steps=[]))
        else:
            step_dicts, branch_out = _export_branch_pipeline(trans, cols_r, col_kinds)
            branches.append(dict(name=name, columns=cols_r, mode="pipeline", steps=step_dicts))
    if len({b["name"] for b in branches}) != len(branches):
        raise ValueError("duplicate branch names")
    remainder = step.remainder
    if remainder not in ("drop", "passthrough"):
        raise ValueError("unsupported remainder")
    rem_cols = sorted(set(range(width)) - used)
    if fitted_remainder is not None and fitted_remainder != rem_cols:
        raise ValueError("remainder columns differ from the fitted complement")
    if remainder == "passthrough" and any(col_kinds[c] != "numeric" for c in rem_cols):
        raise ValueError("passthrough remainder on categorical input")
    n_out = sum(b["steps"][-1]["n_features_out"] if b["mode"] == "pipeline"
                else (len(b["columns"]) if b["mode"] == "passthrough" else 0)
                for b in branches)
    if remainder == "passthrough":
        n_out += len(rem_cols)
    if n_out != len(step.get_feature_names_out()):
        raise ValueError("column transformer output dimension mismatch")
    return (
        dict(kind="column_transformer", n_features_in=width, n_features_out=n_out,
             column_transformer=dict(branches=branches, remainder=remainder, remainder_columns=rem_cols)),
        n_out,
        ["numeric"] * n_out,
    )


def export_preprocessor(step, width, col_kinds, features=None):
    """Export one preprocessing step of a schema v3 pipeline.

    Returns (step_dict, n_features_out, output_column_kinds).
    """
    if type(step) is SimpleImputer:
        return _export_simple_imputer(step, width, col_kinds)
    if type(step) is StandardScaler:
        return _export_standard_scaler(step, width, col_kinds)
    if type(step) is OneHotEncoder:
        return _export_one_hot_encoder(step, width, col_kinds)
    if type(step) is ColumnTransformer:
        return _export_column_transformer(step, width, col_kinds, features or list(range(width)))
    raise ValueError("unsupported pipeline step")


def _infer_input_kinds(pre, n, features):
    if not pre:
        return ["numeric"] * n
    first = pre[0]
    if type(first) is SimpleImputer:
        vals = np.asarray(first.statistics_).tolist()
        if all(isinstance(v, str) for v in vals):
            return ["categorical"] * n
        return ["numeric"] * n
    if type(first) is OneHotEncoder:
        return ["categorical"] * n
    if type(first) is ColumnTransformer:
        kinds = ["numeric"] * n
        for name, trans, cols in first.transformers_:
            if name == "remainder":
                continue
            cols_r = _resolve_columns(cols, n, features)
            if isinstance(trans, Pipeline) and any(type(s) is OneHotEncoder for _, s in trans.steps):
                for c in cols_r:
                    kinds[c] = "categorical"
        return kinds
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
        input_kinds = _infer_input_kinds(pre, n, features)
    input_kinds = list(input_kinds)
    if len(input_kinds) != n or any(k not in ("numeric", "categorical") for k in input_kinds):
        raise ValueError("invalid input kinds")
    width = n
    col_kinds = list(input_kinds)
    step_dicts = []
    for step in pre:
        d, width, col_kinds = export_preprocessor(step, width, col_kinds, features)
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
