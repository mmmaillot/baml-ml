"""Assertions and forensic queries for the unmodified native runtime."""

import math
from pathlib import Path

SPAN_QUERY = """SELECT span_id, parent_span_id, span_name, status,
       input_args, output_value, error_value, context_metadata
FROM spans ORDER BY start_time"""
SPAN_FIELDS = ("id", "parent", "name", "status", "inputs", "output", "error", "metadata")
HERE = Path(__file__).resolve().parent


def spans(rows):
    return [dict(zip(SPAN_FIELDS, row, strict=True)) for row in rows]


def named(records, name):
    return [span for span in records if span["name"] == f"user.{name}"]


def one(records, name):
    match = named(records, name)
    assert len(match) == 1, (name, len(match))
    return match[0]


def tagged(span, key):
    return span["metadata"].get(f"baml_ml.{key}")


def payload_free(span):
    return all(span[field] is None for field in ("inputs", "output", "error"))


def snapshot_ids(records, expected):
    loads = named(records, expected)
    snapshots = [span for span in loads if tagged(span, "lazy_snapshot") is True]
    assert all(payload_free(span) for span in loads if span not in snapshots)
    assert all(s["inputs"]["artifact"]["model_id"] == tagged(s, "model_id") for s in snapshots)
    return [(tagged(span, "model_id"), tagged(span, "shadow_role")) for span in snapshots]


def row_reasons(records, name):
    escalations = named(records, name)
    assert all(tagged(row, "escalated") is True and payload_free(row) for row in escalations)
    return [(tagged(row, "row_index"), tagged(row, "trigger_reasons")) for row in escalations]


def verify_adaptive_batch(records, expected, model_id, operation):
    names = ["ObservedImpute", "ObservedScale", "ObservedDecisionScores",
             "ObservedLabels" if operation == "predict" else "ObservedSoftmax"]
    assert not named(records, "ObservedSoftmax" if operation == "predict" else "ObservedLabels")
    rows = named(records, "EscalatedRow")
    assert row_reasons(records, "EscalatedRow") == expected
    if expected:
        assert snapshot_ids(records, "ModelLoaded") == [(model_id, None)]
        assert len(named(records, "ModelLoaded")) == 2
    else:
        assert not snapshot_ids(records, "ModelLoaded")
        assert len(named(records, "ModelLoaded")) == 1
    main = one(records, "RunSelectiveModel")
    assert payload_free(main)
    for name in names:
        calls = named(records, name)
        assert len(calls) == 1 + len(rows), name
        assert calls[0]["parent"] == main["id"] and payload_free(calls[0])
        for stage, row in zip(calls[1:], rows, strict=True):
            assert stage["parent"] == row["id"]
            assert tagged(stage, "model_id") == model_id
            assert tagged(stage, "row_index") == tagged(row, "row_index")
            assert tagged(stage, "trigger_reasons") == tagged(row, "trigger_reasons")
            assert_redacted(stage["inputs"])
            assert stage["output"] is not None


def assert_redacted(inputs):
    if "state" in inputs:
        assert inputs["state"]["value"] == {"$redacted": "declared"}


def strip_type_info(value):
    # Retain the encoded NaN/redaction markers, omit BTEL type definitions.
    if isinstance(value, dict):
        return {key: strip_type_info(item) for key, item in value.items()
                if key not in ("$class", "$def_id", "$definition")}
    if isinstance(value, list):
        return [strip_type_info(item) for item in value]
    return value


def verify_diffs(records, sql_rows, champion, challenger, operation):
    model_ids = {"champion": champion["model_id"], "challenger": challenger["model_id"]}
    kind = champion["estimator_kind"]
    tree = kind == "decision_tree_classifier"
    root = one(records, "ShadowBatch")
    for key, value in (("champion_model_id", model_ids["champion"]),
                       ("challenger_model_id", model_ids["challenger"]),
                       ("operation", operation), ("estimator_kind", kind)):
        assert tagged(root, key) == value
    assert payload_free(root)
    parent = one(records, "ShadowEscalatedRow")
    assert parent["parent"] == root["id"]
    assert row_reasons(records, "ShadowEscalatedRow")[0][0] == 0
    assert snapshot_ids(records, "TreeModelLoaded" if tree else "ModelLoaded") == [
        (model_ids[role], role) for role in model_ids
    ]
    main_passes = [span for span in named(records, "RunSelectiveModel") if span["parent"] == root["id"]]
    assert [tagged(span, "shadow_role") for span in main_passes] == list(model_ids)
    assert all(payload_free(span) for span in main_passes)
    for main in main_passes:
        stages = [span for span in records if span["parent"] == main["id"] and span["name"].startswith("user.Observed")]
        assert stages and all(payload_free(stage) for stage in stages)

    values = {}
    for index, name, role, model_id, inputs, output in sql_rows:
        assert index == 0 and model_id == model_ids[role]
        assert_redacted(inputs)
        stage = name.removeprefix("user.")
        assert stage not in values.setdefault(role, {})
        values[role][stage] = {"inputs": strip_type_info(inputs), "output": strip_type_info(output)}

    stages = ["ObservedImpute", "ObservedScale"]
    if tree:
        stages.append("ObservedTreeTraversal")
    elif kind == "linear":
        stages.append("ObservedLinear")
    else:
        stages.extend(("ObservedDecisionScores", "ObservedSoftmax"))
    for role, expected in (("champion", 0.0), ("challenger", 2.0)):
        result = values[role]
        assert set(result) == set(stages)
        imputed = result["ObservedImpute"]
        scaled = result["ObservedScale"]
        assert imputed["inputs"]["x"] == [[{"$float": "NaN"}]]
        assert imputed["output"]["x"] == scaled["inputs"]["x"] == scaled["output"]["x"] == [[expected]]
        assert result[stages[2]]["inputs"]["x"] == [[expected]]
        if tree:
            traversal = result["ObservedTreeTraversal"]["output"]
            leaf = 1 if role == "champion" else 2
            assert traversal["paths"] == [[0, leaf]] and traversal["leaves"] == [leaf]
            assert traversal["probs"] == ([[0.5, 0.5]] if role == "champion" else [[0.0, 1.0]])
        elif kind == "linear":
            assert result["ObservedLinear"]["output"]["value"] == ([0.0] if role == "champion" else [-1.0])
        else:
            logit = -2.0 if role == "champion" else 2.0
            scores = result["ObservedDecisionScores"]["output"]
            softmax = result["ObservedSoftmax"]
            assert scores == softmax["inputs"]["scores"] == [[0.0, logit]]
            probability = 1.0 / (1.0 + math.exp(-logit))
            assert all(math.isclose(a, b, rel_tol=0, abs_tol=1e-15)
                       for a, b in zip(softmax["output"]["probs"][0], (1-probability, probability), strict=True))
    return {"row_index": 0, "traces": values}


def verify_runtime(out, execute, query, repeat_probe, fixture, write_json, predict, shadow):
    policy = write_json(out / "cap-one.json", {"max_escalated_rows": 1})
    cap_zero = write_json(out / "cap-zero.json", {"max_escalated_rows": 0})
    invalid_policy = out / "bad-policy.json"
    invalid_policy.write_text("{")
    logistic = dict(fixture("binary"), model_id="adaptive-binary", features=["x"],
                    imputer_means=[0.0], scaler_means=[0.0], scaler_scales=[1.0],
                    weights=[[2.0]], intercepts=[0.0])
    model = write_json(out / "adaptive-model.json", logistic)
    healthy = write_json(out / "adaptive-healthy.json", dict(features=["x"], rows=[[2]]))
    mixed = write_json(out / "adaptive-mixed.json", dict(features=["x"], rows=[[2], [None], [5], [0], [1]]))
    mixed_reasons = [(1, "missing_input,low_margin"), (2, "extreme_zscore"), (3, "low_margin")]
    adaptive = {}
    for name, data, overrides, expected in (
        ("healthy", healthy, None, []),
        ("mixed", mixed, None, mixed_reasons),
        ("capped", mixed, policy, mixed_reasons[:1]),
        ("zero-cap", mixed, cap_zero, []),
    ):
        result, _ = execute(f"adaptive-{name}", predict(model, data, "adaptive", overrides, "predict"))
        assert result["predictions"] == ([1] if name == "healthy" else [1, 0, 1, 0, 1])
        records = spans(query(f"adaptive-{name}"))
        verify_adaptive_batch(records, expected, logistic["model_id"], "predict")
        adaptive[name] = {"rows": row_reasons(records, "EscalatedRow"),
                          "snapshots": snapshot_ids(records, "ModelLoaded")}

    outputs, records = repeat_probe(model)
    assert [row["value"] for row in outputs] == [[1], [1, 0, 1, 0, 1], [0]]
    records = spans(records)
    assert row_reasons(records, "EscalatedRow") == [*mixed_reasons, (0, "low_margin")]
    assert snapshot_ids(records, "ModelLoaded") == [(logistic["model_id"], None)]
    assert len(named(records, "ModelLoaded")) == 2
    adaptive["repeated_handle"] = {"rows": row_reasons(records, "EscalatedRow"),
                                   "snapshots": snapshot_ids(records, "ModelLoaded")}
    _, sizes = execute("adaptive-off", predict(model, mixed, "adaptive"), telemetry="off")
    assert not any(sizes[f"{kind}_bytes"] for kind in ("btel", "cas"))
    execute("invalid-policy", predict(model, healthy, "adaptive", invalid_policy), expected_exit=1)
    execute("misapplied-policy", predict(model, healthy, "semantic", policy), expected_exit=1)

    linear = {**logistic, "estimator_kind": "linear", "classes": []}
    tree = fixture("tree-smoke")
    cases = [
        ("linear", linear, dict(linear, weights=[[-1.0]], intercepts=[1.0]), "predict"),
        ("logistic", dict(logistic, weights=[[1.0]], intercepts=[-2.0]),
         dict(logistic, weights=[[1.0]], intercepts=[0.0]), "predict-proba"),
        ("tree", tree, dict(tree, nodes=[dict(tree["nodes"][0], threshold=1.0), *tree["nodes"][1:]]),
         "predict-proba"),
    ]
    data = write_json(out / "shadow-mixed.json", dict(features=["x"], rows=[[None], [5], [1]]))
    sql = (HERE / "semantic-diff.sql").read_text()
    diffs = {}
    for name, champion, challenger, operation in cases:
        champion = dict(champion, model_id=f"{name}-champion")
        challenger = dict(challenger, model_id=f"{name}-challenger", imputer_means=[2.0],
                          scaler_means=[1.0], scaler_scales=[0.5])
        a = write_json(out / f"{name}-champion.json", champion)
        b = write_json(out / f"{name}-challenger.json", challenger)
        case = f"diff-{name}"
        result, _ = execute(case, shadow(a, b, data, policy, operation))
        baseline, _ = execute(f"champion-{name}", predict(a, data, "semantic", operation=operation))
        assert result["predictions"] == baseline["predictions"]
        assert result["summary"]["challenger_ok"] and result["summary"]["capped"]
        assert result["summary"]["escalated_rows"] == 1
        diffs[name] = verify_diffs(spans(query(case)), query(case, sql), champion, challenger, operation)
        diffs[name]["summary"] = result["summary"]

    a = write_json(out / "overflow-champion.json", dict(linear, model_id="overflow-champion"))
    b = write_json(out / "overflow-challenger.json", dict(linear, model_id="overflow-challenger", weights=[[1e308]]))
    result, _ = execute("isolated-overflow", shadow(a, b, data, cap_zero, "predict"))
    assert result["predictions"] == [0.0, 10.0, 2.0]
    summary = result["summary"]
    assert not summary["challenger_ok"] and summary["challenger_error"]["stage"] == "linear_prediction"
    records = spans(query("isolated-overflow"))
    failure = one(records, "ShadowChallengerFailed")
    assert failure["parent"] == one(records, "ShadowBatch")["id"]
    assert failure["status"] != "return" and tagged(failure, "challenger_isolated") is True
    assert failure["error"]["error"]["stage"] == summary["challenger_error"]["stage"]
    assert not named(records, "ShadowEscalatedRow")
    snapshots = snapshot_ids(records, "ModelLoaded")
    assert snapshots == [("overflow-champion", "champion"), ("overflow-challenger", "challenger")]
    execute("champion-overflow", shadow(b, a, data, policy, "predict"), expected_exit=1)
    execute("shadow-invalid-policy", shadow(a, b, data, invalid_policy, "predict"), expected_exit=1)
    execute("shadow-incompatible", shadow(a, model, data, policy, "predict"), expected_exit=1)
    failure_evidence = {"status": failure["status"], "metadata": failure["metadata"],
                        "error": summary["challenger_error"], "snapshots": snapshots}
    return {"adaptive": adaptive, "shadow": diffs, "shadow_challenger_failure": failure_evidence}
