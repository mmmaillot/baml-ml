"""Tests for issue #11: published benchmark report consistency against raw JSON
artifacts and the reproducible synthetic missing-value inference trace demo.
"""

import json
from pathlib import Path

from benchmarks.run_all import render_report
from examples import trace_demo

DOCS_BENCH = Path(__file__).resolve().parent.parent / "docs" / "benchmarks"


def test_published_artifacts_cover_all_variants_and_match_report():
    sk_doc = json.loads((DOCS_BENCH / "sklearn_benchmark.json").read_text())
    baml_doc = json.loads((DOCS_BENCH / "baml_benchmark.json").read_text())
    trace_doc = json.loads((DOCS_BENCH / "inference_trace_demo.json").read_text())
    report = (DOCS_BENCH / "report.md").read_text()

    sk_variants = {r["variant"] for r in sk_doc["results"]}
    baml_variants = {r["variant"] for r in baml_doc["results"]}
    assert sk_variants == {"sklearn", "sklearn-otel"}
    assert baml_variants == {"baml-telemetry-off", "baml-telemetry-on"}

    assert {r["dataset"] for r in sk_doc["results"]} == {"iris", "diabetes"}
    assert {r["dataset"] for r in baml_doc["results"]} == {"iris", "diabetes"}
    assert all(p["predictions_match"] for p in sk_doc["parity"])
    assert all(p["predictions_match"] and p["matches_sklearn_reference"] for p in baml_doc["parity"])

    # The published report must be an exact render of the committed raw JSON files.
    assert report == render_report(sk_doc, baml_doc, trace_doc)


def test_trace_demo_reproduces_named_stages_and_diagnostic(tmp_path):
    out = tmp_path / "trace_demo.json"
    doc = trace_demo.run(out_path=out, verbose=False)
    assert json.loads(out.read_text()) == doc

    assert doc["synthetic_input"]["sepal_width_cm"] is None
    missing = doc["diagnostics"]["missing_features"]
    assert len(missing) == 1
    assert missing[0]["feature"] == "sepal_width_cm"
    assert 2.5 < missing[0]["imputed_with_train_mean"] < 3.5

    assert doc["parity"]["predicted_class_baml"] == "setosa"
    assert doc["parity"]["predicted_class_sklearn"] == "setosa"
    assert doc["parity"]["max_proba_abs_diff"] <= 1e-3

    baml_names = [lg["function_name"] for lg in doc["baml_collector_trace"]["logs"]]
    assert baml_names == ["ImputerTransform", "ScalerTransform", "LogRegPredictProba"]
    assert doc["baml_collector_trace"]["total_bytes"] > 0

    otel_names = [sp["name"] for sp in doc["sklearn_otel_trace"]["spans_detail"]]
    for expected in (
        "pipeline.stage.imputer.transform",
        "pipeline.stage.scaler.transform",
        "pipeline.stage.clf.predict_proba",
        "pipeline.predict_proba",
    ):
        assert expected in otel_names
    assert doc["sklearn_otel_trace"]["total_bytes"] > 0
