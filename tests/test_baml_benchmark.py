"""Smoke tests for the issue #10 BAML benchmark runner: shared #9 schema,
telemetry on/off behavior (off retains zero logs, on retains named-stage
function logs, nothing is exported by the pinned runtime), parity between
telemetry configurations, #8 parity vs sklearn, and staged-vs-monolithic
pipeline equality. No performance targets are imposed.
"""

import json

import numpy as np
import pytest

from benchmarks import baml_benchmark as bb
from benchmarks.baml_benchmark import make_caller
from benchmarks.datasets import MISSING_FRAC, load_split
from benchmarks.reference import (
    fit_baml_pipeline,
    make_sklearn_pipeline,
    predict_baml,
    staged_fit,
    staged_predict,
)

RESULT_FIELDS = {
    "dataset",
    "task",
    "missing_frac",
    "variant",
    "operation",
    "batch_size",
    "repetitions",
    "warmups",
    "timing_s",
    "throughput_rows_per_s",
    "runtime_only_s",
    "trace",
}


def _run(tmp_path, datasets=("iris",), missing_frac=0.0):
    out = tmp_path / "results" / "baml_benchmark.json"
    doc = bb.main(
        [
            "--datasets",
            *datasets,
            "--repetitions",
            "2",
            "--fit-repetitions",
            "1",
            "--warmups",
            "0",
            "--missing-frac",
            str(missing_frac),
            "--out",
            str(out),
        ]
    )
    on_disk = json.loads(out.read_text())
    assert on_disk == doc
    return doc


@pytest.fixture(scope="module")
def doc(tmp_path_factory):
    # One small benchmark run (iris, missing-value variant) shared by the
    # read-only tests; keeps the BAML fits out of every test.
    return _run(tmp_path_factory.mktemp("bench"), missing_frac=MISSING_FRAC)


def test_output_schema_fields_and_units(doc):
    assert doc["schema_version"] == 1
    assert doc["config"]["seed"] == 42
    assert doc["config"]["timer"] == "time.perf_counter"
    assert doc["config"]["timer_unit"] == "s"
    assert doc["telemetry"]["variants"]["baml-telemetry-on"]
    assert doc["limitations"], "runtime limitations must be documented, not hidden"
    assert doc["environment"]["packages"]["baml-py"] == "0.226.3"
    assert doc["environment"]["threadpool"]

    for r in doc["results"]:
        assert RESULT_FIELDS <= set(r), r
        assert r["variant"] in ("baml-telemetry-off", "baml-telemetry-on")
        assert r["operation"] in ("bridge_probe", "fit", "predict")
        for k in ("p50", "p99", "mean", "min", "max"):
            assert r["timing_s"][k] >= 0.0  # seconds
        assert r["runtime_only_s"] is None  # duration_ms is None for expr functions -> null, not fabricated
        if r["operation"] in ("fit", "bridge_probe"):
            assert r["batch_size"] is None
            assert r["throughput_rows_per_s"] is None
        else:
            assert r["throughput_rows_per_s"] > 0.0  # rows per second
        assert r["trace"]["runtime_spans_exported"] == 0  # pinned runtime never exports expr spans


def test_operations_cover_bridge_fit_batch_and_single_row(doc):
    ops = {(r["operation"], r["batch_size"]) for r in doc["results"] if r["variant"] == "baml-telemetry-on"}
    assert ("bridge_probe", None) in ops
    assert ("fit", None) in ops
    assert ("predict", 30) in ops  # iris 80/20 split -> 30 test rows
    assert ("predict", 1) in ops


def test_telemetry_off_retains_nothing_telemetry_on_retains_named_stages(doc):
    off = [r for r in doc["results"] if r["variant"] == "baml-telemetry-off"]
    on = [r for r in doc["results"] if r["variant"] == "baml-telemetry-on"]
    assert off and all(r["trace"]["spans"] == 0 and r["trace"]["bytes"] == 0 for r in off)
    assert on and all(r["trace"]["spans"] > 0 and r["trace"]["bytes"] > 0 for r in on)
    fit_on = next(r for r in on if r["operation"] == "fit")
    names = fit_on["trace"]["function_log_names"]
    for expected in ("ImputerFit", "ImputerTransform", "ScalerFit", "ScalerTransform", "LogRegFit"):
        assert expected in names, names
    pred_on = next(r for r in on if r["operation"] == "predict" and r["batch_size"] == 30)
    assert "LogRegPredictProba" in pred_on["trace"]["function_log_names"]


def test_predictions_match_between_telemetry_configs_and_vs_sklearn(doc):
    assert len(doc["parity"]) == 1
    p = doc["parity"][0]
    assert p["predictions_match"] is True
    assert p["max_abs_diff"] == 0.0
    assert p["matches_sklearn_reference"] is True  # #8 parity, proba tol 1e-3
    assert p["sklearn_max_abs_diff"] <= 1e-3


def test_staged_execution_equals_monolithic_pipeline_and_sklearn():
    split = load_split("diabetes", missing_frac=MISSING_FRAC)
    plain = make_caller("baml-telemetry-off")[0]
    state = staged_fit(split, plain)
    pred_staged, _ = staged_predict(split, state, split.x_test, plain)
    pipeline = fit_baml_pipeline(split)
    pred_mono, _ = predict_baml(split, pipeline, split.x_test)
    assert np.array_equal(pred_staged, pred_mono)  # staged == pipeline entry point
    ref = make_sklearn_pipeline(split.task).fit(split.x_train, split.y_train)
    assert np.max(np.abs(pred_staged - ref.predict(split.x_test))) <= 1e-9  # #8 regression tolerance


def test_telemetry_on_actually_collects_logs(doc):
    on_fit = next(r for r in doc["results"] if r["variant"] == "baml-telemetry-on" and r["operation"] == "fit")
    off_fit = next(r for r in doc["results"] if r["variant"] == "baml-telemetry-off" and r["operation"] == "fit")
    assert on_fit["trace"]["spans"] == 5  # 5 staged fit calls, one log each
    assert off_fit["trace"]["spans"] == 0
    assert on_fit["trace"]["bytes"] > off_fit["trace"]["bytes"] == 0
