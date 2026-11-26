"""Smoke tests for the issue #9 sklearn benchmark runner: output schema,
timing units, trace volume rules (vanilla emits nothing, OTel emits named
pipeline-stage spans), and instrumented/uninstrumented prediction parity.
No performance targets are imposed.
"""

import json

import numpy as np

from benchmarks import sklearn_benchmark as sb
from benchmarks.datasets import SEED

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
    "trace",
}


def _run(tmp_path, datasets=("iris", "diabetes")):
    out = tmp_path / "results" / "sklearn_benchmark.json"
    doc = sb.main(
        [
            "--datasets",
            *datasets,
            "--repetitions",
            "2",
            "--warmups",
            "1",
            "--out",
            str(out),
        ]
    )
    on_disk = json.loads(out.read_text())
    assert on_disk == doc  # what is written is what was returned
    return doc


def test_output_schema_fields_and_units(tmp_path):
    doc = _run(tmp_path)
    assert doc["schema_version"] == 1
    assert doc["config"]["seed"] == SEED
    assert doc["config"]["timer"] == "time.perf_counter"
    assert doc["config"]["timer_unit"] == "s"
    assert doc["config"]["repetitions"] == 2
    assert doc["config"]["warmups"] == 1
    assert doc["telemetry"]["exporter"]
    assert doc["telemetry"]["granularity"]
    assert doc["environment"]["packages"]["scikit-learn"] == "1.9.1"
    assert doc["environment"]["packages"]["opentelemetry-sdk"]
    assert doc["environment"]["python"]
    assert doc["environment"]["threadpool"]

    results = doc["results"]
    assert results
    for r in results:
        assert RESULT_FIELDS <= set(r), r
        assert r["variant"] in ("sklearn", "sklearn-otel")
        assert r["operation"] in ("fit", "predict")
        for k in ("p50", "p99", "mean", "min", "max"):
            assert r["timing_s"][k] >= 0.0  # seconds
        if r["operation"] == "fit":
            assert r["batch_size"] is None
            assert r["throughput_rows_per_s"] is None  # not applicable -> null
        else:
            assert isinstance(r["batch_size"], int) and r["batch_size"] >= 1
            assert r["throughput_rows_per_s"] > 0.0  # rows per second
        assert set(r["trace"]) == {"spans", "bytes"}


def test_measured_operations_cover_fit_batch_and_single_row(tmp_path):
    doc = _run(tmp_path, datasets=("iris",))
    ops = {(r["operation"], r["batch_size"]) for r in doc["results"] if r["variant"] == "sklearn-otel"}
    assert ("fit", None) in ops
    assert ("predict", 30) in ops  # iris 80/20 split -> 30 test rows
    assert ("predict", 1) in ops  # single-row inference


def test_vanilla_emits_no_traces_otel_emits_named_stages(tmp_path):
    doc = _run(tmp_path, datasets=("iris",))
    vanilla = [r for r in doc["results"] if r["variant"] == "sklearn"]
    otel = [r for r in doc["results"] if r["variant"] == "sklearn-otel"]
    assert vanilla and all(r["trace"] == {"spans": 0, "bytes": 0} for r in vanilla)
    assert otel and all(r["trace"]["spans"] > 0 and r["trace"]["bytes"] > 0 for r in otel)
    sample = doc["trace_sample"]
    assert "pipeline.fit" in sample
    assert "pipeline.predict_proba" in sample  # iris is classification
    for stage in ("imputer", "scaler", "clf"):
        assert any(name.startswith(f"pipeline.stage.{stage}.") for name in sample), stage


def test_instrumented_and_uninstrumented_predictions_match(tmp_path):
    doc = _run(tmp_path)
    assert len(doc["parity"]) == 2  # iris and diabetes
    for p in doc["parity"]:
        assert p["variants"] == ["sklearn", "sklearn-otel"]
        assert p["predictions_match"]
        assert p["max_abs_diff"] == 0.0


def test_missing_frac_variant_is_recorded(tmp_path):
    out = tmp_path / "missing.json"
    doc = sb.main(["--datasets", "iris", "--repetitions", "1", "--warmups", "0",
                   "--missing-frac", "0.10", "--out", str(out)])
    assert doc["config"]["missing_frac"] == 0.10
    assert all(r["missing_frac"] == 0.10 for r in doc["results"])
    assert doc["parity"][0]["predictions_match"]


def test_no_fabricated_nulls_in_timing(tmp_path):
    doc = _run(tmp_path, datasets=("iris",))
    for r in doc["results"]:
        assert all(v is not None for v in r["timing_s"].values())
        assert np.isfinite(list(r["timing_s"].values())).all()
