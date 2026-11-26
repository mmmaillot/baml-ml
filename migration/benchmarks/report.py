"""Build the comparison from measured JSON, never from placeholder values."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    docs = [json.loads((root / p).read_text()) for p in
            ("legacy-baml.json", "sklearn.json", "new-baml.json")]
    old, sk, new = docs
    baseline025 = json.loads((root / "baml-025.json").read_text())
    assert baseline025["environment"]["packages"]["baml_bridge"] == "0.25.0"
    assert new["environment"]["packages"]["baml_bridge"] == "0.26.0"
    assert baseline025["config"] == new["config"]
    assert all(p["matches_sklearn_reference"] for p in baseline025["parity"])
    for doc in docs:
        assert doc["config"]["seed"] == new["config"]["seed"]
        assert doc["config"]["test_size"] == new["config"]["test_size"]
        assert doc["config"]["missing_frac"] == new["config"]["missing_frac"]
        assert doc["config"]["warmups"] == new["config"]["warmups"]
    assert old["config"]["fit_repetitions"] == new["config"]["fit_repetitions"]
    assert old["config"]["repetitions"] == new["config"]["repetitions"] == sk["config"]["repetitions"]
    assert all(p["matches_sklearn_reference"] and p["predictions_match"] for p in old["parity"])
    assert all(p["predictions_match"] for p in sk["parity"])
    assert all(p["matches_sklearn_reference"] for p in new["parity"])
    index = {(r["dataset"], r["operation"], r["batch_size"], r["variant"]): r
             for d in docs for r in d["results"]}
    lines = ["# BAML migration benchmark", "",
             "Same GitHub Actions runner, separate SDK processes (legacy, 0.25.0, 0.26.0); seed 42, clean data, "
             "one warmup (single-row inference warmed by batch). Historical protocol: "
             "3 BAML fit repetitions, 10 sklearn fit repetitions, 10 predict/probe repetitions. "
             "No performance threshold.", "",
             "| Dataset | Operation | Batch | Variant | p50 ms | p99 ms | Rows/s | Trace spans | Trace bytes |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    def fmt(x):
        return "N/A" if x is None else f"{x:.4g}"
    for d in docs:
        for r in d["results"]:
            t = r["timing_s"]
            lines.append("| " + " | ".join(map(str, (r["dataset"], r["operation"], r["batch_size"],
                         r["variant"], fmt(t["p50"]*1000), fmt(t["p99"]*1000),
                         fmt(r["throughput_rows_per_s"]), fmt(r["trace"]["spans"]),
                         fmt(r["trace"]["bytes"])))) + " |")
    lines += ["", "## New runtime relative to historical BAML (no collector)", "",
              "| Dataset | Operation | Batch | Old / new p50 | New / sklearn p50 |",
              "|---|---|---:|---:|---:|"]
    for r in new["results"]:
        key = (r["dataset"], r["operation"], r["batch_size"])
        baseline = index[(*key, "baml-telemetry-off")]
        reference = index.get((*key, "sklearn"))
        assert r["repetitions"] == baseline["repetitions"] and r["warmups"] == baseline["warmups"]
        t = r["timing_s"]["p50"]
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | "
                     f"{fmt(baseline['timing_s']['p50']/t)}x | "
                     f"{fmt(t/reference['timing_s']['p50']) + 'x' if reference else 'N/A'} |")
    lines += ["", "## BAML 0.25.0 versus 0.26.0 (same runner)", "",
              "| Dataset | Operation | Batch | 0.25.0 p50 ms | 0.26.0 p50 ms | Speedup | 0.25.0 p99 ms | 0.26.0 p99 ms |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    prior = {(r["dataset"], r["operation"], r["batch_size"]): r for r in baseline025["results"]}
    for r in new["results"]:
        key = (r["dataset"], r["operation"], r["batch_size"])
        p = prior[key]
        a, b = p["timing_s"], r["timing_s"]
        assert p["repetitions"] == r["repetitions"] and p["warmups"] == r["warmups"]
        lines.append("| " + " | ".join(map(str, (*key, fmt(a["p50"]*1000),
                     fmt(b["p50"]*1000), fmt(a["p50"]/b["p50"])+"x",
                     fmt(a["p99"]*1000), fmt(b["p99"]*1000)))) + " |")
    lines += ["", "## Numerical parity", "",
              "| Dataset | Historical max error vs sklearn | New max error vs sklearn | New single/batch max error |",
              "|---|---:|---:|---:|"]
    for p in new["parity"]:
        prev = next(x for x in old["parity"] if x["dataset"] == p["dataset"])
        lines.append(f"| {p['dataset']} | {prev['sklearn_max_abs_diff']:.3e} | "
                     f"{p['sklearn_max_abs_diff']:.3e} | {p['single_batch_max_abs_diff']:.3e} |")
    lines += ["", "## Limits", "",
              "- New SDK trace capture, runtime-only timings and Rust SDK are not measured; null is not zero. BAML_TELEMETRY=off is documented to disable recording in 0.26.0; identical effective trace settings on 0.25.0 are not independently verified.",
              "- Legacy Collector and OTel payloads are not equivalent.",
              "- End-to-end wall time includes marshaling; bridge probe is only a round-trip proxy.",
              "- Different solver implementations (BAML versus sklearn), tiny datasets, sequential runs, "
              "shared CI noise and low fit sample counts limit conclusions. p99 fit is descriptive.",
              "- Old/new BAML Python SDKs are isolated; results include runtime and SDK changes together.",
              "- No optimization or production-readiness claim. Numerical parity is required for migration; "
              "native observability and scaling remain separate work.", ""]
    report = "\n".join(lines)
    (root / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
