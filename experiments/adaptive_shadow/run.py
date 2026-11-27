"""Measure the packed BAML runtime and validate selective traces with native SQL."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import file_digest
import json
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter_ns

from verify import SPAN_QUERY, verify_runtime

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIXTURES = ROOT / "artifacts" / "fixtures"
ROW_COUNT = 4096
SEED = 42
BAML_VERSION = "0.26.0"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return path


def fixture(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


def clean_env(case, telemetry):
    home = case / "home"
    home.mkdir(exist_ok=True)
    tmp = case / "tmp"
    tmp.mkdir(exist_ok=True)
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(home),
        "TMPDIR": str(tmp),
        "LC_ALL": "C",
        "DO_NOT_TRACK": "1",
        "BAML_TELEMETRY": telemetry,
    }


def command_output(*args):
    return subprocess.check_output(args, text=True).strip()


def build(cli, host, project, binary):
    source = project / "baml_src"
    source.mkdir(parents=True)
    shutil.copyfile(ROOT / "baml.toml", project / "baml.toml")
    for path in (ROOT / "baml_src").glob("*.baml"):
        if not path.name.endswith("_test.baml"):
            shutil.copyfile(path, source / path.name)
    env = clean_env(project, "off")
    subprocess.run((*cli, "check"), cwd=project, env=env, check=True)
    subprocess.run(
        (*cli, "pack", "-f", "predict", "-f", "shadow", "--host-binary", host, "-o", binary),
        cwd=project, env=env, check=True,
    )


def recording_bytes(home):
    sizes = {"btel_bytes": 0, "cas_bytes": 0}
    for path in (home / ".baml" / "btel").rglob("*"):
        if path.is_file():
            if "cas" in path.relative_to(home).parts[:-1]:
                sizes["cas_bytes"] += path.stat().st_size
            elif path.suffix == ".btel":
                sizes["btel_bytes"] += path.stat().st_size
    return sizes


def run_process(command, case, telemetry="medium", expected_exit=0):
    case.mkdir()
    env = clean_env(case, telemetry)
    with (case / "stdout.json").open("wb") as stdout, (case / "stderr.txt").open("wb") as stderr:
        start = perf_counter_ns()
        with subprocess.Popen(command, cwd=case, env=env, stdout=stdout, stderr=stderr) as child:
            try:
                _, status, usage = os.wait4(child.pid, 0)
            except BaseException:
                child.kill()
                child.wait()
                raise
            elapsed = perf_counter_ns() - start
            child.returncode = os.waitstatus_to_exitcode(status)
    if child.returncode != expected_exit:
        raise RuntimeError(f"{command} exited {child.returncode}: {(case / 'stderr.txt').read_text()}")
    output = (case / "stdout.json").read_text()
    if expected_exit:
        assert not output.strip(), output
    rss_bytes = usage.ru_maxrss * (1 if platform.system() == "Darwin" else 1024)
    return (json.loads(output) if output else None), {
        "wall_ns": elapsed,
        "peak_rss_bytes": rss_bytes,
        **recording_bytes(case / "home"),
    }


def query(cli, project, sql):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(project), "DO_NOT_TRACK": "1", "LC_ALL": "C"}
    result = json.loads(subprocess.check_output(
        (*cli, "local", "recordings", "query", "--project-dir", project, "--output-format", "json", sql),
        cwd=project, env=env, text=True,
    ))
    assert result["outcome"]["status"] == "complete", result["outcome"]
    return result["rows"]


def trace_query(cli, case, sql=SPAN_QUERY):
    # The pack host records under HOME. Index only after measuring raw storage.
    home = case / "home"
    (home / "baml.toml").write_text('[package]\nname = "adaptive-shadow-recordings"\n')
    return query(cli, home, sql)


def predict(model, data, mode, policy=None, operation="predict-proba"):
    flags = ["--policy", policy] if policy is not None else []
    return ("predict", "--model", model, "--input", data, "--observation", mode,
            "--operation", operation, *flags, "--output-format", "json")


def shadow(champion, challenger, data, policy, operation):
    return ("shadow", "--champion", champion, "--challenger", challenger, "--input", data,
            "--operation", operation, "--policy", policy, "--output-format", "json")


def benchmark_inputs(out):
    rng = random.Random(SEED)
    iris = fixture("iris")
    model = write_json(out / "iris.json", iris)
    center = (5.0, 3.4, 1.5, 0.2)
    healthy = [[value + rng.uniform(-0.05, 0.05) for value in center] for _ in range(ROW_COUNT)]
    sparse = [list(row) for row in healthy]
    for i in range(0, ROW_COUNT, 80):
        sparse[i][0] = None
    shifted = [[x + 7.0, *rest] for x, *rest in healthy]
    batches = {name: write_json(out / f"{name}.json", dict(features=iris["features"], rows=rows))
               for name, rows in (("healthy", healthy), ("sparse", sparse), ("shifted", shifted))}
    policy = write_json(out / "default-policy.json", {})
    return model, batches, policy


def benchmark(binary, cli, out, trials):
    inputs = out / "inputs"
    inputs.mkdir()
    model, batches, policy = benchmark_inputs(inputs)
    cases = {
        (name, mode): predict(model, data, mode, policy if mode == "adaptive" else None)
        for name, data in batches.items() for mode in ("semantic", "adaptive", "full")
    }
    cases.update({(name, "shadow"): shadow(model, model, data, policy, "predict-proba")
                  for name, data in batches.items()})
    order = list(cases)
    rng = random.Random(SEED)
    triggered = {"healthy": [], "sparse": list(range(0, ROW_COUNT, 80)), "shifted": list(range(ROW_COUNT))}
    baseline = {}
    samples = []
    for trial in range(trials + 1):
        rng.shuffle(order)
        for name, mode in order:
            with TemporaryDirectory(dir=out) as scratch:
                case = Path(scratch) / "run"
                output, measurements = run_process((binary, *cases[name, mode]), case)
                predictions = output["predictions"]
                assert len(predictions) == ROW_COUNT
                assert predictions == baseline.setdefault(name, predictions), (name, mode)
                assert measurements["btel_bytes"] > 0
                if mode == "shadow":
                    summary = output["summary"]
                    k = len(triggered[name])
                    assert summary["challenger_ok"] and summary["label_disagreements"] == 0
                    assert (summary["triggered_rows"], summary["escalated_rows"], summary["capped"]) == (
                        k, min(k, 64), k > 64,
                    )
                if trial:
                    samples.append(dict(batch=name, mode=mode, trial=trial, **measurements))
                elif mode in ("adaptive", "shadow"):
                    prefix = "Shadow" if mode == "shadow" else ""
                    sql = (f"SELECT context_metadata['baml_ml.row_index'] FROM spans "
                           f"WHERE span_name = 'user.{prefix}EscalatedRow' ORDER BY start_time")
                    rows = trace_query(cli, case, sql)
                    assert [row[0] for row in rows] == triggered[name][:64], (name, mode, rows)
        print(f"Benchmark trial {trial}/{trials}" if trial else "Benchmark warmup", flush=True)
    return samples


def metadata(cli, host, binary, trials):
    def digest(path):
        with path.open("rb") as stream:
            return file_digest(stream, "sha256").hexdigest()

    os_details = {"system": platform.system(), "release": platform.release(), "arch": platform.machine()}
    if platform.system() == "Darwin":
        os_details["os_version"] = command_output("/usr/bin/sw_vers", "-productVersion")
        os_details["os_build"] = command_output("/usr/bin/sw_vers", "-buildVersion")
        hardware = json.loads(command_output(
            "/usr/sbin/system_profiler", "SPHardwareDataType", "-detailLevel", "mini", "-json",
        ))["SPHardwareDataType"][0]
        os_details["cpu"] = hardware.get("chip_type") or hardware["cpu_type"]
        os_details["model"] = hardware["machine_model"]
        os_details["cores"] = os.cpu_count()
    else:
        os_details["os_version"] = platform.freedesktop_os_release()["PRETTY_NAME"]
        cpu = Path("/proc/cpuinfo").read_text().splitlines()
        os_details["cpu"] = next((line.split(":", 1)[1].strip() for line in cpu
                                  if line.startswith("model name")), platform.machine())
    os_details["ram_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    source = {}
    for path in sorted((ROOT / "baml_src").glob("*.baml")):
        if not path.name.endswith("_test.baml"):
            source[path.name] = digest(path)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_commit": command_output("git", "-C", ROOT, "rev-parse", "HEAD"),
        "machine": os_details,
        "baml": command_output(*cli, "--version"),
        "cli_sha256": digest(cli[0]),
        "host_sha256": digest(host),
        "binary_sha256": digest(binary),
        "binary_bytes": binary.stat().st_size,
        "python": platform.python_version(),
        "rows_per_batch": ROW_COUNT,
        "seed": SEED,
        "trials_after_warmup": trials,
        "runtime_source_sha256": source,
    }


def write_report(out, info, samples):
    grouped = defaultdict(list)
    for row in samples:
        grouped[row["batch"], row["mode"]].append(row)
    machine = info["machine"]
    system_label = "macOS" if machine["system"] == "Darwin" else machine["system"]
    report = [
        "# Adaptive and shadow on a 4,096-row batch",
        "",
        f"{machine['cpu']}, {machine['ram_bytes'] / 2**30:g} GiB RAM, "
        f"{system_label} {machine['os_version']} ({machine.get('os_build', machine['release'])}), "
        f"{machine['arch']}. {info['baml']}, Python {info['python']}.",
        "",
        f"UTC run time {info['timestamp_utc']}. "
        f"Native executable {info['binary_bytes']:,} bytes. Tool and runtime source hashes "
        "are in [metadata.json](metadata.json).",
        "",
        "Packed Iris multinomial predict-proba. Seeded rows around one Iris input give "
        "0 anomalies (healthy), 52 missing values (sparse, 1.27%), or a large feature shift "
        "in all 4,096 rows (shifted). Adaptive and shadow use the default cap of 64. "
        "Shadow uses two handles for the same artifact to hold prediction and trigger rates constant.",
        "",
        f"Medians across {info['trials_after_warmup']} fresh processes per case after one warmup, "
        "with shuffled case order. Wall time includes process startup, input/output and native "
        "recording flush. Peak RSS is measured for each child. BTEL and CAS are logical file "
        "bytes in a fresh home before querying. Full captures the whole main batch. Replayed rows count additional single-row passes. [Raw samples](samples.jsonl).",
        "",
        "| Batch | Mode | Replayed rows | BTEL bytes | CAS bytes | Peak RSS MiB | Wall ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    triggered = {"healthy": 0, "sparse": 52, "shifted": 4096}
    for batch in ("healthy", "sparse", "shifted"):
        for mode in ("semantic", "adaptive", "full", "shadow"):
            rows = grouped[batch, mode]
            assert len(rows) == info["trials_after_warmup"]
            med = {field: statistics.median(row[field] for row in rows)
                   for field in ("btel_bytes", "cas_bytes", "peak_rss_bytes", "wall_ns")}
            replayed = min(64, triggered[batch]) if mode in ("adaptive", "shadow") else 0
            report.append(f"| {batch} | {mode} | {replayed} | {med['btel_bytes']:,.0f} | "
                          f"{med['cas_bytes']:,.0f} | {med['peak_rss_bytes'] / 2**20:.2f} | "
                          f"{med['wall_ns'] / 1e6:.2f} |")
    report.extend(["", "The harness checked exact champion prediction parity, expected replay indices "
                   "and [SQL stage pairs and failure evidence](sql-evidence.json) on separate small fixtures.", ""])
    (out / "report.md").write_text("\n".join(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="New or empty evidence directory")
    parser.add_argument("--samples", type=int, default=7)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    cli = (Path(os.environ["BAML_CLI"]).resolve(strict=True), "--agent-skill-check", "off")
    host = Path(os.environ["BAML_PACK_HOST"]).resolve(strict=True)
    if command_output(*cli, "--version") != f"baml-cli {BAML_VERSION}":
        parser.error(f"BAML_CLI must be version {BAML_VERSION}")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error("out must be empty")
    with TemporaryDirectory(dir=out) as scratch:
        work = Path(scratch)
        project = work / "project"
        binary = work / "baml-ml"
        build(cli, host, project, binary)

        def execute(name, args, telemetry="medium", expected_exit=0):
            return run_process((binary, *args), work / name, telemetry, expected_exit)

        def repeat_probe(model):
            shutil.copyfile(HERE / "repeated.baml", project / "baml_src" / "repeated.baml")
            result = subprocess.run(
                (*cli, "run", "AdaptiveRepeated", "--output-format", "json", "--", "--model", model),
                cwd=project, env=clean_env(project, "medium"), text=True, check=True, stdout=subprocess.PIPE,
            )
            return json.loads(result.stdout), query(cli, project, SPAN_QUERY)

        evidence = verify_runtime(
            work, execute, lambda name, sql=SPAN_QUERY: trace_query(cli, work / name, sql),
            repeat_probe, fixture, write_json, predict, shadow,
        )
        samples = benchmark(binary, cli, work, args.samples)
        info = metadata(cli, host, binary, args.samples)

    write_json(out / "metadata.json", info)
    write_json(out / "sql-evidence.json", evidence)
    (out / "samples.jsonl").write_text("".join(json.dumps(sample) + "\n" for sample in samples))
    write_report(out, info, samples)
    print(f"Evidence written to {out}")


if __name__ == "__main__":
    main()
