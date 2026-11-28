"""Run the manual macOS ARM64 standalone inference benchmark."""

import argparse
from collections import defaultdict
from datetime import datetime
from hashlib import file_digest
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter_ns

# Set limits before importing NumPy or loading scikit-learn in a child.
THREAD_ENV = dict.fromkeys(("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                           "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                           "BLIS_NUM_THREADS"), "1")
os.environ.update(THREAD_ENV)

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "artifacts"))
from export_sklearn import write_artifact

MODES = ("off", "timing", "semantic", "full")
PHASES = ("read_ns", "prepare_ns", "decode_ns", "inference_ns")
WORKER = (sys.executable, HERE / "sklearn_job.py")
SEED = 42


def command_output(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def sha256(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")


def build(cli, host, out, env):
    for target in ("predict", "loaded"):
        with TemporaryDirectory(dir=out) as scratch:
            project = Path(scratch)
            source = project / "baml_src"
            source.mkdir()
            shutil.copyfile(ROOT / "baml.toml", project / "baml.toml")
            for path in (ROOT / "baml_src").rglob("*.baml"):
                rel = path.relative_to(ROOT / "baml_src")
                if path.name.endswith("_test.baml") or "tests" in rel.parts:
                    continue
                dest = source / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)
            if target == "loaded":
                shutil.copyfile(HERE / "loaded.baml", source / "loaded.baml")
            subprocess.run((cli, "check", "--agent-skill-check", "off"),
                           cwd=project, env=env, check=True)
            subprocess.run((cli, "pack", "--function", target, "--host-binary", host,
                            "--output", out / target), cwd=project, env=env, check=True)
            subprocess.run(("/usr/bin/codesign", "--verify", out / target), check=True)


def fit_inputs(out, sizes):
    iris = load_iris()
    train, test, labels, _ = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=SEED, stratify=iris.target)
    train[0, 0] = np.nan
    fitted = make_pipeline(SimpleImputer(), StandardScaler(),
                           LogisticRegression(max_iter=10000, tol=1e-10)).fit(train, labels)
    write_artifact(fitted, iris.feature_names, out / "model.json")
    model = json.loads((out / "model.json").read_text())
    joblib.dump((model["model_id"], model["features"], fitted), out / "model.joblib")
    expected = {}
    for size in sizes:
        rows = test[np.arange(size) % len(test)].tolist()
        for i in range(0, size, 8):
            rows[i][(i // 8) % len(iris.feature_names)] = None
        write_json(out / f"input-{size}.json", dict(features=iris.feature_names, rows=rows))
        # Use the same serialized batch and conversion as the Python subprocess.
        batch = json.loads((out / f"input-{size}.json").read_text())
        expected[size] = fitted.predict_proba(np.array(batch["rows"], dtype=np.float64))
    return model, expected


def recording_bytes(home):
    result = defaultdict(int, btel=0, cas=0, index=0, other=0)
    for path in home.rglob("*"):
        if not path.is_file():
            continue
        parts = path.relative_to(home).parts
        if path.suffix == ".btel":
            kind = "btel"
        elif {"cas", "blobs", "btel_blobs"}.intersection(parts[:-1]):
            kind = "cas"
        elif "index" in parts[:-1] or ".sqlite" in path.name or ".db" in path.name:
            kind = "index"
        else:
            kind = "other"
        result[kind] += path.stat().st_size
    result["total"] = sum(result.values())
    return dict(result)


def run_child(command, out, mode="off"):
    # wait4 gives the rusage for this child. ru_maxrss is in bytes on Darwin.
    with TemporaryDirectory(dir=out) as scratch:
        case = Path(scratch)
        home = case / "home"
        home.mkdir()
        env = dict(THREAD_ENV, PATH="/usr/bin:/bin", HOME=str(home), TMPDIR=str(case),
                   LC_ALL="C", DO_NOT_TRACK="1", PYTHONDONTWRITEBYTECODE="1",
                   BAML_TELEMETRY="off" if mode == "off" else "medium")
        with (case / "stdout").open("w+b") as stdout, (case / "stderr").open("w+b") as stderr:
            start = perf_counter_ns()
            with subprocess.Popen(command, cwd=case, env=env, stdout=stdout, stderr=stderr) as child:
                try:
                    _, status, usage = os.wait4(child.pid, 0)
                except BaseException:
                    child.kill()
                    raise
                elapsed = perf_counter_ns() - start
                child.returncode = os.waitstatus_to_exitcode(status)
            stdout.seek(0)
            stderr.seek(0)
            output, diagnostic = stdout.read(), stderr.read().decode(errors="replace")
        sample = dict(wall_ns=elapsed, peak_rss_bytes=usage.ru_maxrss,
                      output_bytes=len(output), storage_bytes=recording_bytes(home))
        return sample, child.returncode, output, diagnostic


def prediction_command(out, mode, size, path=None, model=None, operation="predict-proba"):
    if mode == "sklearn":
        return (*WORKER, "--model", out / "model.joblib", "--input", out / f"input-{size}.json")
    command = (out / "predict", "predict", "--model", model or out / "model.json",
               "--input", path or out / f"input-{size}.json", "--operation", operation,
               "--output-format", "json")
    if mode in ("timing", "full"):
        return (*command, "--observation", mode)
    return command


def validate_prediction(result, model, expected, phased):
    if not phased:
        expected_metadata = dict(schema_version=1, model_id=model["model_id"],
                                 estimator_kind=model["estimator_kind"],
                                 operation="predict-proba", classes=model["classes"])
        if any(result.get(k) != v for k, v in expected_metadata.items()):
            raise ValueError("prediction metadata differs from the fitted model")
    actual = np.asarray(result["predictions"])
    if actual.shape != expected.shape:
        raise ValueError(f"prediction shape differs: {actual.shape} != {expected.shape}")
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)
    return float(np.max(np.abs(actual - expected)))


def measure(out, model, expected, cases, sample_count, log):
    rng = random.Random(SEED)
    for trial in range(sample_count + 1):
        rng.shuffle(cases)
        for kind, mode, size, calls, command in cases:
            mode_env = "off" if mode in ("baml", "sklearn") else mode
            sample, rc, output, diagnostic = run_child(command, out, mode_env)
            if rc:
                raise RuntimeError(f"{kind}/{mode}/{size} exited {rc}: {diagnostic}")
            if kind == "help":
                if b"usage:" not in output.lower():
                    raise ValueError(f"missing CLI help for {mode}")
            else:
                phased = kind == "loaded"
                result = json.loads(output)
                sample["max_absolute_error"] = validate_prediction(result, model, expected[size], phased)
                if phased:
                    for key in PHASES:
                        value = result[key]
                        sample[key] = int(value["$bigint"] if isinstance(value, dict) else value)
                        if sample[key] <= 0:
                            raise ValueError(f"nonpositive {mode} phase {key}: {value}")
                expected_recording = mode in MODES and mode != "off"
                if (sample["storage_bytes"]["btel"] > 0) != expected_recording:
                    raise ValueError(f"unexpected BTEL recording in {kind}/{mode}/{size}")
            if trial:
                sample.update(kind=kind, mode=mode, size=size, calls=calls, trial=trial)
                log.write(json.dumps(sample) + "\n")
                log.flush()
        print(f"{cases[0][0]}: {'warmup' if not trial else f'{trial}/{sample_count}'}", flush=True)


def check_errors(out):
    batch = json.loads((out / "input-1.json").read_text())
    artifact = json.loads((out / "model.json").read_text())
    errors = {
        "malformed": "{",
        "wrong-order": dict(batch, features=batch["features"][::-1]),
        "ragged": dict(batch, rows=[batch["rows"][0], [0.1]]),
        "empty": dict(batch, rows=[]),
        "nonnumeric": dict(batch, rows=[["bad", 3.1, 1.4, 0.2]]),
        "invalid-model": dict(artifact, schema_version=2),
        "wrong-operation": None,
    }
    results = {}
    with TemporaryDirectory(dir=out) as scratch:
        path = Path(scratch) / "override.json"
        for name, payload in errors.items():
            override = {}
            if name == "malformed":
                path.write_text(payload)
            elif payload is not None:
                write_json(path, payload)
            if payload is not None:
                override["model" if name == "invalid-model" else "path"] = path
            if name == "wrong-operation":
                override["operation"] = "fit"
            command = prediction_command(out, "off", 1, **override)
            sample, rc, stdout, stderr = run_child(command, out)
            try:
                body = json.loads(stdout)
            except ValueError:
                body = None
            successful_output = isinstance(body, dict) and "predictions" in body
            if not 0 < rc < 125 or not stderr.strip() or successful_output:
                raise RuntimeError(f"invalid rejection for {name}: {rc}, {stdout!r}, {stderr!r}")
            results[name] = dict(exit_code=rc, stderr=stderr, **sample)
    write_json(out / "errors.json", results)


def summarize(samples):
    grouped = defaultdict(list)
    for row in samples:
        grouped[row["kind"], row["size"], row["mode"]].append(row)
    result = []
    for (kind, size, mode), rows in sorted(grouped.items()):
        walls = sorted(row["wall_ns"] for row in rows)
        wall = statistics.median(walls)
        summary = dict(kind=kind, mode=mode, size=size, calls=rows[0]["calls"], samples=len(rows),
                       wall_ms=wall / 1e6, min_ms=walls[0] / 1e6, max_ms=walls[-1] / 1e6,
                       peak_rss_mib=statistics.median(row["peak_rss_bytes"] for row in rows) / 2**20)
        if kind == "process":
            summary["rows_per_second"] = size * 1e9 / wall
            summary["storage_bytes"] = {key: statistics.median(row["storage_bytes"][key] for row in rows)
                                        for key in rows[0]["storage_bytes"]}
            summary["output_bytes"] = statistics.median(row["output_bytes"] for row in rows)
        if kind == "loaded":
            for key in PHASES:
                divisor = rows[0]["calls"] if key == "inference_ns" else 1
                summary[key.removesuffix("_ns") + "_ms"] = statistics.median(row[key] for row in rows) / (divisor * 1e6)
            summary["rows_per_second"] = size * 1e3 / summary["inference_ms"]
        if kind != "help":
            summary["max_absolute_error"] = max(row["max_absolute_error"] for row in rows)
        result.append(summary)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory for binaries, inputs and results")
    parser.add_argument("--samples", type=int, default=15)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if (platform.system(), platform.machine()) != ("Darwin", "arm64"):
        parser.error("this benchmark requires native macOS ARM64")
    if version("scikit-learn") != "1.8.0":
        parser.error("install scikit-learn==1.8.0")
    try:
        cli, host = (Path(os.environ[key]).resolve(strict=True) for key in ("BAML_CLI", "BAML_PACK_HOST"))
    except (KeyError, FileNotFoundError):
        parser.error("set BAML_CLI and BAML_PACK_HOST to the verified BAML 0.26.0 executables")
    baml_version = command_output(cli, "--version").splitlines()[-1]
    if baml_version != "baml-cli 0.26.0":
        parser.error(f"expected baml-cli 0.26.0, found {baml_version}")

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sizes_calls = {1: 500, 128: 100, 4096: 5}
    env = dict(PATH=os.defpath, BAML_TELEMETRY="off", DO_NOT_TRACK="1")
    build(cli, host, out, env)
    with threadpool_limits(limits=1):
        model, expected = fit_inputs(out, sizes_calls)
        metadata = dict(
            revision=command_output("git", "-C", ROOT, "rev-parse", "HEAD"),
            working_tree=command_output("git", "-C", ROOT, "status", "--porcelain"),
            timestamp=datetime.now().astimezone().isoformat(),
            hardware=[line.strip() for line in command_output(
                "/usr/sbin/system_profiler", "SPHardwareDataType").splitlines()
                if line.strip().startswith(("Model Name:", "Chip:", "Total Number of Cores:", "Memory:"))],
            os=command_output("/usr/bin/sw_vers"),
            power=command_output("/usr/bin/pmset", "-g", "batt").splitlines()[0],
            python=sys.version, baml=baml_version, thread_settings=THREAD_ENV,
            threadpools=[dict(pool, filepath=Path(pool["filepath"]).name) for pool in threadpool_info()],
            packages={package: version(package) for package in
                      ("scikit-learn", "numpy", "scipy", "joblib", "threadpoolctl", "cloudpickle")},
            binaries={name: dict(format=command_output("/usr/bin/file", "-b", path),
                                 bytes=path.stat().st_size, sha256=sha256(path))
                      for name, path in dict(predict=out / "predict", loaded=out / "loaded",
                                            python=Path(sys.executable).resolve(),
                                            baml_cli=cli, baml_pack_host=host).items()},
            model_id=model["model_id"], input_seed=SEED, samples=args.samples,
            loaded_calls=sizes_calls, loaded_warmups=3,
            files={path.name: dict(bytes=path.stat().st_size, sha256=sha256(path))
                   for path in out.iterdir() if path.suffix in (".json", ".joblib")},
        )
    write_json(out / "metadata.json", metadata)

    check_errors(out)
    help_cases = [("help", "baml", 0, 0, (out / "predict", "predict", "--help")),
                  ("help", "sklearn", 0, 0, (*WORKER, "--help"))]
    process_cases = [("process", mode, size, 1, prediction_command(out, mode, size))
                     for size in sizes_calls for mode in (*MODES, "sklearn")]
    loaded_cases = []
    for size, calls in sizes_calls.items():
        flags = ("--calls", str(calls), "--warmups", "3")
        input_path = out / f"input-{size}.json"
        loaded_cases += [
            ("loaded", "baml", size, calls,
             (out / "loaded", "loaded", "--model", out / "model.json", "--input", input_path,
              *flags, "--output-format", "json")),
            ("loaded", "sklearn", size, calls,
             (*WORKER, "--model", out / "model.joblib", "--input", input_path, *flags, "--phases")),
        ]
    raw_path = out / "samples.jsonl"
    with raw_path.open("w") as log:
        for cases in (help_cases, process_cases, loaded_cases):
            measure(out, model, expected, cases, args.samples, log)
    summary = summarize(json.loads(line) for line in raw_path.read_text().splitlines())
    write_json(out / "summary.json", summary)
    print(f"Results: {out}", flush=True)


if __name__ == "__main__":
    main()
