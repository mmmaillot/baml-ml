"""Run each probe in an isolated project and home; archive evidence, not guesses."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
CLI = str(Path(os.environ["BAML_CLI"]).resolve())
OUT = Path(sys.argv[1]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
CASES = [
    ("default-nested", "DefaultNested", "default", 42),
    ("off-nested", "SpanNested", "off", 42),
    ("span-nested", "SpanNested", "default", 42),
    ("timing-nested", "TimingNested", "default", 42),
    ("empty-nested", "EmptyNested", "default", 42),
    ("default-loop", "DefaultLoop", "default", 50005000),
    ("timing-loop", "TimingLoop", "default", 50005000),
    ("span-loop", "SpanLoop", "default", 50005000),
    ("matrix-small", "MatrixSmall", "default", [[1, 2], [3, 4]]),
    ("matrix-large", "MatrixLarge", "default", 100),
    ("failure", "Failure", "default", None),
]


def command(args, cwd, env, target):
    start = time.perf_counter()
    p = subprocess.run([CLI, *args], cwd=cwd, env=env, capture_output=True,
                       timeout=120)
    target.with_suffix(".stdout").write_bytes(p.stdout)
    target.with_suffix(".stderr").write_bytes(p.stderr)
    return {"args": args, "exit": p.returncode, "wall_s": time.perf_counter()-start}


def main():
    summary = []
    for name, function, mode, expected in CASES:
        case = OUT / name
        case.mkdir()
        project = case / "project"
        project.mkdir()
        for f in ("baml.toml", "probe.baml"):
            shutil.copyfile(SOURCE / f, project / f)
        home = case / "home"
        home.mkdir()
        env = dict(os.environ, HOME=str(home), XDG_CONFIG_HOME=str(home / "config"),
                   XDG_DATA_HOME=str(home / "data"), XDG_CACHE_HOME=str(home / "cache"),
                   DO_NOT_TRACK="1")
        for key in ("BOUNDARY_API_KEY", "BOUNDARY_PROJECT", "BAML_TELEMETRY"):
            env.pop(key, None)
        if mode == "off":
            env["BAML_TELEMETRY"] = "off"
        checked = command(["check"], project, env, case / "check")
        if checked["exit"]:
            print((case / "check.stderr").read_text())
            raise RuntimeError("Probe does not compile")
        run = command(["run", function, "--output-format", "json"], project, env, case / "run")
        if expected is not None:
            assert run["exit"] == 0, (name, (case / "run.stderr").read_text())
            actual = json.loads((case / "run.stdout").read_text())
            assert actual == expected, (name, actual, expected)
        else:
            assert run["exit"] == 1, (name, run)
        # Snapshot before querying: query itself must not inflate recording bytes.
        recordings = [{"path": str(f.relative_to(case)), "bytes": f.stat().st_size}
                      for f in case.rglob("*") if f.is_file() and
                      (f.suffix in (".btel", ".bcs", ".sqlite", ".db") or
                       "recordings" in f.parts)]
        queries = [
            ("schema", ["local", "recordings", "schema"]),
            ("count", ["local", "recordings", "query", "--output-format", "json",
                       "SELECT COUNT(*) AS n FROM spans"]),
            ("grouped", ["local", "recordings", "query", "--output-format", "json",
                         "SELECT span_name, COUNT(*) AS n FROM spans GROUP BY span_name"]),
            ("sample", ["local", "recordings", "query", "--output-format", "json",
                        "SELECT * FROM spans LIMIT 12"]),
        ]
        evidence = {k: command(a, project, env, case / k) for k, a in queries}
        item = {"case": name, "function": function, "mode": mode,
                "run": run, "queries": evidence, "recording_files": recordings,
                "recording_bytes": sum(f["bytes"] for f in recordings) if recordings else None}
        summary.append(item)
        print(json.dumps(item), flush=True)
        for k in ("schema", "count", "grouped", "sample"):
            print(k, (case / (k + ".stdout")).read_text()[:2500], flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    # Do not interpret absence of recordings as zero spans when queries fail.
    # Require usable query evidence in at least one tracing-enabled experiment.
    assert any(s["queries"]["count"]["exit"] == 0 for s in summary if s["mode"] != "off")


if __name__ == "__main__":
    main()
