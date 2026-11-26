"""A clean checkout can run a deterministic BAML numerical function from
Python without an LLM or API key."""

from pathlib import Path

import conftest
from baml_client import b


def test_dot_runs_without_llm_or_api_key():
    result = b.Dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert result.ok
    assert result.value == 32.0


def test_generated_client_lives_at_repo_root_without_double_nesting():
    repo_root = Path(__file__).resolve().parent.parent
    assert (repo_root / "baml_client" / "__init__.py").exists()
    assert not (repo_root / "baml_client" / "baml_client").exists()


def test_baml_cli_prefers_local_node_modules_over_global_path(monkeypatch, tmp_path):
    fake_repo = tmp_path / "repo"
    local_bin = fake_repo / "node_modules" / ".bin"
    global_bin = tmp_path / "global_bin"
    local_bin.mkdir(parents=True)
    global_bin.mkdir(parents=True)

    local_baml = local_bin / "baml"
    global_baml = global_bin / "baml"
    local_baml.write_text("#!/bin/sh\n")
    global_baml.write_text("#!/bin/sh\n")
    local_baml.chmod(0o755)
    global_baml.chmod(0o755)

    monkeypatch.setattr(conftest, "REPO_ROOT", fake_repo)
    monkeypatch.setenv("PATH", str(global_bin))
    assert Path(conftest._baml_cli()).resolve() == local_baml.resolve()


def test_client_freshness_detects_newer_baml_source(monkeypatch, tmp_path):
    import os

    src_dir = tmp_path / "baml_src"
    client_dir = tmp_path / "baml_client"
    src_dir.mkdir()
    client_dir.mkdir()
    src_file = src_dir / "primitives.baml"
    init_py = client_dir / "__init__.py"
    inlined_py = client_dir / "inlinedbaml.py"
    src_file.write_text("// src\n")
    init_py.write_text("")
    inlined_py.write_text("")

    monkeypatch.setattr(conftest, "BAML_SRC_DIR", src_dir)
    monkeypatch.setattr(conftest, "CLIENT_DIR", client_dir)

    os.utime(src_dir, (100, 100))
    os.utime(src_file, (100, 100))
    os.utime(inlined_py, (200, 200))
    assert conftest._client_is_up_to_date() is True

    os.utime(src_file, (300, 300))
    assert conftest._client_is_up_to_date() is False
