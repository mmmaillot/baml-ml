"""A clean checkout can run a deterministic BAML numerical function from
Python without an LLM or API key."""

from baml_client import b


def test_dot_runs_without_llm_or_api_key():
    result = b.Dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert result.ok
    assert result.value == 32.0
