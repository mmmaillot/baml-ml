"""Replacement smoke coverage for the new generated Python SDK."""
import math
from importlib.metadata import version
from pathlib import Path

import baml_sdk
from compat import b, types


def test_new_sdk_runs_without_llm_or_api_key():
    assert version("baml_bridge") == "0.25.0"
    result = b.Dot([1.0, 2.0], [3.0, 4.0])
    assert isinstance(result, types.NumResult)
    assert result.ok and result.value == 11.0


def test_generated_package_is_isolated_and_not_double_nested():
    package = Path(baml_sdk.__file__).resolve().parent
    assert package.name == "baml_sdk"
    assert package.parent.name == "gen"
    assert package.parent.parent.name == "migration"
    assert not (package / "baml_sdk").exists()


def test_non_finite_values_cross_sdk_in_both_directions():
    assert b.IsNaN(math.nan)
    assert not b.IsFinite(math.inf)
    assert math.isnan(b.Exp(math.nan))
    assert b.Sqrt(math.inf) == math.inf
    imputed = b.ImputerFit([[1.0], [math.nan]])
    assert imputed.ok and imputed.imputer.means == [1.0]
