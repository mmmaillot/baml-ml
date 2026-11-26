"""Run one deterministic BAML numerical function from Python.

No LLM and no API key are required. From the repository root:

    npm install                       # once, provides the baml CLI
    npx baml --features beta generate # once per change to baml_src/
    .venv/bin/python examples/dot_example.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baml_client import b  # noqa: E402


def main() -> None:
    result = b.Dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    if not result.ok:
        raise SystemExit(f"Dot failed: {result.error}")
    print(f"Dot([1, 2, 3], [4, 5, 6]) = {result.value}")


if __name__ == "__main__":
    main()
