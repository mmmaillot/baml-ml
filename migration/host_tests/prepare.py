"""Build a test-only mirror of the unchanged historical Python assertions.

Never import both generated SDKs in the same process. Copy the original tests
without modifying their assertions, and replace only legacy SDK imports and
bootstrap. The generated tree is disposable; the original suite remains the
source of truth.
"""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "generated"
NUMERICAL_TESTS = (
    "test_primitives.py",
    "test_imputer.py",
    "test_scaler.py",
    "test_linreg.py",
    "test_logreg.py",
    "test_numerical_hardening.py",
    "test_pipeline.py",
    "test_classification_pipeline.py",
    "test_examples.py",
    "test_benchmarks.py",
)


def copy(relative):
    src = ROOT / relative
    dst = OUT / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text().replace("from baml_client import b", "from compat import b")
    dst.write_text(text)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "tests").mkdir(parents=True)
    shutil.copyfile(HERE / "compat.py", OUT / "compat.py")
    shutil.copyfile(HERE / "test_new_sdk.py", OUT / "tests" / "test_new_sdk.py")
    for test in NUMERICAL_TESTS:
        copy("tests/" + test)
    for file in ("benchmarks/datasets.py", "benchmarks/sklearn_benchmark.py",
                 "examples/iris_example.py", "examples/diabetes_example.py"):
        copy(file)
    # The old reference module invokes the npm CLI and imports baml_client at
    # module initialization. Replace that bootstrap only; all reference and
    # pipeline orchestration functions stay verbatim.
    src = (ROOT / "benchmarks/reference.py").read_text()
    start = src.index("def _ensure_generated_client()")
    end = src.index("from baml_client import b", start) + len("from baml_client import b")
    ref = src[:start] + "from compat import b\n" + src[end:]
    (OUT / "benchmarks" / "reference.py").write_text(ref)
    print("Prepared", len(NUMERICAL_TESTS), "historical test modules plus new SDK smoke tests")


if __name__ == "__main__":
    main()
