#!/usr/bin/env python3
"""Architectural linter for baml_src/.

Fails if:
1. any file in baml_src/kernels/ or baml_src/contracts/ references
   $trace, trace. or ObservationMode (those layers must stay trace-free);
2. any file under baml_src/ has a _v<digit> version suffix in its name;
3. any non-test .baml file exceeds 300 lines;
4. a frozen @contract(btel-v1) BTEL span function or an @public-api entry
   point is missing or lacks its annotation.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "baml_src"

MAX_NON_TEST_LOC = 300
TRACE_TOKENS = ("$trace", "trace.", "ObservationMode")
VERSION_SUFFIX_RE = re.compile(r"_v\d")

# Frozen BTEL span ABI: user.<FunctionName> becomes the span_name in
# recordings, so these functions must never be renamed.
# See docs/architecture/observation-contract.md.
CONTRACT_SPANS = [
    "PrepareArtifact",
    "PrepareTreeArtifact",
    "PreparePipelineArtifact",
    "ModelLoaded",
    "TreeModelLoaded",
    "PipelineModelLoaded",
    "PredictLoadedLinear",
    "PredictLoadedProba",
    "PredictLoadedLabels",
    "PredictLoadedTree",
    "PredictLoadedPipeline",
    "ObservedImpute",
    "ObservedScale",
    "ObservedImputeStep",
    "ObservedScaleStep",
    "ObservedOneHotEncode",
    "ObservedColumnTransformer",
    "ObservedColumnTransformerBranch",
    "ObservedLinear",
    "ObservedDecisionScores",
    "ObservedSoftmax",
    "ObservedLabels",
    "ObservedTreeTraversal",
    "ObservedForestTraversal",
    "RunAdaptive",
    "RunAdaptiveCells",
    "RunSelectiveModel",
    "AdaptiveSummary",
    "EscalatedRow",
    "ShadowBatch",
    "ShadowEscalatedRow",
    "ShadowChallengerFailed",
]

PUBLIC_API = ["predict", "shadow", "RunBatch", "RunShadow"]


def iter_baml_files():
    yield from sorted(SRC.rglob("*.baml"))


def is_test_file(path):
    return "tests" in path.relative_to(SRC).parts or path.stem.endswith("_test")


def annotated(lines, index, marker):
    """True if the item starting at lines[index] carries the marker comment."""
    i = index - 1
    while i >= 0 and lines[i].lstrip().startswith("//"):
        if marker in lines[i]:
            return True
        i -= 1
    return False


def check_required_functions(errors):
    required = [(name, "@contract(btel-v1)") for name in CONTRACT_SPANS]
    required += [(name, "@public-api") for name in PUBLIC_API]
    for name, marker in required:
        pattern = re.compile(rf"^function {re.escape(name)}\(")
        found = False
        for path in iter_baml_files():
            lines = path.read_text().splitlines()
            for i, line in enumerate(lines):
                if pattern.match(line):
                    if annotated(lines, i, marker):
                        found = True
                    else:
                        errors.append(
                            f"{path.relative_to(ROOT)}:{i + 1}: function {name} is missing its {marker} annotation"
                        )
        if not found and not any(name in e for e in errors):
            errors.append(f"required function {name} ({marker}) not found under baml_src/")


def main():
    if not SRC.is_dir():
        print(f"error: {SRC} not found", file=sys.stderr)
        return 1
    errors = []
    for path in iter_baml_files():
        rel = path.relative_to(ROOT)
        text = path.read_text()
        layer = path.relative_to(SRC).parts[0] if len(path.relative_to(SRC).parts) > 1 else ""
        if layer in ("kernels", "contracts"):
            for token in TRACE_TOKENS:
                if token in text:
                    errors.append(f"{rel}: forbidden token {token!r} in trace-free layer {layer}/")
        if VERSION_SUFFIX_RE.search(path.stem):
            errors.append(f"{rel}: version suffix in file name (use domain names, schema_version is wire data only)")
        if not is_test_file(path) and len(text.splitlines()) > MAX_NON_TEST_LOC:
            errors.append(
                f"{rel}: {len(text.splitlines())} lines exceeds the {MAX_NON_TEST_LOC}-line limit for non-test files"
            )
    check_required_functions(errors)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        print(f"check_architecture: {len(errors)} violation(s)", file=sys.stderr)
        return 1
    files = list(iter_baml_files())
    print(f"check_architecture: OK ({len(files)} .baml files, layers kernels/ contracts/ observability/ runtime/ surface/ tests/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
