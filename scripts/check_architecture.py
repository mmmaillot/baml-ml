#!/usr/bin/env python3
"""Architectural linter for baml_src/.

Fails if:
1. a source sits outside the declared layers (top-level directory must be one
   of kernels/ contracts/ observability/ runtime/ surface/ tests/, and tests/
   is only allowed at the top level), or any file in kernels/ or contracts/
   references the trace machinery ($trace directives, @trace.* annotations,
   trace.<module member> calls/types) or ObservationMode — comments and
   ordinary field names are not matched;
2. any file under baml_src/ has a _v<digit> version suffix in its name;
3. any non-test .baml file exceeds 300 lines — "test" uses the same
   classification as scripts/stage_runtime.sh (a tests/ path component or a
   *_test.baml name), so the LOC limit and the packaged source set agree;
4. a frozen @contract(btel-v1) BTEL span function or an @public-api entry
   point is missing from the packageable (non-test) sources or lacks its
   annotation.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "baml_src"

MAX_NON_TEST_LOC = 300
LAYERS = ("kernels", "contracts", "observability", "runtime", "surface", "tests")
VERSION_SUFFIX_RE = re.compile(r"_v\d")

# Actual trace-machinery references in code (comments are stripped first):
# the $trace directive parameter, @trace.* field annotations, and members of
# the trace module. Field accesses on an unrelated local named "trace" (e.g.
# trace.leaf) do not match.
TRACE_PATTERNS = [
    re.compile(r"\$trace\b"),
    re.compile(r"@trace\."),
    re.compile(r"\btrace\.(Options|span|empty_span|timing|context)\b"),
    re.compile(r"\bObservationMode\b"),
]

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


def is_test_file(rel_parts, stem):
    """Same test/runtime classification as scripts/stage_runtime.sh."""
    return "tests" in rel_parts or stem.endswith("_test")


def code_only(line):
    """Strip line comments so prose mentioning trace. is not matched."""
    idx = line.find("//")
    return line if idx < 0 else line[:idx]


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
    # Only packageable (non-test) sources are searched: stage_runtime.sh omits
    # test files, so a required function defined only under tests/ must fail.
    packageable = [
        path for path in iter_baml_files()
        if not is_test_file(path.relative_to(SRC).parts, path.stem)
    ]
    for name, marker in required:
        pattern = re.compile(rf"^function {re.escape(name)}\(")
        found = False
        for path in packageable:
            lines = path.read_text().splitlines()
            for i, line in enumerate(lines):
                if pattern.match(line):
                    if annotated(lines, i, marker):
                        found = True
                    else:
                        errors.append(
                            f"{path.relative_to(ROOT)}:{i + 1}: function {name} is missing its {marker} annotation"
                        )
        if not found and not any(f"function {name} " in e for e in errors):
            errors.append(f"required function {name} ({marker}) not found in packageable baml_src sources")


def main():
    if not SRC.is_dir():
        print(f"error: {SRC} not found", file=sys.stderr)
        return 1
    errors = []
    for path in iter_baml_files():
        rel = path.relative_to(SRC)
        rel_path = path.relative_to(ROOT)
        text = path.read_text()
        parts = rel.parts
        layer = parts[0] if len(parts) > 1 else ""
        if len(parts) == 1:
            errors.append(f"{rel_path}: .baml file directly under baml_src/ is outside the declared layers")
            continue
        if layer not in LAYERS:
            errors.append(f"{rel_path}: unknown layer {layer!r} (expected one of {', '.join(LAYERS)})")
        if "tests" in parts[1:]:
            errors.append(f"{rel_path}: tests/ directories are only allowed at baml_src/tests/")
        if layer in ("kernels", "contracts"):
            code = "\n".join(code_only(line) for line in text.splitlines())
            for pattern in TRACE_PATTERNS:
                if pattern.search(code):
                    errors.append(f"{rel_path}: forbidden trace reference {pattern.pattern!r} in trace-free layer {layer}/")
        if VERSION_SUFFIX_RE.search(path.stem):
            errors.append(f"{rel_path}: version suffix in file name (use domain names, schema_version is wire data only)")
        if not is_test_file(parts, path.stem) and len(text.splitlines()) > MAX_NON_TEST_LOC:
            errors.append(
                f"{rel_path}: {len(text.splitlines())} lines exceeds the {MAX_NON_TEST_LOC}-line limit for non-test files"
            )
    check_required_functions(errors)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        print(f"check_architecture: {len(errors)} violation(s)", file=sys.stderr)
        return 1
    files = list(iter_baml_files())
    print(f"check_architecture: OK ({len(files)} .baml files, layers {'/ '.join(LAYERS)}/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
