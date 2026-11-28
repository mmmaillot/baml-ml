#!/usr/bin/env python3
"""Generate the declarative negative conformance catalog.

Writes artifacts/conformance/invalid_mutations.json plus one JSON file per
mutation under artifacts/conformance/mutations/ and the synthetic schema 3
base under artifacts/conformance/bases/. Mutations are derived from the
committed sklearn fixtures (and one hand-built synthetic schema 3 pipeline)
so every cataloged artifact stays decodable JSON; the static validators and
the Prepare* loaders must reject each one with the cataloged message.

Non-finite floats have no JSON form (the BAML JSON decoder rejects NaN and
out-of-range numbers), so non-finite parameter mutations are exercised
in-memory in baml_src/tests/conformance_test.baml instead of here.

Usage: python3 artifacts/conformance/generate.py   (from the repo root)
"""

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "conformance"
FIXTURES = ROOT / "artifacts" / "fixtures"


def synthetic_v3_base() -> dict:
    """Hand-built valid schema 3 pipeline: numeric branch, categorical branch
    (string imputer + one-hot), passthrough remainder, linear estimator."""
    return {
        "schema_version": 3,
        "source": "scikit-learn",
        "source_version": "1.8.0",
        "model_id": "conformance-synthetic",
        "features": ["a", "b", "c", "d"],
        "input_kinds": ["numeric", "numeric", "categorical", "numeric"],
        "steps": [
            {
                "kind": "column_transformer",
                "n_features_in": 4,
                "n_features_out": 5,
                "column_transformer": {
                    "branches": [
                        {
                            "name": "num",
                            "columns": [0, 1],
                            "mode": "pipeline",
                            "steps": [
                                {
                                    "kind": "standard_scaler",
                                    "n_features_in": 2,
                                    "n_features_out": 2,
                                    "scaler": {"means": [0.0, 0.0], "scales": [1.0, 1.0]},
                                }
                            ],
                        },
                        {
                            "name": "cat",
                            "columns": [2],
                            "mode": "pipeline",
                            "steps": [
                                {
                                    "kind": "simple_imputer",
                                    "n_features_in": 1,
                                    "n_features_out": 1,
                                    "imputer": {
                                        "strategy": "most_frequent",
                                        "value_kind": "string",
                                        "statistics": [],
                                        "fill_strings": ["x"],
                                    },
                                },
                                {
                                    "kind": "one_hot_encoder",
                                    "n_features_in": 1,
                                    "n_features_out": 2,
                                    "one_hot": {
                                        "columns": [{"kind": "string", "strings": ["x", "y"], "ints": []}],
                                        "handle_unknown": "error",
                                        "drop": None,
                                    },
                                },
                            ],
                        },
                    ],
                    "remainder": "passthrough",
                    "remainder_columns": [3],
                },
            }
        ],
        "estimator": {
            "kind": "linear",
            "n_features_in": 5,
            "weights": [[1.0, 0.0, 0.0, 1.0, 2.0]],
            "intercepts": [0.5],
            "classes": [],
            "nodes": [],
            "trees": [],
        },
    }


def load_fixture(rel: str) -> dict:
    return json.loads((FIXTURES / rel).read_text())


BASES = [
    {"id": "v1-iris", "schema_version": 1, "file": "../fixtures/iris.json"},
    {"id": "v1-binary", "schema_version": 1, "file": "../fixtures/binary.json"},
    {"id": "v1-diabetes", "schema_version": 1, "file": "../fixtures/diabetes.json"},
    {"id": "v2-tree-smoke", "schema_version": 2, "file": "../fixtures/tree-smoke.json"},
    {"id": "v3-hetero-logreg", "schema_version": 3, "file": "../fixtures/v3/v3-ct-hetero-logreg.json"},
    {"id": "v3-synthetic", "schema_version": 3, "file": "bases/v3-synthetic.json"},
]

# (id, schema_version, base id, mutate(doc), expected error substring)
MUTATIONS = []


def mut(id, schema, base, fn, expect):
    MUTATIONS.append(
        {"id": id, "schema_version": schema, "base": base, "file": f"mutations/{id}.json", "expect_error_contains": expect, "_fn": fn}
    )


def v1_mutations():
    iris = "v1-iris"
    mut("v1-bad-provenance", 1, iris, lambda d: d.update(source="xgboost"), "artifact: invalid provenance")
    mut("v1-empty-source-version", 1, iris, lambda d: d.update(source_version=""), "artifact: invalid provenance")
    mut("v1-empty-model-id", 1, iris, lambda d: d.update(model_id=""), "artifact: invalid provenance")
    mut("v1-unsupported-schema-version", 1, iris, lambda d: d.update(schema_version=9), "artifact: unsupported schema version")
    mut("v1-unsupported-preprocessing", 1, iris, lambda d: d.update(imputer_kind="median"), "artifact: unsupported preprocessing")
    mut("v1-invalid-preprocessing-dimensions", 1, iris, lambda d: d.update(imputer_means=[0.0]), "artifact: invalid preprocessing dimensions")
    mut("v1-empty-feature-name", 1, iris, lambda d: d["features"].__setitem__(0, ""), "artifact: empty feature name")
    mut("v1-duplicate-feature-name", 1, iris, lambda d: d["features"].__setitem__(1, d["features"][0]), "artifact: duplicate feature name")
    mut("v1-zero-scaler-scale", 1, iris, lambda d: d["scaler_scales"].__setitem__(0, 0.0), "artifact: invalid preprocessing parameters")
    mut("v1-invalid-class-order", 1, iris, lambda d: d.update(classes=[2, 1, 0]), "artifact: invalid class order")
    mut("v1-invalid-multinomial-dimensions", 1, iris, lambda d: d.update(weights=d["weights"][:2]), "artifact: invalid multinomial dimensions")
    mut("v1-unsupported-estimator", 1, iris, lambda d: d.update(estimator_kind="svm"), "artifact: unsupported estimator")


def v2_mutations():
    tree = "v2-tree-smoke"
    mut("v2-unsupported-schema-or-estimator", 2, tree, lambda d: d.update(schema_version=5), "artifact: unsupported tree schema or estimator")
    mut("v2-bad-provenance", 2, tree, lambda d: d.update(model_id=""), "artifact: invalid provenance")

    def dup_feature(d):
        d["features"] = ["x", "x"]
        d["imputer_means"] = [0.0, 0.0]
        d["scaler_means"] = [0.0, 0.0]
        d["scaler_scales"] = [1.0, 1.0]

    mut("v2-duplicate-feature-name", 2, tree, dup_feature, "artifact: duplicate feature name")
    mut("v2-invalid-child-edge", 2, tree, lambda d: d["nodes"][0].update(left=0), "tree: invalid child edge")


def v3_mutations():
    syn = "v3-synthetic"
    mut("v3-bad-provenance", 3, syn, lambda d: d.update(source="xgboost"), "artifact: invalid provenance")
    mut("v3-empty-feature-list", 3, syn, lambda d: d.update(features=[]), "artifact: empty feature list")
    mut("v3-duplicate-feature-name", 3, syn, lambda d: d["features"].__setitem__(1, d["features"][0]), "artifact: duplicate feature name")
    mut("v3-input-kind-count-mismatch", 3, syn, lambda d: d.update(input_kinds=["numeric", "numeric"]), "artifact: input kind count mismatch")
    mut("v3-unsupported-input-kind", 3, syn, lambda d: d["input_kinds"].__setitem__(2, "ordinal"), "artifact: unsupported input kind")
    mut("v3-step-input-dimension-mismatch", 3, syn, lambda d: d["steps"][0].update(n_features_in=3), "artifact: step input dimension mismatch")
    mut("v3-unsupported-step-kind", 3, syn, lambda d: d["steps"][0].update(kind="mystery"), "artifact: unsupported pipeline step kind")

    def zero_scale(d):
        d["steps"][0]["column_transformer"]["branches"][0]["steps"][0]["scaler"]["scales"][0] = 0.0

    mut("v3-invalid-scaler-parameters", 3, syn, zero_scale, "artifact: invalid scaler parameters")

    def scaler_on_categorical(d):
        d["steps"] = [
            {
                "kind": "standard_scaler",
                "n_features_in": 4,
                "n_features_out": 4,
                "scaler": {"means": [0.0] * 4, "scales": [1.0] * 4},
            }
        ]
        d["estimator"]["n_features_in"] = 4

    mut("v3-scaler-on-categorical-input", 3, syn, scaler_on_categorical, "artifact: scaler on categorical input")

    def remainder_wrong_order(d):
        d["steps"][0]["column_transformer"]["remainder_columns"] = [2]

    mut("v3-remainder-not-ordered-complement", 3, syn, remainder_wrong_order, "artifact: remainder columns are not the ordered branch complement")

    def remainder_wrong_set(d):
        d["steps"][0]["column_transformer"]["remainder_columns"] = [0, 3]

    mut("v3-remainder-not-complement", 3, syn, remainder_wrong_set, "artifact: remainder columns are not the branch complement")

    def terminal_not_numeric(d):
        d["steps"] = []
        d["estimator"]["n_features_in"] = 4
        d["estimator"]["weights"] = [[1.0, 0.0, 0.0, 1.0]]

    mut("v3-terminal-estimator-input-not-numeric", 3, syn, terminal_not_numeric, "artifact: terminal estimator input is not numeric")

    def ohe_zero_width(d):
        ohe = d["steps"][0]["column_transformer"]["branches"][1]["steps"][1]["one_hot"]
        ohe["columns"][0]["strings"] = ["x"]
        ohe["drop"] = "first"

    mut("v3-one-hot-zero-width", 3, syn, ohe_zero_width, "artifact: one-hot output has zero width")

    mut("v3-unsupported-estimator", 3, syn, lambda d: d["estimator"].update(kind="svm"), "artifact: unsupported estimator")
    mut("v3-estimator-input-dimension-mismatch", 3, syn, lambda d: d["estimator"].update(n_features_in=99), "artifact: estimator input dimension mismatch")

    def nested_ct(d):
        d["steps"][0]["column_transformer"]["branches"][0]["steps"].append(
            {"kind": "column_transformer", "n_features_in": 2, "n_features_out": 2, "column_transformer": {"branches": [], "remainder": "drop", "remainder_columns": []}}
        )

    mut("v3-nested-column-transformer", 3, syn, nested_ct, "artifact: nested column transformer")

    def passthrough_categorical(d):
        branch = d["steps"][0]["column_transformer"]["branches"][1]
        branch["mode"] = "passthrough"
        branch["steps"] = []

    mut("v3-passthrough-branch-on-categorical", 3, syn, passthrough_categorical, "artifact: passthrough branch on categorical input")
    mut("v3-real-fixture-bad-provenance", 3, "v3-hetero-logreg", lambda d: d.update(source="other"), "artifact: invalid provenance")


def main():
    v1_mutations()
    v2_mutations()
    v3_mutations()
    bases = {b["id"]: (synthetic_v3_base() if b["id"] == "v3-synthetic" else load_fixture(b["file"].replace("../fixtures/", ""))) for b in BASES}

    (OUT / "bases").mkdir(parents=True, exist_ok=True)
    (OUT / "mutations").mkdir(parents=True, exist_ok=True)
    (OUT / "bases" / "v3-synthetic.json").write_text(json.dumps(bases["v3-synthetic"], indent=2) + "\n")

    catalog = {"schema": 1, "bases": [{k: b[k] for k in ("id", "schema_version", "file")} for b in BASES], "mutations": []}
    for m in MUTATIONS:
        doc = copy.deepcopy(bases[m["base"]])
        m["_fn"](doc)
        (OUT / m["file"]).write_text(json.dumps(doc, indent=2) + "\n")
        catalog["mutations"].append({k: m[k] for k in ("id", "schema_version", "file", "expect_error_contains")})

    (OUT / "invalid_mutations.json").write_text(json.dumps(catalog, indent=2) + "\n")
    print(f"wrote {len(catalog['bases'])} bases and {len(catalog['mutations'])} mutations under {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
