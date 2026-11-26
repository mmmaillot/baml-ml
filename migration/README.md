# BAML language migration (phase 1)

Isolated port to BAML language 0.25.0. The historical project and NumBam are unchanged. Baseline: [v0.1.0](https://github.com/mmmaillot/baml-ml/tree/v0.1.0).

Use the exact Linux CLI asset and SHA-256 from [migration CI](../.github/workflows/baml-migration.yml), extracted from the [official release](https://github.com/BoundaryML/baml/releases/tag/baml-language-0.25.0). The version manifest returned 404 during the interop POC, so CI verifies the release archive directly. With `BAML_CLI` set to the extracted executable's absolute path:

```bash
cd migration
"$BAML_CLI" check
"$BAML_CLI" test --list
"$BAML_CLI" test --max-concurrency 1
```

## Primitive compatibility

- Replace handwritten Exp, Log, Sqrt, Abs, IsNaN and IsFinite with [native float methods](https://github.com/BoundaryML/baml/blob/baml-language-0.25.0/baml_language/crates/baml_builtins2/baml_std/baml/float.baml).
- New language float equality is reflexive even for NaN, and comparisons have a total order. Use the predicate wrappers before ordering; never detect NaN with `x != x`.
- Retain the old finite negative Log sentinel for nonpositive values and Sqrt clamp to positive zero. NaN propagates.
- Native Exp intentionally improves subnormal behavior: e.g. exp(-720) is positive, whereas the old reciprocal implementation underflowed early. Platform math may differ in last bits.
- Retain matrix shape checks, result classes, ordered dot, scaled norm and corrected two-pass variance; no NumBam calls or solver changes.
- Port all 18 original primitive test cases to native BAML tests, plus Exp and Log checks against f64 reference values. No full estimator migration or SDK parity claim in this ticket.

CI validation pending. Follow [#25](https://github.com/mmmaillot/baml-ml/issues/25).
