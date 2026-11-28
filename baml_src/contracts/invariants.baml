// Single-source-of-truth static artifact invariants shared by the schema 1, 2
// and 3 contracts, plus the internal enums parsed once at wire boundaries.
// Wire JSON (artifacts, batch inputs, BatchOutput, ShadowOutput) keeps plain
// strings; enums exist only inside the process.

enum Operation {
  Predict,
  PredictProba,
}

enum ColumnKind {
  Numeric,
  Categorical,
}

function OperationName(op: Operation) -> string {
  if (op == Operation.PredictProba) { return "predict-proba"; }
  "predict"
}

// CLI/JSON entry boundary: the only place an operation string is parsed.

function ParseOperation(value: string) -> Operation {
  if (value == "predict") { return Operation.Predict; }
  if (value == "predict-proba") { return Operation.PredictProba; }
  throw baml.errors.InvalidArgument { message: "operation must be predict or predict-proba" };
}

function ColumnKindOf(value: string) -> ColumnKind? {
  if (value == "numeric") { return ColumnKind.Numeric; }
  if (value == "categorical") { return ColumnKind.Categorical; }
  null
}

function NumericColumnKinds(n: int) -> ColumnKind[] {
  let kinds: ColumnKind[] = [];
  for (let j = 0; j < n; j += 1) { kinds.push(ColumnKind.Numeric); }
  kinds
}

// Training provenance is informational, not authentication.

function ProvenanceError(source: string, source_version: string, model_id: string) -> string {
  if (source != "scikit-learn" || source_version == "" || model_id == "") { return "artifact: invalid provenance"; }
  ""
}

// Non-blank, strictly unique ordered feature names. The empty-list message is
// caller-specific (schema 1 folds it into the dimension check, schema 3 has its
// own), so it is not checked here.

function FeatureNameErrorAt(features: string[], j: int) -> string {
  if (features[j] == "") { return "artifact: empty feature name"; }
  for (let k = 0; k < j; k += 1) {
    if (features[j] == features[k]) { return "artifact: duplicate feature name"; }
  }
  ""
}

function FeatureNamesError(features: string[]) -> string {
  for (let j = 0; j < features.length(); j += 1) {
    let err = FeatureNameErrorAt(features, j);
    if (err != "") { return err; }
  }
  ""
}

// Request-time feature contract against the loaded artifact.

function LoadedFeatureError(features: string[], expected: string[]) -> string {
  if (features.length() != expected.length()) { return "artifact: feature order mismatch"; }
  for (let j = 0; j < features.length(); j += 1) {
    if (features[j] != expected[j]) { return "artifact: feature order mismatch"; }
  }
  ""
}

function SortedUniqueClasses(classes: int[]) -> bool {
  for (let k = 1; k < classes.length(); k += 1) {
    if (classes[k] <= classes[k - 1]) { return false; }
  }
  true
}

function SortedUniqueClassesMin(classes: int[], min_count: int) -> bool {
  if (classes.length() < min_count) { return false; }
  SortedUniqueClasses(classes)
}

// IsFinite everywhere: x != x only detects NaN, never +-inf.

function ScalerParamValid(mean: float, scale: float) -> bool {
  IsFinite(mean) && IsFinite(scale) && scale > 0.0
}

function FiniteMatrixParams(weights: float[][], intercepts: float[], n: int) -> string {
  if (weights.length() != intercepts.length()) { return "artifact: invalid model dimensions"; }
  for (let i = 0; i < weights.length(); i += 1) {
    if (weights[i].length() != n || !IsFinite(intercepts[i])) { return "artifact: invalid model dimensions or intercept"; }
    for (let j = 0; j < n; j += 1) {
      if (!IsFinite(weights[i][j])) { return "artifact: non-finite weight"; }
    }
  }
  ""
}
