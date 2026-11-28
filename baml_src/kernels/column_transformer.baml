// Column transformer branch schema, sub-pipeline validation and row slicing.
class ColumnTransformerBranch {
  name string
  columns int[]
  mode string // "drop" | "passthrough" | "pipeline"
  steps PipelineStep[]
}


class ColumnTransformerStep {
  branches ColumnTransformerBranch[]
  remainder string // "drop" | "passthrough"
  remainder_columns int[] // resolved to explicit indices at export time
}


function BranchStepsError(steps: PipelineStep[], n_in: int, kinds: string[]) -> string {
  let expected = n_in;
  let sub_kinds = kinds;
  for (let k = 0; k < steps.length(); k += 1) {
    let s = steps[k];
    if (s.kind == "column_transformer") { return "artifact: nested column transformer"; }
    let err = PipelineStepError(s, expected, sub_kinds);
    if (err != "") { return err; }
    expected = s.n_features_out;
    if (s.kind == "one_hot_encoder") {
      sub_kinds = NumericKinds(expected);
    }
  }
  for (let j = 0; j < sub_kinds.length(); j += 1) {
    if (sub_kinds[j] != "numeric") { return "artifact: branch pipeline output is not numeric"; }
  }
  ""
}


function SubRow(row: Cell[], columns: int[]) -> Cell[] {
  let out: Cell[] = [];
  for (let c = 0; c < columns.length(); c += 1) {
    out.push(row[columns[c]]);
  }
  out
}

// Internal: validated step state. Selects explicit column subsets, runs
// each branch (drop/passthrough/sub-pipeline) and concatenates the branch
// outputs horizontally in branch order, then the resolved remainder columns.

