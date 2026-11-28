// One-hot encoding: fitted category schema, validation and dense expansion core.
class OneHotColumn {
  kind string // "string" | "integer"
  strings string[]
  ints float[]
}


class OneHotEncoderStep {
  columns OneHotColumn[]
  handle_unknown string // "error" | "ignore"
  drop string? // null (sklearn None) | "first" | "if_binary"
}

// One named branch of a column transformer: explicit validated input column
// indices plus "drop", "passthrough" (numeric columns only) or a nested
// sequential sub-pipeline of supported preprocessors (no nested transformer).

function OneHotBlockSize(col: OneHotColumn, drop: string?) -> int {
  let n = if (col.kind == "string") { col.strings.length() } else { col.ints.length() };
  let adj = if (drop == "first" || (drop == "if_binary" && n == 2)) { 1 } else { 0 };
  n - adj
}


function OneHotColumnError(col: OneHotColumn) -> string {
  let n = if (col.kind == "string") { col.strings.length() } else { col.ints.length() };
  if (col.kind != "string" && col.kind != "integer") { return "artifact: unsupported one-hot category kind"; }
  if (n == 0) { return "artifact: empty one-hot categories"; }
  if (col.kind == "string") {
    if (col.ints.length() != 0) { return "artifact: malformed one-hot string categories"; }
    for (let i = 0; i < n; i += 1) {
      for (let k = 0; k < i; k += 1) {
        if (col.strings[i] == col.strings[k]) { return "artifact: duplicate one-hot category"; }
      }
    }
  } else {
    if (col.strings.length() != 0) { return "artifact: malformed one-hot integer categories"; }
    for (let i = 0; i < n; i += 1) {
      if (!IsFinite(col.ints[i]) || col.ints[i] != col.ints[i].floor() || Abs(col.ints[i]) >= 9007199254740992.0) { return "artifact: invalid one-hot integer category"; }
      for (let k = 0; k < i; k += 1) {
        if (col.ints[i] == col.ints[k]) { return "artifact: duplicate one-hot category"; }
      }
    }
  }
  ""
}

// Validate a nested sequential sub-pipeline (inside a column transformer
// branch). Nested column transformers are rejected: unsupported, not approximated.

function OneHotStepCore(cells: Cell[][], ohe: OneHotEncoderStep) -> CellMatrixResult {
  let n = ohe.columns.length();
  let err = CheckCellMatrixDims(cells, n);
  let out: Cell[][] = [];
  if (err == "") {
    for (let i = 0; i < cells.length() && err == ""; i += 1) {
      let row: Cell[] = [];
      for (let j = 0; j < n && err == ""; j += 1) {
        let col = ohe.columns[j];
        let block = OneHotBlockSize(col, ohe.drop);
        let idx = -1;
        if (col.kind == "string" && cells[i][j].kind == "string") {
          for (let k = 0; k < col.strings.length(); k += 1) {
            if (cells[i][j].str == col.strings[k]) { idx = k; }
          }
        } else if (col.kind == "integer" && cells[i][j].kind == "number") {
          for (let k = 0; k < col.ints.length(); k += 1) {
            if (cells[i][j].num == col.ints[k]) { idx = k; }
          }
        }
        if (idx < 0 && ohe.handle_unknown == "error") {
          err = "one_hot_encoding: unseen category in column " + j.to_string();
        } else {
          // if_binary drops only when the original category count is two.
          let cat_count = if (col.kind == "string") { col.strings.length() } else { col.ints.length() };
          let drop_adj = if (ohe.drop == "first" || (ohe.drop == "if_binary" && cat_count == 2)) { 1 } else { 0 };
          for (let k = 0; k < block; k += 1) {
            let on = idx >= drop_adj && (idx - drop_adj) == k;
            row.push(Cell { kind: "number", num: if (on) { 1.0 } else { 0.0 }, str: "" });
          }
        }
      }
      if (err == "") { out.push(row); }
    }
  }
  if (err == "") {
    CellMatrixResult { ok: true, cells: out, error: "" }
  } else {
    CellMatrixResult { ok: false, cells: [], error: err }
  }
}


