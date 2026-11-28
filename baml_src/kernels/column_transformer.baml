// Column transformer row slicing (tabular operation).
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


