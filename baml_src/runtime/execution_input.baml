// Keep direct numeric input in its original representation. Cell input is used
// by heterogeneous pipelines and by the row-level Adaptive and shadow paths.
class NumericExecutionInput {
  rows float[][]
}

type ExecutionInput = NumericExecutionInput | Cell[][]


function ExecutionFloatMatrix(input: ExecutionInput) -> float[][] throws InferenceFailure {
  if (input is NumericExecutionInput) { return input.rows; }
  CellsToFloatMatrix(input)
}
