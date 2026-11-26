// Synthetic deterministic workloads. No logging or tracing calls in bodies.
function default_leaf(x: int) -> int { x + 1 }
function default_middle(x: int) -> int { default_leaf(x) * 2 }
function DefaultNested() -> int { default_middle(20) }

/// baml:$trace=trace.span
function span_leaf(x: int) -> int { x + 1 }
/// baml:$trace=trace.span
function span_middle(x: int) -> int { span_leaf(x) * 2 }
/// baml:$trace=trace.span
function SpanNested() -> int { span_middle(20) }

/// baml:$trace=trace.timing
function timing_leaf(x: int) -> int { x + 1 }
/// baml:$trace=trace.timing
function timing_middle(x: int) -> int { timing_leaf(x) * 2 }
function TimingNested() -> int { timing_middle(20) }

/// baml:$trace=trace.empty_span
function empty_leaf(x: int) -> int { x + 1 }
/// baml:$trace=trace.empty_span
function empty_middle(x: int) -> int { empty_leaf(x) * 2 }
function EmptyNested() -> int { empty_middle(20) }

function DefaultLoop() -> int {
    let mut total = 0;
    for (let i in 0..10000) { total = total + default_leaf(i); }
    total
}
function TimingLoop() -> int {
    let mut total = 0;
    for (let i in 0..10000) { total = total + timing_leaf(i); }
    total
}
function SpanLoop() -> int {
    let mut total = 0;
    for (let i in 0..10000) { total = total + span_leaf(i); }
    total
}

/// baml:$trace=trace.span
function matrix_echo(x: float[][]) -> float[][] { x }

function MatrixSmall() -> float[][] { matrix_echo([[1.0, 2.0], [3.0, 4.0]]) }
function MatrixLarge() -> int {
    let mut x: float[][] = [];
    for (let i in 0..100) {
        let mut row: float[] = [];
        for (let j in 0..100) { row.push(1.25); }
        x.push(row);
    }
    let y = matrix_echo(x);
    y.length()
}

/// baml:$trace=trace.span
function fail_leaf(x: int[]) -> int { x[5] }
/// baml:$trace=trace.span
function Failure() -> int { fail_leaf([1]) }
