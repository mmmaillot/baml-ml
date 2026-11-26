// Distinct values expose truncation/capture and content-addressed storage.
function PayloadScalar() -> int { span_middle(20) }
function PayloadMatrix() -> int {
    let x: float[][] = [];
    let value = 0.125;
    for (let i = 0; i < 100; i += 1) {
        let row: float[] = [];
        for (let j = 0; j < 100; j += 1) {
            row.push(value);
            value = value + 0.125;
        }
        x.push(row);
    }
    let y = matrix_echo(x);
    y.length()
}
