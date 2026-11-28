"""Generate the deterministic heterogeneous campaign batch (500 rows, ~5% nulls).

Usage: python generate_batch.py <output_batch.json>
The distribution matches the v3-ct-hetero-forest training fixture.
"""
import json
import sys

import numpy as np

rng = np.random.default_rng(20260627)
n = 500
age = rng.uniform(20.0, 60.0, n)
income = rng.uniform(30.0, 90.0, n) * 1000.0
cities = np.array(["paris", "lyon", "berlin", "madrid"], dtype=object)[rng.integers(0, 4, n)]
rows = []
for i in range(n):
    row = [float(age[i]), float(income[i]), str(cities[i])]
    for j in range(3):
        if rng.random() < 0.05:
            row[j] = None
    rows.append(row)
batch = {"features": ["age", "income", "city"], "rows": rows}
with open(sys.argv[1] if len(sys.argv) > 1 else "batch.json", "w", encoding="utf-8") as f:
    json.dump(batch, f)
    f.write("\n")
print(f"wrote {sys.argv[1] if len(sys.argv) > 1 else 'batch.json'} with {n} rows")
