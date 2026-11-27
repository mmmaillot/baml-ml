# Adaptive and shadow on a 4,096-row batch

Apple M1 Pro, 32 GiB RAM, macOS 27.2.1 (26B131), arm64. baml-cli 0.26.0, Python 3.12.1.

UTC run time 2026-11-27T19:06:20.067913+00:00. Native executable 34,703,058 bytes. Tool and runtime source hashes are in [metadata.json](metadata.json).

Packed Iris multinomial predict-proba. Seeded rows around one Iris input give 0 anomalies (healthy), 52 missing values (sparse, 1.27%), or a large feature shift in all 4,096 rows (shifted). Adaptive and shadow use the default cap of 64. Shadow uses two handles for the same artifact to hold prediction and trigger rates constant.

Medians across 7 fresh processes per case after one warmup, with shuffled case order. Wall time includes process startup, input/output and native recording flush. Peak RSS is measured for each child. BTEL and CAS are logical file bytes in a fresh home before querying. Full captures the whole main batch. Replayed rows count additional single-row passes. [Raw samples](samples.jsonl).

| Batch | Mode | Replayed rows | BTEL bytes | CAS bytes | Peak RSS MiB | Wall ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| healthy | semantic | 0 | 19,086 | 797 | 54.14 | 152.82 |
| healthy | adaptive | 0 | 27,885 | 797 | 56.84 | 161.43 |
| healthy | full | 0 | 19,393 | 833,271 | 68.75 | 163.09 |
| healthy | shadow | 0 | 31,026 | 2,276 | 59.05 | 199.76 |
| sparse | semantic | 0 | 19,083 | 797 | 56.38 | 154.22 |
| sparse | adaptive | 52 | 55,017 | 115,962 | 55.97 | 241.72 |
| sparse | full | 0 | 19,390 | 1,058,608 | 68.12 | 161.61 |
| sparse | shadow | 52 | 89,579 | 205,886 | 64.02 | 298.23 |
| shifted | semantic | 0 | 19,086 | 797 | 54.38 | 152.46 |
| shifted | adaptive | 64 | 60,530 | 141,934 | 56.48 | 257.90 |
| shifted | full | 0 | 19,393 | 833,271 | 68.05 | 164.46 |
| shifted | shadow | 64 | 102,007 | 252,158 | 62.94 | 320.85 |

The harness checked exact champion prediction parity, expected replay indices and [SQL stage pairs and failure evidence](sql-evidence.json) on separate small fixtures.
