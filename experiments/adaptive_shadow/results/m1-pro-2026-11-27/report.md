# Adaptive and shadow on a 4,096-row batch

Apple M1 Pro, 32 GiB RAM, macOS 27.2.1 (26B131), arm64. baml-cli 0.26.0, Python 3.12.1.

UTC run time 2026-11-27T21:03:42.587066+00:00. Native executable 34,703,058 bytes. Tool and runtime source hashes are in [metadata.json](metadata.json).

Packed Iris multinomial predict-proba. Seeded rows around one Iris input give 0 anomalies (healthy), 52 missing values (sparse, 1.27%), or a large feature shift in all 4,096 rows (shifted). Adaptive and shadow use the default cap of 64. Shadow uses two handles for the same artifact to hold prediction and trigger rates constant.

Medians across 7 fresh processes per case after one warmup, with shuffled case order. Wall time includes process startup, input/output and native recording flush. Peak RSS is measured for each child. BTEL and CAS are logical file bytes in a fresh home before querying. Full captures the whole main batch. Replayed rows count additional single-row passes. [Raw samples](samples.jsonl).

| Batch | Mode | Replayed rows | BTEL bytes | CAS bytes | Peak RSS MiB | Wall ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| healthy | semantic | 0 | 19,088 | 797 | 56.69 | 151.68 |
| healthy | adaptive | 0 | 28,224 | 797 | 56.89 | 159.12 |
| healthy | full | 0 | 19,395 | 833,271 | 70.44 | 160.30 |
| healthy | shadow | 0 | 31,006 | 2,276 | 61.19 | 196.81 |
| sparse | semantic | 0 | 19,085 | 797 | 57.14 | 149.73 |
| sparse | adaptive | 52 | 55,672 | 116,621 | 58.27 | 238.20 |
| sparse | full | 0 | 19,392 | 1,058,608 | 70.22 | 159.06 |
| sparse | shadow | 52 | 89,559 | 205,886 | 64.20 | 290.96 |
| shifted | semantic | 0 | 19,088 | 797 | 56.81 | 147.52 |
| shifted | adaptive | 64 | 61,185 | 142,593 | 59.03 | 258.20 |
| shifted | full | 0 | 19,395 | 833,271 | 70.38 | 160.73 |
| shifted | shadow | 64 | 101,987 | 252,158 | 64.89 | 316.83 |

The harness checked exact champion prediction parity, expected replay indices and [SQL stage pairs and failure evidence](sql-evidence.json) on separate small fixtures.
