"""Generate docs/assets/cold-start-latency.svg from the README standalone benchmark table."""
import pathlib

OUT = pathlib.Path(__file__).with_name("cold-start-latency.svg")
DATA = {  # median ms, fresh process per batch (README / m1-pro-2026-11-27.md)
    "1 row": (32.0, 110.6, 898.3),
    "128 rows": (33.2, 111.7, 895.0),
    "4,096 rows": (64.7, 148.9, 895.4),
}
SERIES = (("baml-ml, telemetry off", "#1a7f37"),
          ("baml-ml, semantic traces", "#4ac26b"),
          ("Python + scikit-learn", "#8c959f"))
W, H = 960, 380
x0, barmax, maxv = 130, 700, 900
p = []
a = p.append
a(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
  'font-family="-apple-system, BlinkMacSystemFont, \'Segoe UI\', Helvetica, Arial, sans-serif">')
a('<title>Latency of one batch launched as a fresh process</title>')
a('<style>.k{font-size:12px;fill:#57606a}.g{font-size:13px;font-weight:700;fill:#1f2328}'
  '.v{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;fill:#1f2328}</style>')
a(f'<rect x="0.5" y="0.5" width="{W-1}" height="{H-1}" rx="14" fill="#ffffff" stroke="#d0d7de"/>')
a('<text x="480" y="32" text-anchor="middle" font-size="18" font-weight="700" fill="#1f2328">'
  'One batch = one process: end-to-end latency (lower is better)</text>')
# legend
lx = 150
for name, color in SERIES:
    a(f'<rect x="{lx}" y="48" width="12" height="12" rx="2" fill="{color}"/>'
      f'<text x="{lx+18}" y="58" class="k">{name}</text>')
    lx += 230
y = 84
for group, values in DATA.items():
    a(f'<text x="{x0-14}" y="{y+34}" text-anchor="end" class="g">{group}</text>')
    for (name, color), v in zip(SERIES, values):
        w = barmax * v / maxv
        a(f'<rect x="{x0}" y="{y}" width="{w:.1f}" height="18" rx="4" fill="{color}"/>')
        a(f'<text x="{x0+w+8:.1f}" y="{y+13}" class="v">{v:,.1f} ms</text>')
        y += 22
    y += 18
a(f'<text x="480" y="{H-38}" text-anchor="middle" class="k">Median of 15 runs, predict-proba on the Iris multinomial model, Apple M1 Pro; '
  'includes startup, model load, JSON I/O and shutdown.</text>')
a(f'<text x="480" y="{H-18}" text-anchor="middle" class="k">With both models already loaded in memory, scikit-learn is faster from 128 rows up (see table below).</text>')
a('</svg>')
OUT.write_text("\n".join(p) + "\n")
print(OUT, OUT.stat().st_size)
