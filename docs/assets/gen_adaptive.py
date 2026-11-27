"""Generate docs/assets/adaptive-observability.svg from measured README numbers."""
import pathlib

OUT = pathlib.Path(__file__).with_name("adaptive-observability.svg")
ROWS = 4096
TRIGGERED = set(range(0, ROWS, 80))  # experiments/adaptive_shadow/run.py sparse batch
assert len(TRIGGERED) == 52

W, H = 960, 490
p = []
a = p.append
a(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
  'font-family="-apple-system, BlinkMacSystemFont, \'Segoe UI\', Helvetica, Arial, sans-serif">')
a('<title>Adaptive observation: record everything only for the rows that deserve it</title>')
a('<style>.k{font-size:12px;fill:#57606a}.h{font-size:14px;font-weight:700;fill:#1f2328}'
  '.v{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;fill:#1f2328}'
  '.b{font-weight:700}</style>')
a(f'<rect x="0.5" y="0.5" width="{W-1}" height="{H-1}" rx="14" fill="#ffffff" stroke="#d0d7de"/>')
a('<text x="480" y="32" text-anchor="middle" font-size="18" font-weight="700" fill="#1f2328">'
  'Adaptive observation — full detail only for the rows that deserve it</text>')

# ---- Left: the batch grid (one cell per row) ----
gx, gy, cell, gap = 28, 78, 4, 1
a(f'<text x="{gx}" y="64" class="h">A real 4,096-row batch, one square per row</text>')
step = cell + gap
a(f'<defs><pattern id="g" x="{gx}" y="{gy}" width="{step}" height="{step}" patternUnits="userSpaceOnUse">'
  f'<rect width="{cell}" height="{cell}" fill="#d8dee4"/></pattern></defs>')
a(f'<rect x="{gx}" y="{gy}" width="{64*step}" height="{64*step}" fill="url(#g)"/>')
for i in sorted(TRIGGERED):
    x = gx + (i % 64) * step
    y = gy + (i // 64) * step
    a(f'<rect x="{x-1}" y="{y-1}" width="{cell+2}" height="{cell+2}" fill="#e16f24"/>')
ly = gy + 64 * (cell + gap) + 22
a(f'<rect x="{gx}" y="{ly-10}" width="10" height="10" fill="#d8dee4"/>'
  f'<text x="{gx+16}" y="{ly}" class="k"><tspan class="b" fill="#1f2328">4,044 healthy rows</tspan> → Semantic: metadata-only spans</text>')
a(f'<rect x="{gx}" y="{ly+12}" width="10" height="10" fill="#fb8f44"/>'
  f'<text x="{gx+16}" y="{ly+22}" class="k"><tspan class="b" fill="#bc4c00">52 rows with a missing value</tspan> → replayed in Full, stage by stage</text>')

# ---- Right: measured storage bars ----
rx = 390
a(f'<text x="{rx}" y="64" class="h">Trace storage for that batch (BTEL + CAS, measured)</text>')
maxv = 1_078_000
barw = 400


def bar(y, label, value, color, txt, bold=False):
    w = max(2, barw * value / maxv)
    a(f'<text x="{rx}" y="{y+13}" class="k">{label}</text>')
    a(f'<rect x="{rx+90}" y="{y}" width="{w:.1f}" height="18" rx="4" fill="{color}"/>')
    weight = ' class="v b"' if bold else ' class="v"'
    a(f'<text x="{rx+90+w+8:.1f}" y="{y+13}"{weight}>{txt}</text>')


a(f'<text x="{rx}" y="96" class="k b" fill="#1f2328">Sparse batch (52 anomalies)</text>')
bar(108, "full", 1_078_000, "#8c959f", "1,078 kB")
bar(134, "adaptive", 172_293, "#fb8f44", "172 kB  (−84%)", bold=True)

a(f'<text x="{rx}" y="190" class="k b" fill="#1f2328">Healthy batch (0 anomalies)</text>')
bar(202, "full", 852_666, "#8c959f", "853 kB")
bar(228, "adaptive", 29_021, "#fb8f44", "29 kB", bold=True)
bar(254, "semantic", 19_885, "#d8dee4", "20 kB")

# Trade-off box
a(f'<rect x="{rx}" y="296" width="542" height="98" rx="10" fill="#fff8c5" stroke="#d4a72c"/>')
a(f'<text x="{rx+16}" y="320" class="k b" fill="#1f2328">The honest trade-off</text>')
a(f'<text x="{rx+16}" y="342" class="k">Each replayed row emits its own stage spans: on the sparse batch, adaptive</text>')
a(f'<text x="{rx+16}" y="360" class="k">took <tspan class="v">238 ms</tspan> vs <tspan class="v">159 ms</tspan> for full. <tspan class="v">max_escalated_rows</tspan> caps both</text>')
a(f'<text x="{rx+16}" y="378" class="k">storage and latency (64 here).</text>')

a(f'<text x="480" y="{H-18}" text-anchor="middle" class="k">Median of 7 fresh processes, predict-proba on Iris, Apple M1 Pro. '
  'Source: experiments/adaptive_shadow (report m1-pro-2026-11-27).</text>')
a('</svg>')
OUT.write_text("\n".join(p) + "\n")
print(OUT, OUT.stat().st_size)
