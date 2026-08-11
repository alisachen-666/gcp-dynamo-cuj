#!/usr/bin/env python3
"""Generate reports/figures/bench-flow.svg — the per-point e2e benchmarking flow
(the record-quality pipeline) as a serpentine stage diagram with failure branches."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'reports', 'figures', 'bench-flow.svg')

BLUE, ORANGE = '#2a78d6', '#eb6834'
SURF, TEXT, MUTED, GRID = '#fcfcfb', '#52514e', '#898781', '#e1e0d9'
BOLD = ' font-weight="600"'

W, H = 900, 470
BW, BH, GAP = 156, 66, 24
X0, Y1, Y2, Y3 = 24, 64, 220, 366
COLS = [X0 + i * (BW + GAP) for i in range(5)]


def box(x, y, label, sub, stroke=BLUE):
    s = [f'<rect x="{x}" y="{y}" width="{BW}" height="{BH}" fill="#ffffff" stroke="{stroke}" stroke-width="1.2" rx="6"/>',
         f'<text x="{x+BW/2}" y="{y+20}" font-size="10.5" fill="{TEXT}" text-anchor="middle"{BOLD}>{label}</text>']
    for i, ln in enumerate(sub.split('|')):
        s.append(f'<text x="{x+BW/2}" y="{y+34+i*12}" font-size="9" fill="{MUTED}" text-anchor="middle">{ln}</text>')
    return '\n'.join(s)


def arrow(x1, y1, x2, y2, color=MUTED):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="1.8" marker-end="url(#ah-{color[1:]})"/>')


markers = ''.join(
    f'<marker id="ah-{c[1:]}" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">'
    f'<polygon points="0 0, 9 3.5, 0 7" fill="{c}"/></marker>' for c in (BLUE, ORANGE, MUTED))
p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
     'font-family="system-ui, -apple-system, Segoe UI, sans-serif">',
     f'<defs>{markers}</defs>',
     f'<rect width="{W}" height="{H}" fill="{SURF}" rx="8"/>',
     f'<text x="24" y="26" font-size="14" fill="{TEXT}"{BOLD}>'
     'E2E benchmarking flow — one data point, deploy to report (record-quality pipeline)</text>']

# row 1: setup + gates, left -> right
r1 = [('1 · Vendor + diff recipe', 'per-point flags from srt-slurm|recipes, diffed flag-by-flag'),
      ('2 · Deploy point', 'ComputeDomain + workers|+ frontend (kubectl apply)'),
      ('3 · Registration wait', 'etcd shows all expected|workers (poll, crash-abort)'),
      ('4 · Serving gate', '3 consecutive real|completions via Service'),
      ('5 · Transport gate (RDMA)', 'KV probe must be rc_mlx5;|+ in-bench watchdog')]
for i, (l, s) in enumerate(r1):
    p.append(box(COLS[i], Y1, l, s))
    if i:
        p.append(arrow(COLS[i-1] + BW, Y1 + BH/2, COLS[i] - 4, Y1 + BH/2))
p.append(f'<text x="{COLS[4]+BW/2}" y="{Y1+BH+16}" font-size="9" fill="{ORANGE}" text-anchor="middle"{BOLD}>TCP fallback → kill run, never recorded</text>')

# row 2: measurement, right -> left (serpentine)
r2 = [('9 · Upload + teardown', 'server/bench logs, configs,|results → GCS; delete point'),
      ('8 · Extract before teardown', 'results parsed from job log|(rotation-proof SLIM tail)'),
      ('7 · JIT/stall assertion', 'every measured window vs|prefill logs → VALID/SUSPECT'),
      ('6b · Record pass', '2×conc warmup discarded|+ 10×conc measured @ 8k1k'),
      ('6a · Cache-filler pass', 'full bench, results discarded|forces every JIT shape')]
for i, (l, s) in enumerate(r2):
    p.append(box(COLS[i], Y2, l, s))
for i in range(4, 0, -1):
    p.append(arrow(COLS[i], Y2 + BH/2, COLS[i-1] + BW + 4, Y2 + BH/2))
p.append(arrow(COLS[4] + BW/2, Y1 + BH, COLS[4] + BW/2, Y2 - 4))
p.append(f'<text x="{COLS[2]+BW/2}" y="{Y2+BH+16}" font-size="9" fill="{ORANGE}" text-anchor="middle"{BOLD}>SUSPECT → cheap re-record against the standing deployment</text>')

# row 3: reporting
p.append(f'<rect x="{X0}" y="{Y3}" width="{COLS[4]+BW-X0}" height="58" fill="#ffffff" stroke="{MUTED}" stroke-width="1.2" rx="6"/>')
p.append(f'<text x="{(X0+COLS[4]+BW)/2}" y="{Y3+22}" font-size="11" fill="{TEXT}" text-anchor="middle"{BOLD}>'
         '10 · Reports regenerate from slims — comparison tables, RDMA report, Pareto figures, e2e analysis</text>')
p.append(f'<text x="{(X0+COLS[4]+BW)/2}" y="{Y3+40}" font-size="9.5" fill="{MUTED}" text-anchor="middle">'
         'optimal-final policy: best audited variant per point (TCP/NATS × MNNVL/RDMA) becomes the summary row; all others kept as extras — every cell links logs + bench + config</text>')
p.append(arrow(X0 + BW/2, Y2 + BH, X0 + BW/2, Y3 - 4))
p.append('</svg>')
open(OUT, 'w').write('\n'.join(p))
print(f'wrote {os.path.relpath(OUT)}')
