"""Export key curves as static SVGs for GitHub rendering (reports/img/).

Light-theme, explicit background (GitHub README context). Three exports:
  knee.svg      — TTFT p95 vs concurrency per policy (log y), 3:3
  frontier.svg  — efficient hull, tok/s/user x tok/s/GPU per policy, 3:3
  aic_agg.svg   — AIC SILICON pareto, aggregated vs disaggregated (warm)
Data: dynosim_sweep_v4+v5 CSVs and AIC pareto CSVs in aic-results/.
"""
import csv
import math
from pathlib import Path

AR = Path.home() / "kv-cache-aware-bench/aic-results"
IMG = Path.home() / "kv-cache-aware-bench/reports/img"
IMG.mkdir(parents=True, exist_ok=True)
W, H, M = 760, 440, dict(t=28, r=20, b=52, l=64)
COL = {"rr": "#2a78d6", "kv-nvda": "#eb6834", "kv-t": "#1baf7a", "ll": "#898781",
       "agg": "#eb6834", "disagg": "#1baf7a"}
NAME = {"rr": "Round-robin", "kv-nvda": "KV defaults", "kv-t": "KV tuned",
        "ll": "Least-loaded", "agg": "AIC aggregated", "disagg": "AIC disaggregated"}
INK, INK2, MUT, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def fam(p):
    return "kv-t" if p.startswith("kv-t") else p


def cells():
    out = {}
    for f in ["dynosim_sweep_v4.csv", "dynosim_sweep_v5.csv"]:
        for r in csv.DictReader(open(AR / f)):
            k = (r["pd"], int(r["conc"]), fam(r["policy"]))
            if k not in out or f.endswith("v5.csv") or \
               float(r["throughput_tok_s"]) > float(out[k]["throughput_tok_s"]):
                out[k] = r
    return out


def hull(pts):
    pts = sorted(pts, key=lambda p: (-p[0], -p[1]))
    front, by = [], -1e18
    for p in pts:
        if p[1] > by:
            front.append(p)
            by = p[1]
    return sorted(front)


def svg_open(title, xlabel, ylabel):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
            f'font-family="system-ui,sans-serif">',
            f'<rect width="{W}" height="{H}" fill="{SURF}"/>',
            f'<text x="{M["l"]}" y="18" font-size="14" font-weight="600" fill="{INK}">{title}</text>',
            f'<text x="{(M["l"]+W-M["r"])/2}" y="{H-8}" text-anchor="middle" font-size="12" fill="{INK2}">{xlabel}</text>',
            f'<text x="16" y="{(M["t"]+H-M["b"])/2}" text-anchor="middle" font-size="12" fill="{INK2}" '
            f'transform="rotate(-90 16 {(M["t"]+H-M["b"])/2})">{ylabel}</text>']


def axis(g, X, Y, xticks, yticks, xfmt=str, yfmt=str):
    for v in xticks:
        g.append(f'<line x1="{X(v):.1f}" y1="{M["t"]}" x2="{X(v):.1f}" y2="{H-M["b"]}" stroke="{GRID}"/>')
        g.append(f'<text x="{X(v):.1f}" y="{H-M["b"]+16}" text-anchor="middle" font-size="11" fill="{MUT}">{xfmt(v)}</text>')
    for v in yticks:
        g.append(f'<line x1="{M["l"]}" y1="{Y(v):.1f}" x2="{W-M["r"]}" y2="{Y(v):.1f}" stroke="{GRID}"/>')
        g.append(f'<text x="{M["l"]-7}" y="{Y(v)+4:.1f}" text-anchor="end" font-size="11" fill="{MUT}">{yfmt(v)}</text>')


def legend(g, keys, x=None, y=None):
    x = x or M["l"] + 8
    y = y or M["t"] + 10
    for i, k in enumerate(keys):
        g.append(f'<rect x="{x}" y="{y + i*18 - 8}" width="10" height="10" rx="2" fill="{COL[k]}"/>')
        g.append(f'<text x="{x+15}" y="{y + i*18 + 1}" font-size="11.5" fill="{INK2}">{NAME[k]}</text>')


def line(g, pts, color, dash=""):
    d = "".join(f"{'L' if i else 'M'}{x:.1f} {y:.1f}" for i, (x, y) in enumerate(pts))
    g.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.2"{dash}/>')
    for x, y in pts:
        g.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="{color}"/>')


C = cells()

# ---- knee.svg
series = {}
for (pd, conc, f), r in sorted(C.items(), key=lambda kv: kv[0][1]):
    if pd == "3:3":
        series.setdefault(f, []).append((conc, float(r["ttft_p95_s"])))
concs = sorted({c for pts in series.values() for c, _ in pts})
lx = lambda c: M["l"] + (math.log(c/concs[0]) / math.log(concs[-1]/concs[0])) * (W-M["l"]-M["r"])
vals = [v for pts in series.values() for _, v in pts]
lo, hi = math.log10(max(0.05, min(vals))), math.log10(max(vals)*1.4)
ly = lambda v: H-M["b"] - ((math.log10(max(0.05, v))-lo)/(hi-lo)) * (H-M["t"]-M["b"])
g = svg_open("TTFT p95 vs concurrency — the capacity knee (3:3, DynoSim)",
             "concurrency (log scale)", "TTFT p95 (s, log scale)")
axis(g, lx, ly, concs, [0.1, 1, 5, 10, 100],
     yfmt=lambda v: f"{v:g}s")
g.append(f'<line x1="{M["l"]}" y1="{ly(5):.1f}" x2="{W-M["r"]}" y2="{ly(5):.1f}" stroke="{MUT}" stroke-dasharray="5 4"/>')
g.append(f'<text x="{W-M["r"]-4}" y="{ly(5)-5:.1f}" text-anchor="end" font-size="10.5" fill="{MUT}">recipe goodput 5 s</text>')
for f in ["rr", "ll", "kv-nvda", "kv-t"]:
    if f in series:
        line(g, [(lx(c), ly(v)) for c, v in series[f]], COL[f])
legend(g, ["rr", "ll", "kv-nvda", "kv-t"])
g.append("</svg>")
(IMG / "knee.svg").write_text("\n".join(g))

# ---- frontier.svg (hull only)
cloud = {}
for (pd, conc, f), r in C.items():
    if pd == "3:3":
        cloud.setdefault(f, []).append((1000/float(r["tpot_mean_ms"]),
                                        float(r["throughput_tok_s"])/24))
xs = [x for pts in cloud.values() for x, _ in pts]
ys = [y for pts in cloud.values() for _, y in pts]
fx = lambda v: M["l"] + (v/(max(xs)*1.08)) * (W-M["l"]-M["r"])
fy = lambda v: H-M["b"] - (v/(max(ys)*1.1)) * (H-M["t"]-M["b"])
g = svg_open("Efficiency frontier — efficient hull per policy (3:3, DynoSim)",
             "tokens/s per user (decode rate)", "output tokens/s per GPU")
axis(g, fx, fy, [0, 20, 40, 60], [0, 40, 80, 120])
for f in ["rr", "ll", "kv-nvda", "kv-t"]:
    if f in cloud:
        line(g, [(fx(x), fy(y)) for x, y in hull(cloud[f])], COL[f])
legend(g, ["rr", "ll", "kv-nvda", "kv-t"])
g.append("</svg>")
(IMG / "frontier.svg").write_text("\n".join(g))

# ---- aic_agg.svg
aic = {}
for arch in ("agg", "disagg"):
    f = AR / f"silicon-warm-{arch}-pareto.csv"
    if f.exists():
        aic[arch] = [(float(r["tokens/s/user"]), float(r["tokens/s/gpu"]))
                     for r in csv.DictReader(open(f))]
xs = [x for pts in aic.values() for x, _ in pts]
ys = [y for pts in aic.values() for _, y in pts]
ax_ = lambda v: M["l"] + (v/(max(xs)*1.08)) * (W-M["l"]-M["r"])
ay_ = lambda v: H-M["b"] - (v/(max(ys)*1.12)) * (H-M["t"]-M["b"])
g = svg_open("AIC SILICON pareto — aggregated vs disaggregated (GB300, warm prefix)",
             "tokens/s per user (decode rate)", "output tokens/s per GPU")
axis(g, ax_, ay_, [0, 20, 40, 60, 80], [0, 25, 50, 75])
for arch in ("agg", "disagg"):
    if arch in aic:
        line(g, [(ax_(x), ay_(y)) for x, y in hull(aic[arch])], COL[arch])
legend(g, ["agg", "disagg"], x=W-230)
g.append("</svg>")
(IMG / "aic_agg.svg").write_text("\n".join(g))

print("wrote", *[p.name for p in IMG.glob("*.svg")])
