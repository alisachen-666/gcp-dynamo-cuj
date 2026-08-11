#!/usr/bin/env python3
"""Generate Pareto-curve SVGs (ours vs InferenceX) into reports/figures/.

Sets produced:
  pareto-dsr1-rdma-{interactivity,e2e-latency,ttft}.svg  — DSR1-FP4, ours = RDMA arm (rev4)
  pareto-dsv4-{interactivity,e2e-latency,ttft}.svg       — DSv4-Pro, ours = optimal finals

Conventions (match pareto-curves.md + the summary tables):
  Y      = total (input+output) throughput per GPU  = total_token_throughput / total_gpus
  X (1)  = interactivity (tok/s/user) = 1000 / mean TPOT ms   [linear axis]
  X (2)  = median E2E latency (s)                              [log axis]
  X (3)  = median TTFT (s)                                     [log axis]
Style: house palette (#2a78d6 ours / #eb6834 InferenceX on #fcfcfb), 2px lines,
surface-filled ring markers, recessive grid, legend + selective conc labels.
Pure stdlib — no plotting deps.
"""
import importlib.util
import json
import math
import os

DSR1_PT = {(4, 16, 4): 'p1', (4, 16, 8): 'p2', (4, 16, 32): 'p3', (4, 16, 64): 'p4',
           (24, 48, 512): 'p5', (24, 48, 2048): 'p6', (24, 48, 4096): 'p7', (40, 32, 2048): 'p8'}
DSV4_PT = {(4, 4): 'p1', (4, 24): 'p2', (4, 8): 'p3', (4, 16): 'p4',
           (8, 8): 'p5', (16, 8): 'p6', (24, 8): 'p7', (32, 8): 'p8'}

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
FIGS = os.path.join(ROOT, 'reports', 'figures')

BLUE, ORANGE = '#2a78d6', '#eb6834'
SURF, TEXT, MUTED, GRID = '#fcfcfb', '#52514e', '#898781', '#e1e0d9'
W, H = 760, 420
L, R, T, B = 56, 736, 44, 384


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def dsr1_series():
    d1 = _mod('cmp1', os.path.join(ROOT, 'dsr1-sweep', 'compare_inferencex_dsr1.py'))
    ix = [dict(conc=r['conc'], tput=r['total_per_gpu'], inter=1000.0 / r['mean_tpot_ms'],
               e2el=r['median_e2el_ms'] / 1000, ttft=r['median_ttft_ms'] / 1000)
          for r in d1.load_csv(os.path.join(ROOT, 'dsr1-sweep', 'reference',
                                            'inferencex-published-dsr1-fp4-gb300.csv'))]
    ours = []
    import glob
    for f in sorted(glob.glob(os.path.join(ROOT, 'dsr1-sweep', 'results-summary', '*.json'))):
        b = os.path.basename(f)
        if not b.startswith(('llr-', 'm4r-', 'mxr-')) or 'INVALID' in b:
            continue
        d = json.load(open(f))
        tot = 20 if b.startswith('llr-') else 72
        pgdg = (4, 16) if b.startswith('llr-') else ((40, 32) if b.startswith('mxr-') else (24, 48))
        ours.append(dict(conc=d['max_concurrency'], pid=DSR1_PT.get(pgdg + (d['max_concurrency'],), ''),
                         tput=d['total_token_throughput'] / tot,
                         inter=1000.0 / d['mean_tpot_ms'], e2el=d['median_e2el_ms'] / 1000,
                         ttft=d['median_ttft_ms'] / 1000))
    return ix, ours


def dsv4_series():
    d4 = _mod('cmp4', os.path.join(ROOT, 'dsv4-sweep', 'compare_inferencex.py'))
    theirs, mine = d4.load_theirs(), d4.load_ours()
    ix, ours = [], []
    p8lat = None
    f8 = os.path.join(ROOT, 'dsv4-sweep', 'results-summary', 'p8-rerun-latency-only.extracted.json')
    if os.path.exists(f8):
        p8lat = json.load(open(f8))
    for r in theirs:
        ix.append(dict(conc=r['conc'], tput=r['total_per_gpu'], inter=1000.0 / r['mean_tpot_ms'],
                       e2el=r['median_e2el_ms'] / 1000, ttft=r['median_ttft_ms'] / 1000))
        key = (r['conc'], r['pg'], r['dg'])
        rec = mine.get(key)
        # p8's drift-run row is superseded by the 2026-08-07 measured record (+ the
        # latency-parity rerun) — never plot the drifted MTP-3/4 run
        if key == (8192, 32, 8) and p8lat and p8lat.get('_measured_throughput_record_20260807'):
            rec = None
        if rec:
            pid, d, drift = rec
            ours.append(dict(conc=r['conc'], pid=DSV4_PT.get((r['pg'], r['dg']), ''),
                             tput=d['total_token_throughput'] / r['total_gpus'],
                             inter=1000.0 / d['mean_tpot_ms'], e2el=d['median_e2el_ms'] / 1000,
                             ttft=d['median_ttft_ms'] / 1000))
        elif key == (8192, 32, 8) and p8lat:
            m = p8lat.get('_measured_throughput_record_20260807')
            if m:
                ours.append(dict(conc=8192, pid='p8', tput=m['total_per_gpu'],
                                 inter=1000.0 / p8lat['mean_tpot_ms'],
                                 e2el=p8lat['median_e2el_ms'] / 1000,
                                 ttft=p8lat['median_ttft_ms'] / 1000))
    return ix, ours


def _nice_ticks(lo, hi, n=5):
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    t0 = math.ceil(lo / step) * step
    out = []
    v = t0
    while v <= hi + 1e-9:
        out.append(round(v, 6))
        v += step
    return out


def _log_ticks(lo, hi):
    out = []
    d = math.floor(math.log10(lo))
    while 10 ** d <= hi * 1.001:
        for m in (1, 3):
            v = m * 10 ** d
            if lo * 0.999 <= v <= hi * 1.001:
                out.append(v)
        d += 1
    return out or [lo, hi]


def _fmt(v):
    if v >= 1000:
        return f'{v/1000:g}k' if v % 1000 == 0 else f'{v:,.0f}'
    return f'{v:g}'


def chart(path, title, xlabel, ix, ours, xkey, log_x, ours_label):
    xs = [p[xkey] for p in ix + ours]
    ys = [p['tput'] for p in ix + ours]
    if log_x:
        xlo, xhi = min(xs) / 1.35, max(xs) * 1.35
        xmap = lambda v: L + (math.log10(v) - math.log10(xlo)) / (math.log10(xhi) - math.log10(xlo)) * (R - L)
        xticks = _log_ticks(min(xs), max(xs))
    else:
        xlo, xhi = 0, max(xs) * 1.12
        xmap = lambda v: L + (v - xlo) / (xhi - xlo) * (R - L)
        xticks = _nice_ticks(xlo, xhi)
    ylo, yhi = 0, max(ys) * 1.12
    ymap = lambda v: B - (v - ylo) / (yhi - ylo) * (B - T)

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'font-family="system-ui, -apple-system, Segoe UI, sans-serif">',
         f'<rect width="{W}" height="{H}" fill="{SURF}" rx="8"/>',
         f'<text x="{L}" y="20" font-size="12.5" fill="{TEXT}" font-weight="600">{title}</text>']
    # legend (top-right, stacked rows — no width guessing, no collisions)
    lx = R - 168
    for i, (col, lab) in enumerate(((BLUE, ours_label), (ORANGE, 'InferenceMax (published)'))):
        y = 14 + i * 15
        s.append(f'<line x1="{lx}" x2="{lx+18}" y1="{y}" y2="{y}" stroke="{col}" stroke-width="2"/>')
        s.append(f'<circle cx="{lx+9}" cy="{y}" r="3.2" fill="{SURF}" stroke="{col}" stroke-width="1.8"/>')
        s.append(f'<text x="{lx+22}" y="{y+4}" font-size="10.5" fill="{TEXT}">{lab}</text>')
    # grid + axes
    for v in _nice_ticks(ylo, yhi):
        y = ymap(v)
        s.append(f'<line x1="{L}" x2="{R}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{L-6}" y="{y+3.5:.1f}" font-size="10.5" fill="{MUTED}" text-anchor="end">{_fmt(v)}</text>')
    for v in xticks:
        x = xmap(v)
        s.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{B}" y2="{B+4}" stroke="{MUTED}" stroke-width="1"/>')
        s.append(f'<text x="{x:.1f}" y="{B+16}" font-size="10.5" fill="{MUTED}" text-anchor="middle">{_fmt(v)}</text>')
    s.append(f'<line x1="{L}" x2="{R}" y1="{B}" y2="{B}" stroke="{MUTED}" stroke-width="1"/>')
    s.append(f'<text x="{(L+R)/2}" y="{H-6}" font-size="11" fill="{TEXT}" text-anchor="middle">{xlabel}</text>')
    # series: line through points sorted by x, ring markers, conc labels.
    # Labels use greedy collision avoidance: try offsets until the label's box
    # overlaps neither a previously placed label nor any data marker.
    placed = []          # (x0, y0, x1, y1) boxes of placed labels
    markers = []         # marker centers, both series
    for pts in (ix, ours):
        markers += [(xmap(p[xkey]), ymap(p['tput'])) for p in pts]

    def _label(x, y, txt):
        w, h = 5.6 * len(txt) + 2, 10.0
        for dx, dy in ((6, -6), (6, 13), (-w - 6, -6), (-w - 6, 13), (6, -16), (-w - 6, -16),
                       (6, 24), (-w - 6, 24), (6, -26), (-w - 6, -26), (10, 4), (-w - 14, 4)):
            bx0, by1 = x + dx, y + dy
            by0 = by1 - h
            bx1 = bx0 + w
            if bx0 < L or bx1 > R or by0 < T or by1 > B:
                continue
            if any(bx0 < px1 and bx1 > px0 and by0 < py1 and by1 > py0
                   for px0, py0, px1, py1 in placed):
                continue
            if any(bx0 - 3 < mx < bx1 + 3 and by0 - 3 < my < by1 + 3 for mx, my in markers):
                continue
            placed.append((bx0, by0, bx1, by1))
            return f'<text x="{bx0:.1f}" y="{by1:.1f}" font-size="9.5" fill="{MUTED}">{txt}</text>'
        return ''  # no clean slot — drop the label rather than overlap (selective labeling)

    for pts, col in ((ix, ORANGE), (ours, BLUE)):
        pp = sorted(pts, key=lambda p: p[xkey])
        poly = ' '.join(f'{xmap(p[xkey]):.1f},{ymap(p["tput"]):.1f}' for p in pp)
        s.append(f'<polyline points="{poly}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round"/>')
        for p in pp:
            x, y = xmap(p[xkey]), ymap(p['tput'])
            s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" fill="{SURF}" stroke="{col}" stroke-width="1.8"/>')
            if col == BLUE:  # label ours only — selective, avoids double-labeling every x
                cc = f'c{_fmt(p["conc"])}'
                txt = f'{p["pid"]}({cc})' if p.get('pid') else cc
                lab = _label(x, y, txt)
                if lab:
                    s.append(lab)
    s.append('</svg>')
    open(path, 'w').write('\n'.join(s))
    print(f'wrote {os.path.relpath(path, ROOT)} ({len(ix)} IX pts, {len(ours)} ours pts)')


def main():
    os.makedirs(FIGS, exist_ok=True)
    for tag, (ix, ours), model, olab in (
            ('dsr1-rdma', dsr1_series(), 'DeepSeek-R1 FP4 — KV over GPUDirect RDMA', 'Ours: GKE, RDMA KV'),
            ('dsv4', dsv4_series(), 'DeepSeek-V4-Pro FP4', 'Ours: GKE (finals)')):
        chart(os.path.join(FIGS, f'pareto-{tag}-interactivity.svg'),
              f'{model} — total throughput per GPU (tok/s)',
              'Interactivity (tok/s/user, 1000 / mean TPOT)', ix, ours, 'inter', False, olab)
        chart(os.path.join(FIGS, f'pareto-{tag}-e2e-latency.svg'),
              f'{model} — total throughput per GPU (tok/s)',
              'Median end-to-end latency (s, log scale)', ix, ours, 'e2el', True, olab)
        chart(os.path.join(FIGS, f'pareto-{tag}-ttft.svg'),
              f'{model} — total throughput per GPU (tok/s)',
              'Median time-to-first-token (s, log scale)', ix, ours, 'ttft', True, olab)


if __name__ == '__main__':
    main()
