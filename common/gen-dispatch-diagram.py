#!/usr/bin/env python3
"""Generate reports/figures/ft-dispatch-flow.svg — frontend traffic dispatch order in
Dynamo disaggregated serving, as implemented in ai-dynamo v0.8.1 (verified in source:
lib/llm/src/kv_router/prefill_router.rs + components/src/dynamo/sglang/request_handlers).
Sequence-diagram style: four lanes, numbered steps, prefill-first dispatch."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'reports', 'figures', 'ft-dispatch-flow.svg')

BLUE, ORANGE = '#2a78d6', '#eb6834'
SURF, TEXT, MUTED, GRID = '#fcfcfb', '#52514e', '#898781', '#e1e0d9'
BOLD = ' font-weight="600"'

W, H = 900, 470
LANES = [('Client', 100), ('Frontend', 330, 'PrefillRouter (Rust)'),
         ('Prefill worker', 590, 'prefill_handler'), ('Decode worker', 810, 'decode_handler')]


def lane(name, x, sub=''):
    s = [f'<rect x="{x-72}" y="46" width="144" height="{40 if sub else 30}" fill="#ffffff" stroke="{MUTED}" stroke-width="1.2" rx="6"/>',
         f'<text x="{x}" y="{65 if sub else 66}" font-size="11" fill="{TEXT}" text-anchor="middle"{BOLD}>{name}</text>']
    if sub:
        s.append(f'<text x="{x}" y="{79}" font-size="8.5" fill="{MUTED}" text-anchor="middle">{sub}</text>')
    s.append(f'<line x1="{x}" y1="{88}" x2="{x}" y2="{H-46}" stroke="{GRID}" stroke-width="1.2" stroke-dasharray="4 4"/>')
    return '\n'.join(s)


def msg(x1, x2, y, num, label, color=MUTED, dash='', lift=6):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    mid = (x1 + x2) / 2
    return '\n'.join([
        f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" stroke-width="1.8"{d} marker-end="url(#ah-{color[1:]})"/>',
        f'<circle cx="{x1 + (14 if x2 > x1 else -14)}" cy="{y-14}" r="8.5" fill="{color}"/>',
        f'<text x="{x1 + (14 if x2 > x1 else -14)}" y="{y-10.5}" font-size="9.5" fill="#ffffff" text-anchor="middle"{BOLD}>{num}</text>',
        f'<text x="{mid}" y="{y-lift}" font-size="9.5" fill="{color if color != MUTED else TEXT}" text-anchor="middle">{label}</text>'])


markers = ''.join(
    f'<marker id="ah-{c[1:]}" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">'
    f'<polygon points="0 0, 9 3.5, 0 7" fill="{c}"/></marker>' for c in (BLUE, ORANGE, MUTED))
p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
     'font-family="system-ui, -apple-system, Segoe UI, sans-serif">',
     f'<defs>{markers}</defs>',
     f'<rect width="{W}" height="{H}" fill="{SURF}" rx="8"/>',
     f'<text x="24" y="26" font-size="14" fill="{TEXT}"{BOLD}>'
     'Frontend dispatch order — Dynamo disaggregated serving (prefill-first, from v0.8.1 source)</text>']

for entry in LANES:
    p.append(lane(entry[0], entry[1], entry[2] if len(entry) > 2 else ''))

CL, FE, PF, DE = 100, 330, 590, 810
p.append(msg(CL, FE - 4, 122, '1', 'HTTP POST /v1/completions'))
p.append(msg(FE, PF - 4, 164, '2', 'prefill request (NATS request plane)'))
p.append(f'<text x="{PF+8}" y="186" font-size="9" fill="{MUTED}">prefill engine starts</text>')
p.append(f'<text x="{PF+8}" y="197" font-size="9" fill="{MUTED}">8k-prompt compute</text>')
p.append(msg(PF - 4, FE, 214, '3', 'bootstrap_info {host, port, room}', dash='5 4'))
p.append(msg(FE, DE - 4, 256, '4', 'generate + injected bootstrap_info'))
p.append(f'<text x="{DE-160}" y="276" font-size="9" fill="{MUTED}">decode joins room, preallocates KV</text>')
p.append(msg(PF, DE - 4, 306, '5', 'KV cache transfer — path A (MNNVL) or path B (RDMA)', color=BLUE, lift=8))
p.append(f'<text x="{(PF+DE)/2}" y="318" font-size="9" fill="{MUTED}" text-anchor="middle">engine-to-engine, once both sides joined the bootstrap room</text>')
p.append(msg(DE - 4, CL + 4, 356, '6', 'streamed output tokens (via frontend)', color=ORANGE))
p.append(f'<text x="24" y="{H-24}" font-size="9.5" fill="{MUTED}">'
         'Source: lib/llm/src/kv_router/prefill_router.rs ("calls a prefill worker before routing to decode"); '
         'decode_handler.generate requires pre-computed bootstrap_info.</text>')
p.append('</svg>')
open(OUT, 'w').write('\n'.join(p))
print(f'wrote {os.path.relpath(OUT)}')
