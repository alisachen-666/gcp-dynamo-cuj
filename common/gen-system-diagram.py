#!/usr/bin/env python3
"""Generate reports/figures/system-dynamo-disagg-gb300.svg — the e2e system diagram.

Layout: client/frontend/control-plane column on the left; NVL72 domain box with prefill
and decode worker groups separated by a wide center channel that carries the two KV-path
arrows (labels sized to stay inside the channel — no box overlap); NIC strip and GKE
platform strip below the workers. Rev2 (user feedback): record-quality pipeline strip
removed; KV labels aligned into the channel.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'reports', 'figures', 'system-dynamo-disagg-gb300.svg')

BLUE, ORANGE = '#2a78d6', '#eb6834'
SURF, TEXT, MUTED, GRID = '#fcfcfb', '#52514e', '#898781', '#e1e0d9'
BOLD = ' font-weight="600"'


def box(x, y, w, h, label, sub='', stroke=MUTED, bold=True, fs=11):
    s = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#ffffff" stroke="{stroke}" stroke-width="1.2" rx="6"/>']
    cy = y + 17 if sub else y + h / 2 + 4
    weight = BOLD if bold else ''
    s.append(f'<text x="{x+w/2}" y="{cy}" font-size="{fs}" fill="{TEXT}" text-anchor="middle"{weight}>{label}</text>')
    if sub:
        for i, ln in enumerate(sub.split('|')):
            s.append(f'<text x="{x+w/2}" y="{y+31+i*13}" font-size="9.5" fill="{MUTED}" text-anchor="middle">{ln}</text>')
    return '\n'.join(s)


def arrow(x1, y1, x2, y2, color, dash=''):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="1.8"{d} marker-end="url(#ah-{color[1:]})"/>')


def ctext(x, y, txt, color, fs=9.5, bold=False):
    w = BOLD if bold else ''
    return f'<text x="{x}" y="{y}" font-size="{fs}" fill="{color}" text-anchor="middle"{w}>{txt}</text>'


W, H = 900, 420
CH_L, CH_R = 452, 646          # center channel between prefill and decode boxes
CH_C = (CH_L + CH_R) / 2
markers = ''.join(
    f'<marker id="ah-{c[1:]}" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">'
    f'<polygon points="0 0, 9 3.5, 0 7" fill="{c}"/></marker>' for c in (BLUE, ORANGE, MUTED))
p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
     'font-family="system-ui, -apple-system, Segoe UI, sans-serif">',
     f'<defs>{markers}</defs>',
     f'<rect width="{W}" height="{H}" fill="{SURF}" rx="8"/>',
     f'<text x="24" y="26" font-size="14" fill="{TEXT}"{BOLD}>'
     'Dynamo disaggregated serving on GB300 NVL72 — e2e system (GKE A4X Max)</text>']

# left column
p.append(box(24, 60, 150, 58, 'Bench client', 'sa-bench Job|2×warmup + 10× measured'))
p.append(box(24, 172, 150, 58, 'Frontend', 'Dynamo ingress pods|K8s Service :8000'))
p.append(box(24, 304, 150, 58, 'Control plane', 'etcd — worker registry|NATS — request plane'))
p.append(arrow(99, 118, 99, 170, MUTED))
p.append(f'<text x="106" y="148" font-size="9.5" fill="{TEXT}">HTTP /v1/completions</text>')
p.append(arrow(99, 230, 99, 302, MUTED))
p.append(f'<text x="106" y="270" font-size="9.5" fill="{TEXT}">register / route</text>')

# NVL72 domain
p.append(f'<rect x="220" y="52" width="656" height="330" fill="none" stroke="{GRID}" stroke-width="1.6" rx="10"/>')
p.append(f'<text x="234" y="72" font-size="11" fill="{MUTED}"{BOLD}>'
         'NVL72 scale-up domain — 18× GB300 nodes (4 GPU/node)</text>')

# worker boxes flank the channel
p.append(box(240, 92, 212, 112, 'Prefill workers',
             'sglang disagg mode=prefill|DEP4 groups per recipe|8k-token prompt prefill|NVFP4 + DeepGEMM JIT', stroke=BLUE))
p.append(box(CH_R, 92, 212, 112, 'Decode workers',
             'sglang disagg mode=decode|DEP48 / DEP32 wide-EP|DeepEP all-to-all (NVLink)|token generation', stroke=BLUE))

# KV paths inside the channel — labels centered in the channel, never over the boxes
p.append(ctext(CH_C, 116, 'KV path A — MNNVL (NVLink)', ORANGE, bold=True))
p.append(ctext(CH_C, 128, 'cuda_ipc / mooncake', ORANGE))
p.append(arrow(CH_L + 6, 138, CH_R - 6, 138, ORANGE, dash='5 4'))
p.append(ctext(CH_C, 166, 'KV path B — GPUDirect RDMA', BLUE, bold=True))
p.append(ctext(CH_C, 178, 'rc_mlx5, NIC-offloaded', BLUE))
p.append(arrow(CH_L + 6, 188, CH_R - 6, 188, BLUE))

# NIC strip + platform strip
p.append(box(240, 236, 618, 54, '8× CX-7 mrdma NICs per node (DraNet DRA claims) — 8-rail RoCE fabric, disjoint /64 per rail',
             'UCX: NET_DEVICES=mlx5_0..7 (ordered) · GID_INDEX=5 · rail-aware same-subnet pairing · TLS=cuda_copy,rc_x,tcp',
             stroke=BLUE, bold=False, fs=10))
p.append(arrow(346, 206, 346, 234, BLUE))
p.append(arrow(752, 206, 752, 234, BLUE))
p.append(box(240, 312, 618, 52, 'GKE platform layer',
             'DRA ComputeDomain → IMEX channels (MNNVL) · hostPath DeepGEMM JIT cache · a4x-max metal node pool',
             bold=False, fs=10))

# request dispatch + token stream
p.append(arrow(176, 201, 238, 150, MUTED))
p.append(f'<text x="234" y="168" font-size="9.5" fill="{TEXT}" text-anchor="end">dispatch (NATS)</text>')
p.append(arrow(CH_R, 84, 176, 84, MUTED))
p.append(ctext(430, 80, 'streamed tokens back to client', TEXT))

p.append('</svg>')
open(OUT, 'w').write('\n'.join(p))
print(f'wrote {os.path.relpath(OUT)}')
