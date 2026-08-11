#!/usr/bin/env python3
"""Generate reports/figures/slurm-vs-gke.svg — InferenceX bare-metal Slurm stack vs our
GKE stack, layer by layer. Rows align horizontally so each difference reads straight
across; the shared-hardware strip at the bottom anchors what is identical."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'reports', 'figures', 'slurm-vs-gke.svg')

BLUE, ORANGE = '#2a78d6', '#eb6834'
SURF, TEXT, MUTED, GRID = '#fcfcfb', '#52514e', '#898781', '#e1e0d9'
BOLD = ' font-weight="600"'

W, H = 900, 560
LX, LW = 168, 348          # left column (Slurm)
RX, RW = 532, 348          # right column (GKE)
ROW_Y0, ROW_H, ROW_GAP = 92, 56, 10

ROWS = [
    ('Launcher', 'srt-slurm launch scripts', 'sbatch/srun per run',
     'run-point-v2.sh pipeline', 'gates · filler · record · assert'),
    ('Admission', 'Scheduler request queue', 'requests held, batched entry',
     'Eager admission', 'K8s Service → Dynamo frontends'),
    ('Workers', 'Processes on long-lived nodes', 'state persists across runs',
     'Pods (Deployments/StatefulSets)', 'fresh filesystem per deploy'),
    ('IMEX / MNNVL', 'Slurm prolog configures IMEX', 'node-level, out of band',
     'DRA ComputeDomain', 'IMEX channels injected per claim'),
    ('KV transport', 'MNNVL only (custom UCX build)', '/usr/local/ucx, tuned lanes',
     'MNNVL (NIXL wheel) or GPUDirect RDMA', 'DraNet NICs · UCX rev4 lane fix'),
    ('JIT caches', 'Warm across runs', 'persistent node disk, free', 'hostPath cache + filler pass',
     'assertion stamps VALID/SUSPECT'),
]


def box(x, y, w, h, label, sub, stroke):
    return '\n'.join([
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#ffffff" stroke="{stroke}" stroke-width="1.2" rx="6"/>',
        f'<text x="{x+w/2}" y="{y+22}" font-size="11" fill="{TEXT}" text-anchor="middle"{BOLD}>{label}</text>',
        f'<text x="{x+w/2}" y="{y+38}" font-size="9.5" fill="{MUTED}" text-anchor="middle">{sub}</text>'])


p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
     'font-family="system-ui, -apple-system, Segoe UI, sans-serif">',
     f'<rect width="{W}" height="{H}" fill="{SURF}" rx="8"/>',
     f'<text x="24" y="26" font-size="14" fill="{TEXT}"{BOLD}>'
     'Bare-metal Slurm (InferenceX) vs GKE (ours) — same hardware, different platform stack</text>',
     f'<text x="{LX+LW/2}" y="72" font-size="12.5" fill="{ORANGE}" text-anchor="middle"{BOLD}>InferenceX — bare-metal Slurm</text>',
     f'<text x="{RX+RW/2}" y="72" font-size="12.5" fill="{BLUE}" text-anchor="middle"{BOLD}>Ours — GKE A4X Max</text>']

y = ROW_Y0
for cat, ltxt, lsub, rtxt, rsub in ROWS:
    p.append(f'<text x="{LX-12}" y="{y+ROW_H/2+4}" font-size="10" fill="{MUTED}" text-anchor="end"{BOLD}>{cat}</text>')
    p.append(box(LX, y, LW, ROW_H, ltxt, lsub, ORANGE))
    p.append(box(RX, y, RW, ROW_H, rtxt, rsub, BLUE))
    y += ROW_H + ROW_GAP

# shared-hardware anchor strip
p.append(f'<rect x="{LX}" y="{y+6}" width="{RX+RW-LX}" height="52" fill="none" stroke="{GRID}" stroke-width="1.6" rx="8"/>')
p.append(f'<text x="{(LX+RX+RW)/2}" y="{y+27}" font-size="11" fill="{TEXT}" text-anchor="middle"{BOLD}>'
         'Identical below this line</text>')
p.append(f'<text x="{(LX+RX+RW)/2}" y="{y+44}" font-size="9.5" fill="{MUTED}" text-anchor="middle">'
         'GB300 NVL72 (18 nodes × 4 GPU) · NVFP4 models · sglang engine + vendored per-point flags · sa-bench 2×warmup + 10×measured</text>')
p.append('</svg>')
open(OUT, 'w').write('\n'.join(p))
print(f'wrote {os.path.relpath(OUT)}')
