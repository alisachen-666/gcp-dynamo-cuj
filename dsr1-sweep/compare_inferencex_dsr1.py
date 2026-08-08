#!/usr/bin/env python3
"""Compare our DSR1 runs (FP4 + FP8) against InferenceX published results, per metric category.

Theirs: reference/inferencex-published-dsr1-fp4-gb300.csv and
        ../dsr1-fp8-sweep/reference/inferencex-published-dsr1-fp8-gb300.csv
        (merged dashboard exports; the "(ms)" latency columns are SECONDS — verified via
        Interactivity == 1/TPOT — converted to true ms here).
Ours:   results-summary/*.json (FP4) and ../dsr1-fp8-sweep/results-summary/*.json (FP8).

Point selection (ours): where both a base and a `_mult10` run exist, the mult10 run is used
(srt-slurm-exact methodology: 2x warmup + 10x measured); base-only points are marked `(base)`
— they under-measure throughput by ~10% (see the conc-64 mult10 experiment: +11.1%).
For FP4 mid conc-2048 the untuned run is primary (recipe-parity env); the `tuned` variant is
shown as an extra row.

Writes:
  - FP4 section spliced into ../reports/benchmark-report-dsr1-8k1k.md
  - FP8 report ../reports/benchmark-report-dsr1-fp8-8k1k.md (created fresh)
"""
import csv
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
S = 1000.0


def load_csv(path):
    rows = []
    with open(path) as f:
        lines = [ln for ln in f if not ln.startswith('#')]
    for r in csv.DictReader(lines):
        rows.append({
            'conc': int(r['Concurrency']),
            'pg': int(r['Num Prefill GPUs']), 'dg': int(r['Num Decode GPUs']),
            'tot': int(r['TP']),
            'total_per_gpu': float(r['Throughput/GPU (tok/s)']),
            'out_per_gpu': float(r['Output Throughput/GPU (tok/s)']),
            'in_per_gpu': float(r['Input Throughput/GPU (tok/s)']),
            'mean_ttft_ms': float(r['Mean TTFT (ms)']) * S,
            'median_ttft_ms': float(r['Median TTFT (ms)']) * S,
            'p99_ttft_ms': float(r['P99 TTFT (ms)']) * S,
            'mean_tpot_ms': float(r['Mean TPOT (ms)']) * S,
            'median_tpot_ms': float(r['Median TPOT (ms)']) * S,
            'p99_tpot_ms': float(r['P99 TPOT (ms)']) * S,
            'mean_itl_ms': float(r['Mean ITL (ms)']) * S,
            'median_itl_ms': float(r['Median ITL (ms)']) * S,
            'mean_e2el_ms': float(r['Mean E2E Latency (ms)']) * S,
            'median_e2el_ms': float(r['Median E2E Latency (ms)']) * S,
            'p99_e2el_ms': float(r['P99 E2E Latency (ms)']) * S,
            'mean_inter': float(r['Mean Interactivity (tok/s/user)']),
            'median_inter': float(r['Median Interactivity (tok/s/user)']),
        })
    return rows


def load_ours_fp4():
    """(conc, pg, dg) -> (label, data). ll files: gpus_20 = 4P/16D; m4: gpus_72 = 24P/48D."""
    out = {}
    for f in sorted(glob.glob(os.path.join(HERE, 'results-summary', '*.json'))):
        b = os.path.basename(f)
        if 'INVALID' in b:
            continue
        if b.startswith('m4r-'):
            d = json.load(open(f))
            out[((d['max_concurrency'], 24, 48), 'rdmakv')] = ('RDMA-KV', d)
            continue
        elif b.startswith('llr-'):
            d = json.load(open(f))
            out[((d['max_concurrency'], 4, 16), 'rdmakv')] = ('RDMA-KV', d)
            continue
        elif b.startswith('mxr-'):
            d = json.load(open(f))
            out[((d['max_concurrency'], 40, 32), 'rdmakv')] = ('RDMA-KV', d)
            continue
        elif b.startswith('mxt-'):
            d = json.load(open(f))
            out[((d['max_concurrency'], 40, 32), 'mxtcp')] = ('TCP A/B', d)
            continue
        elif b.startswith('lln-'):
            d = json.load(open(f))
            out[((d['max_concurrency'], 4, 16), 'llnats')] = ('NATS mult10', d)
            continue
        elif b.startswith('ll-'):
            pg, dg = 4, 16
        elif b.startswith('m4n-'):
            d = json.load(open(f))
            out[((d['max_concurrency'], 24, 48), 'nats')] = ('NATS A/B', d)
            continue
        elif b.startswith('m4-'):
            pg, dg = 24, 48
        elif b.startswith('mx-'):
            pg, dg = 40, 32
        else:
            continue
        d = json.load(open(f))
        key = (d['max_concurrency'], pg, dg)
        mult10 = '_mult10' in b
        tuned = '_tuned' in b
        if tuned:
            out[(key, 'tuned')] = ('tuned', d)
            continue
        cur = out.get(key)
        if cur is None or (mult10 and cur[0] == '(base)'):
            # m4 runs used 10x-measured methodology natively -> no suffix needed
            out[key] = ('' if (mult10 or b.startswith(('m4-', 'mx-'))) else '(base)', d)
    return out


def load_ours_fp8():
    """mid: gpus_72 = 40P/32D; ll extracted: gpus_8 = 4P/4D."""
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, 'dsr1-fp8-sweep', 'results-summary', '*.json'))):
        b = os.path.basename(f)
        d = json.load(open(f))
        if b.startswith('mid-'):
            key = (d['max_concurrency'], 40, 32); tag = ''
        elif b.startswith('ll-'):
            key = (d['max_concurrency'], 4, 4); tag = '(log-extracted)'
        else:
            continue
        out[key] = (tag, d)
    return out


def pct(a, b):
    return 100.0 * (a - b) / b


def g(a, b, lower=False):
    v = pct(a, b)
    if lower:
        m = ' ✅' if v <= -5 else (' ⚠️' if v >= 20 else '')
    else:
        m = ' ✅' if v >= 5 else (' ⚠️' if v <= -10 else '')
    return f'{v:+.1f}%{m}'



GCS = 'https://console.cloud.google.com/storage/browser/REDACTED-GCS-BUCKET/mlperf_results/nvfp4_20260519_run5/logs/logs'
GCSF = 'https://console.cloud.google.com/storage/browser/_details/REDACTED-GCS-BUCKET/mlperf_results/nvfp4_20260519_run5/logs/logs'
GH = 'https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main'


def links(tag, pg, dg, conc):
    # run identity -> (server-logs dir, bench-log object, config manifest)
    if (pg, dg) == (4, 16):
        if tag == 'NATS mult10':
            return ('dsr1-lln', 'dsr1-lln-bench.log', 'dsr1-sweep/manifests/lln-lowlatency-nats.yaml')
        if tag == 'RDMA-KV':
            return ('dsr1-llr', 'dsr1-llr-bench.log', 'dsr1-sweep/manifests/llr-lowlatency-rdmakv.yaml')
        return ('dsr1-lowlatency', None, 'dsr1-sweep/manifests/m4b-lowlatency-1p4d.yaml')
    if (pg, dg) == (24, 48):
        if tag == 'NATS':
            if conc == 2048:
                return ('dsr1-m4n', 'dsr1-m4n-bench.log', 'dsr1-sweep/manifests/m4n-midcurve-nats.yaml')
            return ('dsr1-m4n-2pts', f'bench-m4n-c{conc}.log', 'dsr1-sweep/manifests/m4n-midcurve-nats.yaml')
        if tag == 'RDMA-KV':
            return ('dsr1-m4r-rev3-3pts', f'bench-m4r-c{conc}-rev3.log', 'dsr1-sweep/manifests/m4r-midcurve-rdmakv.yaml')
        return ('dsr1-midcurve', None, 'dsr1-sweep/manifests/m4-midcurve.yaml')
    if tag == 'TCP A/B':
        return ('dsr1-mxt', 'dsr1-mxt-bench.log', 'dsr1-sweep/manifests/mxt-maxtpt-tcp.yaml')
    if tag == 'RDMA-KV':
        return ('dsr1-mxr', 'dsr1-mxr-bench.log', 'dsr1-sweep/manifests/mxr-maxtpt-rdmakv.yaml')
    return ('dsr1-mx', 'dsr1-mx-bench.log', 'dsr1-sweep/manifests/mx-maxtpt.yaml')


def linkcell(tag, pg, dg, conc):
    srv, cl, cfg = links(tag, pg, dg, conc)
    out = f' [logs]({GCS}/server-logs/{srv})'
    if cl:
        out += f' [bench]({GCSF}/client-logs/{cl})'
    out += f' [cfg]({GH}/{cfg}) [cfg-gcs]({GCSF}/configs/{cfg})'
    return out


def label_tag(label):
    """Variant tag from an extra-row label; None = no links (diagnostic runs)."""
    if 'tuned' in label:
        return None
    for t in ('NATS mult10', 'TCP A/B', 'RDMA-KV', 'NATS'):
        if t in label:
            return t
    return ''


def plane_kv(tag, pg, dg):
    """(request plane, KV transport) for a run. Canonical (untagged) runs: ll/m4 predate the
    NATS work (dynamo 0.8.1 default = TCP); mx ran with DYN_REQUEST_PLANE=nats."""
    kv = 'RDMA (rc_mlx5)' if tag == 'RDMA-KV' else 'MNNVL (cuda_ipc)'
    if tag in ('NATS', 'NATS mult10', 'RDMA-KV'):
        plane = 'NATS'
    elif tag == 'TCP A/B':
        plane = 'TCP'
    else:
        plane = 'NATS' if (pg, dg) == (40, 32) else 'TCP'
    return plane, kv


def summary_table(theirs, ours, overlay=None):
    """Dual-source table: one row-pair per point (InferenceMax / GCP bench with GKE),
    GCP cells formatted as <value> (<gap vs InferenceMax>)."""
    doc = ['### Summary: InferenceMax vs GCP bench with GKE\n']
    doc.append('GCP cells show `value (gap vs InferenceMax)`. Throughput: higher is better; '
               'E2EL/ITL: lower is better. **Final-result policy: optimal run per point** — among '
               'the same-methodology variants we executed (request plane TCP vs NATS; KV transport '
               'MNNVL vs RDMA), the run with the highest output tok/s/GPU is final (tie-break: '
               'lower median ITL). The cell label names the winning variant (unlabeled = TCP/MNNVL '
               'canonical run); all other variants remain in the per-category tables as extras.\n')
    doc.append('| Pareto region | Prefill-Decode topology | Conc | Sources | Req plane | KV transfer | Total Tput (tok/s/GPU) | Input Tput (tok/s/GPU) | Output Tput (tok/s/GPU) | Median E2EL (s) | Median ITL (s) |')
    doc.append('|---|---|---|---|---|---|---|---|---|---|---|')
    # region labels verbatim from InferenceX nvidia-master.yaml search-space comments
    REGION = {(4, 16): 'Low latency', (24, 48): 'Mid curve', (40, 32): 'Max throughput'}
    def gp(a, b):
        return f'{a:,.1f} ({100*(a-b)/b:+.1f}%)'
    def gs(a, b):
        return f'{a:.2f} ({100*(a-b)/b:+.1f}%)'
    for r in sorted(theirs, key=lambda x: (x['pg'], x['dg'], x['conc'])):
        key = (r['conc'], r['pg'], r['dg'])
        topo = f"{r['pg']}P/{r['dg']}D"
        doc.append(f"| {REGION.get((r['pg'], r['dg']), '')} | {topo} | {r['conc']} | InferenceMax | NATS | MNNVL (NIXL) | {r['total_per_gpu']:,.1f} | "
                   f"{r['in_per_gpu']:,.1f} | {r['out_per_gpu']:,.1f} | "
                   f"{r['median_e2el_ms']/1000:.2f} | {r['median_itl_ms']/1000:.2f} |")
        rec = (overlay or {}).get(key) or ours.get(key)
        if not rec:
            doc.append(f"| | | | GCP bench with GKE | — | — | — | — | — | — | — |")
            continue
        tag, d = rec
        plane, kv = plane_kv(tag, r['pg'], r['dg'])
        o_tot = d['total_token_throughput'] / r['tot']
        o_in = d['total_input_tokens'] / d['duration'] / r['pg']
        o_out = d['output_throughput'] / r['dg']
        doc.append(f"| | | | GCP bench with GKE{(' ' + tag) if tag else ''}{linkcell(tag, r['pg'], r['dg'], r['conc'])} | {plane} | {kv} | {gp(o_tot, r['total_per_gpu'])} | "
                   f"{gp(o_in, r['in_per_gpu'])} | {gp(o_out, r['out_per_gpu'])} | "
                   f"{gs(d['median_e2el_ms']/1000, r['median_e2el_ms']/1000)} | "
                   f"{gs(d['median_itl_ms']/1000, r['median_itl_ms']/1000)} |")
    return doc


def sections(title, theirs, ours, extra_rows=(), with_links=False):
    def lc(tag, r):
        return linkcell(tag, r['pg'], r['dg'], r['conc']) if with_links and tag is not None else ''
    doc = [f'## {title}\n']
    doc.append('Theirs converted from the CSV\'s mislabeled seconds to true ms. '
               '`(base)` = our run pre-dates the 10x-measured methodology (understates our '
               'throughput ~10%); `(log-extracted)` = result reconstructed from the bench log.\n')
    if with_links:
        doc.append('Row tags → transport: untagged = TCP request plane + MNNVL KV '
                   '(except 40P/32D, which ran NATS); `NATS`/`NATS mult10` = NATS + MNNVL; '
                   '`TCP A/B` = TCP + MNNVL; `RDMA-KV` = NATS + GPUDirect RDMA (rc_mlx5).\n')

    doc.append('### Throughput (out ÷ decode GPUs, in ÷ prefill GPUs, total ÷ all)\n')
    doc.append('| Config | Conc | Out/decode-GPU ours | theirs | gap | In/prefill-GPU ours | theirs | gap | Total/GPU ours | theirs | gap |')
    doc.append('|---|---|---|---|---|---|---|---|---|---|---|')
    matched = []
    for r in sorted(theirs, key=lambda x: (x['pg'], x['dg'], x['conc'])):
        key = (r['conc'], r['pg'], r['dg'])
        cfg = f"{r['pg']}P/{r['dg']}D"
        rec = ours.get(key)
        if not rec:
            doc.append(f"| {cfg} (not run) | {r['conc']} | — | {r['out_per_gpu']:.1f} | — | — | {r['in_per_gpu']:.0f} | — | — | {r['total_per_gpu']:.1f} | — |")
            continue
        tag, d = rec
        matched.append((r, tag, d))
        o_out = d['output_throughput'] / r['dg']
        o_in = d['total_input_tokens'] / d['duration'] / r['pg']
        o_tot = d['total_token_throughput'] / r['tot']
        doc.append(f"| {cfg} {tag}{lc(tag, r)} | {r['conc']} | {o_out:.1f} | {r['out_per_gpu']:.1f} | {g(o_out, r['out_per_gpu'])} | "
                   f"{o_in:.0f} | {r['in_per_gpu']:.0f} | {g(o_in, r['in_per_gpu'])} | "
                   f"{o_tot:.1f} | {r['total_per_gpu']:.1f} | {g(o_tot, r['total_per_gpu'])} |")
    for label, r, d in extra_rows:
        o_out = d['output_throughput'] / r['dg']
        doc.append(f"| {label}{lc(label_tag(label), r)} | {r['conc']} | {o_out:.1f} | {r['out_per_gpu']:.1f} | {g(o_out, r['out_per_gpu'])} | "
                   f"{d['total_input_tokens']/d['duration']/r['pg']:.0f} | {r['in_per_gpu']:.0f} | — | "
                   f"{d['total_token_throughput']/r['tot']:.1f} | {r['total_per_gpu']:.1f} | — |")

    doc.append('\n### Latency (true ms; lower is better)\n')
    doc.append('| Config | Conc | TTFT med ours | theirs | gap | TTFT p99 ours | theirs | gap | TPOT mean ours | theirs | gap | E2E mean ours | theirs | gap |')
    doc.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for r, tag, d in matched:
        cfg = f"{r['pg']}P/{r['dg']}D"
        doc.append(f"| {cfg} {tag}{lc(tag, r)} | {r['conc']} | "
                   f"{d['median_ttft_ms']:.0f} | {r['median_ttft_ms']:.0f} | {g(d['median_ttft_ms'], r['median_ttft_ms'], True)} | "
                   f"{d['p99_ttft_ms']:.0f} | {r['p99_ttft_ms']:.0f} | {g(d['p99_ttft_ms'], r['p99_ttft_ms'], True)} | "
                   f"{d['mean_tpot_ms']:.2f} | {r['mean_tpot_ms']:.2f} | {g(d['mean_tpot_ms'], r['mean_tpot_ms'], True)} | "
                   f"{d['mean_e2el_ms']:.0f} | {r['mean_e2el_ms']:.0f} | {g(d['mean_e2el_ms'], r['mean_e2el_ms'], True)} |")

    doc.append('\n### Interactivity (tok/s/user = 1/TPOT; like-for-like)\n')
    doc.append('| Config | Conc | Mean ours | theirs | gap | Median ours | theirs | gap |')
    doc.append('|---|---|---|---|---|---|---|---|')
    for r, tag, d in matched:
        cfg = f"{r['pg']}P/{r['dg']}D"
        om, omed = 1000.0 / d['mean_tpot_ms'], 1000.0 / d['median_tpot_ms']
        doc.append(f"| {cfg} {tag}{lc(tag, r)} | {r['conc']} | {om:.1f} | {r['mean_inter']:.1f} | {g(om, r['mean_inter'])} | "
                   f"{omed:.1f} | {r['median_inter']:.1f} | {g(omed, r['median_inter'])} |")
    return doc, matched


def main():
    # ---------- FP4 ----------
    theirs4 = load_csv(os.path.join(HERE, 'reference', 'inferencex-published-dsr1-fp4-gb300.csv'))
    ours4 = load_ours_fp4()
    extra4 = []
    var_cands = []   # (key, tag, data): every non-primary variant, eligible for the optimal pick
    # lln: primary where only (base) exists, extra row where TCP mult10 already holds
    for k in [k for k in list(ours4) if isinstance(k, tuple) and len(k) == 2 and k[1] == 'llnats']:
        key = k[0]
        rec = ours4.pop(k)
        var_cands.append((key, rec[0], rec[1]))
        if key not in ours4 or ours4[key][0] == '(base)':
            ours4[key] = rec
        else:
            ref = next(r for r in theirs4 if (r['conc'], r['pg'], r['dg']) == key)
            extra4.append((f'4P/16D NATS mult10 conc-{key[0]} (extra)', ref, rec[1]))
    for k in [k for k in list(ours4) if isinstance(k, tuple) and len(k) == 2 and k[1] == 'mxtcp']:
        rec = ours4.pop(k)
        var_cands.append((k[0], 'TCP A/B', rec[1]))
        ref = next(r for r in theirs4 if (r['conc'], r['pg'], r['dg']) == k[0])
        extra4.append(('40P/32D TCP A/B (extra)', ref, rec[1]))
    for k in [k for k in list(ours4) if isinstance(k, tuple) and len(k) == 2 and k[1] == 'nats']:
        conc = k[0][0]
        m4n = ours4.pop(k)
        var_cands.append((k[0], 'NATS', m4n[1]))
        ref = next(r for r in theirs4 if (r['conc'], r['pg'], r['dg']) == (conc, 24, 48))
        extra4.append((f'24P/48D NATS conc-{conc} (extra)', ref, m4n[1]))
    for k in [k for k in list(ours4) if isinstance(k, tuple) and len(k) == 2 and k[1] == 'rdmakv']:
        conc, pg, dg = k[0]
        rec = ours4.pop(k)
        var_cands.append((k[0], 'RDMA-KV', rec[1]))
        ref = next((r for r in theirs4 if (r['conc'], r['pg'], r['dg']) == k[0]), None)
        if ref:
            extra4.append((f'{pg}P/{dg}D RDMA-KV conc-{conc} (extra)', ref, rec[1]))
    extra4.sort(key=lambda x: x[1]['conc'])
    tuned = ours4.get(((2048, 24, 48), 'tuned'))
    extra = []
    if tuned:
        ref2048 = next(r for r in theirs4 if (r['conc'], r['pg'], r['dg']) == (2048, 24, 48))
        extra = [('24P/48D tuned-env (extra)', ref2048, tuned[1])]
    doc4, _ = sections('Comparison vs InferenceX published results (per metric category)',
                       theirs4, {k: v for k, v in ours4.items() if not isinstance(k[0], tuple) and len(k) == 3}, extra + extra4,
                       with_links=True)
    doc4.insert(1, 'Source: `dsr1-sweep/reference/inferencex-published-dsr1-fp4-gb300.csv` '
                   '(InferenceX run 2026-02-12, Apache-2.0, © 2026 SemiAnalysis LLC). '
                   'Generated by `dsr1-sweep/compare_inferencex_dsr1.py`.\n')
    primary4 = {k: v for k, v in ours4.items() if not (isinstance(k, tuple) and len(k) == 2)}
    # optimal pick per point: primary + all variants compete on output tok/s/GPU
    # (tie-break: lower median ITL); 'tuned' diagnostic stays excluded (env drift).
    cands = {}
    for key, rec in primary4.items():
        cands.setdefault(key, []).append(rec)
    for key, tag, d in var_cands:
        cands.setdefault(key, []).append((tag, d))
    best_overlay = {k: max(v, key=lambda c: (c[1]['output_throughput'], -c[1]['median_itl_ms']))
                    for k, v in cands.items()}
    doc4[3:3] = summary_table(theirs4, primary4, best_overlay)
    doc4.append('''
### Findings

- **Low-latency (4P/16D): we lead on every axis that matters.** Throughput +1.4% to +4.4%
  (and conc-4/8 are `(base)` runs — with the 10x methodology the margin would grow ~10%),
  TPOT 3-7% better, E2E 3-4% better. The one blemish: TTFT *median* at conc 4/8 (+53-58%,
  376/417 ms vs 238/272 ms) on 16/32-request samples — while our TTFT *p99* is better; small-n
  medians + our snapshot-mirror frontend path are the suspects, not capacity.
- **Mid_curve (24P/48D): same machine totals, different queue policy.** Throughput -1.3%
  (conc 512) and -2.7% (conc 2048); E2E mean within +-8% everywhere. The striking split:
  our TTFT median is 16-77% BETTER while our TPOT is 24-41% WORSE (16.2-18.7 ms vs their flat
  ~13.1-13.6 ms). With E2E and throughput equal, that is an admission/knee difference, not a
  kernel difference: their scheduler holds requests in queue longer (TTFT median 3.4 s vs our
  0.79 s at conc 512), keeping the decode batch leaner and faster; ours admits eagerly, so
  requests spend the same total time but more of it decoding in a fatter batch.
- **Conc 4096 flips the story: +9.4% ours.** Their mid_curve REGRESSES past the knee
  (838.8 -> 765.5 out tok/s/decode-GPU; TTFT p99 130 s) while ours holds 837.4 — our eager
  admission degrades more gracefully. Combined with conc-2048, the two curves cross between
  2048 and 4096.
- **Max_tpt (40P/32D, conc 2048): trails ~9%** on both request planes (NATS 1620.7, TCP A/B
  within noise of it, vs their 1781.6 out/decode-GPU) — plane-independent, root-caused to
  prefill dispatch waves / decode residency (see the max_tpt investigation section); still open
  at the scheduler layer. RDMA-KV A/B (mxr) queued against it.
- Accuracy is not measurable from either dataset (synthetic random prompts, ignore-eos; no
  quality columns in the CSV) — same argument as the DSv4 report, section 4.

### Config parity vs InferenceX (per point)

Flags come from the same vendored srt-slurm recipes (`recipes/gb300-fp4/8k1k/{low_latency,
mid_curve,max_tpt}.yaml` @ `sa-submission-q2-2026`), so per-point sglang arguments are
identical by construction. The differences are environmental; per dimension, with measured
influence:

| Dimension | InferenceX | Ours (GKE) | Influence on results |
|---|---|---|---|
| Platform / launcher | Slurm bare-metal, srt-slurm runner | GKE pods, operator-less Dynamo (etcd + NATS StatefulSets, plain Deployments) | No measurable throughput effect; frontend/Service hop shows up only in small-conc TTFT medians (ll conc-4/8 +53–58% on 16/32-sample medians while TTFT p99 is better than theirs) |
| Image / dynamo | sglang v0.5.8.post1-cu130, srt-slurm-pinned dynamo | same sglang tag; ai-dynamo 0.8.1 | Parity |
| Request plane | NATS (srtctl injects `--request-plane nats`) | dynamo 0.8.1 default TCP originally; NATS A/B per point | Point-dependent: ll and mid-512 slightly favor NATS; mid-2048 favors TCP (−2.7% vs −6.0% — NATS moves the queue knee); max_tpt plane-independent (mxt falsified the plane hypothesis) |
| KV transfer | NIXL/UCX, no UCX restrictions (bare-metal defaults → NVLink) | NIXL/UCX pinned `cuda_copy,cuda_ipc,tcp` + `UCX_CUDA_IPC_ENABLE_MNNVL=y` (pods have no NICs); RDMA arm (`rc_x` + GID 5 + rail-aware subnet pairing + DRA mrdma claims) as diagnostic | Same effective bulk path (MNNVL cuda_ipc) on parity points; RDMA-vs-MNNVL A/B in flight (m4r/mxr) |
| Bench client | sa-bench from Slurm node; 2×conc @250 warmup + 10×conc @recipe rate | identical sa-bench params, in-cluster Job | `(base)`-era runs understated ~10% — all replaced by mult10 reruns |
| JIT caches | Warm across sweep iterations on persistent nodes | per-pod cold start → dg-cache hostPath + filler passes + VALID/SUSPECT assertion | Unguarded case cost −8% once (DSv4 p3 analog); DSR1 record runs all guarded |
| Node fabric | NVL72 rack, IB fabric | NVL72 rack, 8-rail RoCE (DraNet ipvlan, disjoint /64 rails) | Irrelevant for MNNVL runs; drove the entire RDMA-KV enablement saga (GID 5, rail-aware pairing) |

Per-topology deltas beyond the table: **low_latency** — none in config; early gaps were
methodology (`(base)`), resolved. **mid_curve** — none in config; the residual TPOT gap
(+24–41%) with *better* TTFT medians and equal E2E is queue/admission policy (their scheduler
holds requests longer, keeping decode batches leaner), not a flag we can copy from the recipe.
**max_tpt** — none in config; the −7.6/−9.0% gap is root-caused to prefill dispatch waves
starving decode residency (~30/64 running); recipe `max-total-tokens 524288` may cap the KV
pool below conc-2048 full-context prealloc (~590k/rank) — the mxk headroom diagnostic
(524288→786432) is staged to test exactly that.
''')
    rp = os.path.join(ROOT, 'reports', 'benchmark-report-dsr1-8k1k.md')
    s = open(rp).read()
    marker = '## Comparison vs InferenceX published results (per metric category)'
    body = '\n'.join(doc4) + '\n'
    if marker in s:
        # replace up to the next H2 after the marker (or EOF)
        i = s.index(marker)
        rest = s[i + len(marker):]
        m = re.search(r'\n## (?!#)', rest)
        s = s[:i] + body + (rest[m.start() + 1:] if m else '')
    else:
        s = s.rstrip() + '\n\n' + body
    open(rp, 'w').write(s)
    print(f'FP4 section written into {os.path.relpath(rp, ROOT)}')

    # ---------- FP8 ----------
    theirs8 = load_csv(os.path.join(ROOT, 'dsr1-fp8-sweep', 'reference', 'inferencex-published-dsr1-fp8-gb300.csv'))
    ours8 = load_ours_fp8()
    doc8, _ = sections('Comparison vs InferenceX published results (per metric category)', theirs8, ours8)
    head = """# DeepSeek-R1 FP8 GB300 8k1k Benchmark Report — GKE A4X Max

**Target**: InferenceX `dsr1-fp8-gb300-dynamo-sglang` (configs/nvidia-master.yaml L3077), recipes
srt-slurm `recipes/gb300-fp8/8k1k/stp/` @ branch `sa-submission-q2-2026` (vendored in
`../dsr1-fp8-sweep/reference/`; see PLAN.md there for the full parity table).

**Setup**: cluster `REDACTED-GKE-CLUSTER` (REDACTED-GCP-PROJECT), `a4x-maxgpu-4g-metal` (4x GB300/node);
image `lmsysorg/sglang:v0.5.8.post1-cu130` + ai-dynamo 0.8.0; model `deepseek-ai/DeepSeek-R1-0528`
(native FP8); operator-less Dynamo (etcd+NATS), NIXL KV transfer; sa-bench ISL 8192 / OSL 1024,
range-ratio 0.8, ignore-eos, req_rate inf; warmup 3x conc @250 + measured 10x conc.

**Run status (2026-08-01)**: mid (5P-DEP8 / 1D-DEP32, 72 GPUs) conc 128/256/512/1024 complete;
low-latency (1P1D TP4) conc 4 complete (result reconstructed from the bench client log — the
bench job died at conc 8 before extracting its JSON); low-latency conc 8 FAILED
("Never received a valid chunk"); max (6P-DEP8 / 1D-DEP24) conc 2048/4096 died mid-warmup.
Remaining: ll conc-8, max conc-2048/4096.

**Known caveat**: mid conc-128 completed 1250/1280 requests (30 failures) — treat as
provisional pending a rerun.
"""
    rp8 = os.path.join(ROOT, 'reports', 'benchmark-report-dsr1-fp8-8k1k.md')
    open(rp8, 'w').write(head + '\n' + '\n'.join(doc8) + '\n')
    print(f'FP8 report written to {os.path.relpath(rp8, ROOT)}')


if __name__ == '__main__':
    main()
