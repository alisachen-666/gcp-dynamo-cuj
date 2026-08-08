#!/usr/bin/env python3
"""Generate reports/benchmark-report-rdma-kv.md — the RDMA-KV-only network report.

Compares every point run with GPUDirect RDMA KV transfer against (a) InferenceMax and
(b) our MNNVL final for the same point. Missing points render as `pending`, so this can
be regenerated at any stage of the sweep. Sources:
  DSR1: dsr1-sweep/results-summary/{llr,m4r,mxr}-*.json  (NIXL/UCX rc_x)
  DSv4: dsv4-sweep/results-summary/p[1-8]r-*.json        (mooncake, MC_FORCE_MNNVL=0)
"""
import glob, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'dsr1-sweep'))
sys.path.insert(0, os.path.join(ROOT, 'dsv4-sweep'))
import compare_inferencex_dsr1 as dsr1
import compare_inferencex as dsv4

OUT = os.path.join(ROOT, 'reports', 'benchmark-report-rdma-kv.md')


def pct(a, b):
    return f'{100.0*(a-b)/b:+.1f}%'


def dsr1_rows():
    theirs = dsr1.load_csv(os.path.join(ROOT, 'dsr1-sweep', 'reference',
                                        'inferencex-published-dsr1-fp4-gb300.csv'))
    ours = dsr1.load_ours_fp4()
    rdma = {k[0]: v for k, v in ours.items()
            if isinstance(k, tuple) and len(k) == 2 and k[1] == 'rdmakv'}
    # MNNVL final = optimal pick among non-RDMA variants (mirrors the main report policy)
    cands = {}
    for k, rec in ours.items():
        if isinstance(k, tuple) and len(k) == 2:
            if k[1] == 'rdmakv':
                continue
            cands.setdefault(k[0], []).append(rec)
        else:
            cands.setdefault(k, []).append(rec)
    mnnvl = {k: max(v, key=lambda c: (c[1]['output_throughput'], -c[1]['median_itl_ms']))
             for k, v in cands.items()}
    rows = []
    for r in sorted(theirs, key=lambda x: (x['pg'], x['dg'], x['conc'])):
        key = (r['conc'], r['pg'], r['dg'])
        rows.append((f"{r['pg']}P/{r['dg']}D", r['conc'], r, mnnvl.get(key), rdma.get(key),
                     r['dg'], r['tot']))
    return rows


def dsv4_rows():
    theirs = dsv4.load_theirs()
    ours = dsv4.load_ours()
    rdma = {}
    for f in sorted(glob.glob(os.path.join(ROOT, 'dsv4-sweep', 'results-summary',
                                           'p[1-8]r-*.json'))):
        if 'INVALID' in os.path.basename(f):
            continue
        d = json.load(open(f))
        pid = os.path.basename(f).split('-')[0]
        rdma[(d['max_concurrency'], pid)] = d
    rows = []
    for r in sorted(theirs, key=lambda x: (x['pg'], x['dg'], x['conc'])):
        key = (r['conc'], r['pg'], r['dg'])
        rec = ours.get(key)
        pid = rec[0] if rec else None
        mn = (rec[0], rec[1]) if rec and not rec[2] else None
        rd = rdma.get((r['conc'], (pid or '') + 'r'))
        rows.append((f"{r['pg']}P/{r['dg']}D", r['conc'], r, mn, rd, r['dg'], r['total_gpus']))
    return rows


def table(rows, out_key='out_per_gpu'):
    doc = ['| Topology | Conc | InferenceMax out/GPU | MNNVL final out/GPU | RDMA out/GPU | '
           'RDMA vs MNNVL | RDMA vs InferenceMax | RDMA TPOT vs MNNVL |',
           '|---|---|---|---|---|---|---|---|']
    for topo, conc, r, mn, rd, dg, tot in rows:
        ix = r[out_key]
        mtxt = rtxt = vs_m = vs_i = tp = '—'
        mn_out = mn_tpot = None
        if mn:
            _, md = mn if isinstance(mn, tuple) and len(mn) == 2 else (None, mn[1])
            mn_out = md['output_throughput'] / dg
            mn_tpot = md['mean_tpot_ms']
            mtxt = f'{mn_out:,.1f}'
        if rd:
            d = rd[1] if isinstance(rd, tuple) else rd
            rd_out = d['output_throughput'] / dg
            rtxt = f'{rd_out:,.1f}'
            vs_i = pct(rd_out, ix)
            if mn_out:
                vs_m = pct(rd_out, mn_out)
                tp = pct(d['mean_tpot_ms'], mn_tpot) + ' (lower better)'
        else:
            rtxt = '*pending*'
        doc.append(f'| {topo} | {conc} | {ix:,.1f} | {mtxt} | {rtxt} | {vs_m} | {vs_i} | {tp} |')
    return doc


def _network_table(title, rows, REGION, pick):
    """Dual-source table in the main-report format: InferenceMax row (absolutes) then the
    GCP row with value (gap vs InferenceMax) on every metric. `pick(row)` -> (tag, data)."""
    doc = [title, '',
           '| Region | Topology | Conc | Sources | Out/decode-GPU (tok/s) | TPOT mean (ms) | TTFT med (s) | Median ITL (s) |',
           '|---|---|---|---|---|---|---|---|']
    def gv(a, b, fmt=',.1f'):
        return f'{a:{fmt}} ({100*(a-b)/b:+.1f}%)'
    for topo, conc, r, mn, rd, dg in rows:
        reg = REGION.get((r['pg'], r['dg']), '')
        doc.append(f"| {reg} | {topo} | {conc} | InferenceMax | {r['out_per_gpu']:,.1f} | "
                   f"{r['mean_tpot_ms']:.2f} | {r['median_ttft_ms']/1000:.2f} | {r['median_itl_ms']/1000:.2f} |")
        rec = pick((topo, conc, r, mn, rd, dg))
        if rec:
            tag, d = rec
            out = d['output_throughput'] / dg
            lc = dsr1.linkcell(tag, r['pg'], r['dg'], conc)
            label = 'GCP bench with GKE' + (f' {tag}' if tag else '')
            doc.append(f"| | | | {label}{lc} | {gv(out, r['out_per_gpu'])} | "
                       f"{gv(d['mean_tpot_ms'], r['mean_tpot_ms'], '.2f')} | "
                       f"{gv(d['median_ttft_ms']/1000, r['median_ttft_ms']/1000, '.2f')} | "
                       f"{gv(d['median_itl_ms']/1000, r['median_itl_ms']/1000, '.2f')} |")
        else:
            doc.append('| | | | GCP bench with GKE | *pending* | — | — | — |')
    return doc


def dsr1_dual_tables():
    """Two per-network dual-source tables (MNNVL final vs RDMA) + computed trend deltas."""
    rows0 = dsr1_rows()
    REGION = {(4, 16): 'Low latency', (24, 48): 'Mid curve', (40, 32): 'Max throughput'}
    pairs = [(topo, conc, r, mn, rd, dg) for topo, conc, r, mn, rd, dg, tot in rows0]
    doc = ['## DSR1-FP4: MNNVL vs RDMA — per-network tables', '',
           'Format matches the main summary table: InferenceMax row = absolute values; GCP row = '
           '`value (gap vs InferenceMax)` per metric. Throughput: higher is better; TPOT/TTFT/ITL: '
           'lower is better (negative gap = we are faster).', '']
    doc += _network_table('### Table 1 — KV over MNNVL (NVLink; final per point, optimal variant)',
                          pairs, REGION, lambda p: p[3])
    doc += ['']
    doc += _network_table('### Table 2 — KV over GPUDirect RDMA (rev4: `UCX_NET_DEVICES` + GID 5 + rail-aware pairing)',
                          pairs, REGION,
                          lambda p: (('RDMA-KV', p[4][1] if isinstance(p[4], tuple) else p[4]) if p[4] else None))
    doc += ['', '### Network delta (RDMA vs MNNVL, same recipe/methodology/cluster)', '',
            '| Region | Conc | Δ Out tput | Δ TPOT (lower better) | Δ TTFT med | Δ ITL med |',
            '|---|---|---|---|---|---|']
    for topo, conc, r, mn, rd, dg in pairs:
        reg = REGION.get((r['pg'], r['dg']), '')
        if mn and rd:
            _, md = mn
            d = rd[1] if isinstance(rd, tuple) else rd
            doc.append(f"| {reg} | {conc} | {pct(d['output_throughput'], md['output_throughput'])} | "
                       f"{pct(d['mean_tpot_ms'], md['mean_tpot_ms'])} | "
                       f"{pct(d['median_ttft_ms'], md['median_ttft_ms'])} | "
                       f"{pct(d['median_itl_ms'], md['median_itl_ms'])} |")
        else:
            doc.append(f'| {reg} | {conc} | *pending* | — | — | — |')
    doc += ['', """### Trend analysis: what changing the KV network does

- **Decode-side contention relief grows with concurrency.** MNNVL KV transfers ride the same
  NVLink fabric (and copy-engine/SM resources) the decode batch computes on; at fat batches
  every prefill->decode handoff steals from token generation. Moving KV to the NICs
  (GPUDirect RDMA) offloads that entirely. First confirmation at mid conc-2048: TPOT dropped
  from ~16-19 ms (MNNVL variants) to 13.75 ms — closing the long-standing +24-41% TPOT gap
  vs InferenceMax to ~3% — while throughput rose to the best value recorded on the point
  (-1.2% vs InferenceMax, vs -2.7% best MNNVL). Expect the effect to shrink toward zero at
  low-latency concurrencies (thin batches, KV traffic sparse) and to be the key question at
  max_tpt (whether dispatch-wave stalls interact with transport).
- **TTFT moves the other way at saturation.** RDMA per-transfer setup adds prefill-side
  latency and the freed decode capacity admits deeper queues; conc-2048 TTFT median rose vs
  MNNVL. At 8k1k this trades a one-time wait for steady-state speed — E2E/interactivity
  judge the net (per-metric tables in the main report).
- **InferenceX never ran this configuration** (their KV rides NVLink in every published
  GB300 result, both stacks) — so RDMA-vs-MNNVL deltas here are new information about the
  platform envelope, not a reproduction check.
- Rows marked *pending* fill in as the sweep lands (mid 512/4096, max_tpt, low-latency 4-64);
  the trend claims above will be re-evaluated against the full matrix."""]
    return doc


def main():
    doc = ['# RDMA-KV Network Report — GPUDirect RDMA transport A/B on GKE A4X Max',
           '',
           'Every point re-run with the KV-transfer path on **GPUDirect RDMA** over the 8-rail',
           'RoCE fabric (DraNet ipvlan, DRA `mrdma.google.com` claims) instead of MNNVL/NVLink.',
           'NCCL/DeepEP collectives stay on NVLink in both arms — only KV transfer differs.',
           'MNNVL results remain the recipe-parity record; this report isolates the network',
           'transport variable.',
           '',
           '**DSR1 (NIXL/UCX)**: `UCX_TLS=cuda_copy,rc_x,tcp`, `UCX_IB_GID_INDEX=5` (in-pod',
           'ipvlan RoCEv2 GID), rail-aware pairing `UCX_IB_ROCE_LOCAL_SUBNET=y` +',
           '`UCX_IB_ROCE_SUBNET_PREFIX_LEN=64` (rails are 8 disjoint /64s, no cross-rail routes).',
           'Transport audited per run: bulk ops must be 100% `rc_mlx5`, zero `cuda_ipc`.',
           '',
           '**DSv4 (mooncake)**: `MC_FORCE_MNNVL=0` + mrdma claims (mooncake pairs NICs by GPU',
           'affinity — rail-aligned by construction on A4X).',
           '',
           'Methodology: identical sa-bench parameters as the MNNVL record runs (mult10 for',
           'DSR1, 3x warm + 10x measured for DSv4). DSv4 RDMA points p1r–p7r ran PASSES=1',
           '(diagnostic) with the JIT/stall assertion stamping VALID/SUSPECT; m4r/mxr/p8r/llr',
           'per their runner defaults. `pending` = not yet collected.',
           '',
           '## DSR1-FP4 8k1k',
           '']
    doc += table(dsr1_rows())
    doc += ['']
    doc += dsr1_dual_tables()
    doc += ['', '## DSv4 8k1k', '',
            '**Descoped (user decision 2026-08-07): no DSv4 RDMA sweep.** DSv4 record results',
            'are MNNVL (recipe parity); the rows below show the mooncake-RDMA arms as designed',
            'but not executed.', '']
    doc += table(dsv4_rows())
    doc += ['',
            '## Reading the numbers',
            '',
            '- **RDMA vs MNNVL** is the controlled A/B (same recipe, same methodology, same',
            '  cluster) — it isolates the KV-transport contribution.',
            '- **RDMA vs InferenceMax** contextualizes against the published bare-metal numbers.',
            '- Transport evidence per run (rc_mlx5/cuda_ipc/NIXL-error totals) is in the run',
            '  logs under `server-logs/dsr1-{llr,m4r-rev3-3pts,mxr}` and `server-logs/dsv4-p*r`',
            '  in GCS; configs under `configs/` (see the main reports for the per-cell links).',
            '']
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w').write('\n'.join(doc) + '\n')
    n_rd = sum(1 for _, _, _, _, rd, _, _ in dsr1_rows() + dsv4_rows() if rd)
    print(f'wrote {os.path.relpath(OUT, ROOT)} ({n_rd}/18 RDMA points collected)')


if __name__ == '__main__':
    main()
