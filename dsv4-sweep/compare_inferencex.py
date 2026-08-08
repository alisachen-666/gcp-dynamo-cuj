#!/usr/bin/env python3
"""Compare our DSv4-Pro runs against InferenceX published results (throughput + latency).

Source of truth for theirs: reference/inferencex-published-dsv4-gb300.csv
  UNIT WARNING: that CSV's "(ms)" latency columns are actually SECONDS. Verified by the
  identity Interactivity == 1/TPOT, which holds exactly on all 10 rows only under that reading.
  We convert to ms here (x1000) so both sides are true milliseconds.

Source of truth for ours: results-summary/*.json (+ results-summary/drift/*.json for the
  parity-drift runs p5-p8, which are labelled and NOT treated as clean comparisons).

METRIC-CONVENTION FIX (2026-07-31): the previous comparison in gen-dsv4-report.py compared
OUR mean-derived interactivity (1000/mean_tpot_ms) against THEIR *median* interactivity
(the JSON reference stored the median column). That is apples-to-oranges and made every point
look worse on interactivity than it is. Here we compare mean-to-mean and median-to-median.

Usage:
  ./compare_inferencex.py          # writes ../reports/comparison-dsv4-vs-inferencex.md
  from compare_inferencex import build_sections   # used by gen-dsv4-report.py to embed the
                                                  # same tables in the main DSv4 report
"""
import csv
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, 'reference', 'inferencex-published-dsv4-gb300.csv')
OUT = os.path.join(HERE, '..', 'reports', 'comparison-dsv4-vs-inferencex.md')

POINT_TOPO = {
    'p1': '1P TP4 + 1D TP4',
    'p2': '1P DEP4 + 1D DEP24',
    'p3': '1P DEP4 + 1D DEP8',
    'p4': '1P DEP4 + 1D DEP16',
    'p5': '2P DEP4 + 1D DEP8',
    'p6': '4P DEP4 + 1D DEP8',
    'p7': '6P DEP4 + 1D DEP8',
    'p8': '8P DEP4 + 1D DEP8',
}
S = 1000.0  # their seconds -> ms


def load_theirs():
    rows = []
    with open(CSV) as f:
        lines = [ln for ln in f if not ln.startswith('#')]
    for r in csv.DictReader(lines):
        rows.append({
            'conc': int(r['Concurrency']),
            'pg': int(r['Num Prefill GPUs']),
            'dg': int(r['Num Decode GPUs']),
            'total_gpus': int(r['TP']),
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


def load_ours():
    """key (conc, prefill_gpus, decode_gpus) -> (point_id, data, drift?)"""
    out = {}
    for drift in (False, True):
        pat = os.path.join(HERE, 'results-summary', 'drift' if drift else '', '*.json')
        for f in sorted(glob.glob(pat)):
            g = re.search(r'gpus_(\d+)_ctx_(\d+)_gen_(\d+)', f)
            if not g:
                continue
            d = json.load(open(f))
            pid = os.path.basename(f).split('-')[0]
            key = (d['max_concurrency'], int(g.group(2)), int(g.group(3)))
            # a clean (non-drift) run always wins over a drift run for the same point
            if key in out and not out[key][2]:
                continue
            out[key] = (pid, d, drift)
    return out


def pct(ours, theirs):
    return 100.0 * (ours - theirs) / theirs if theirs else float('nan')


def fmt_gap(ours, theirs, lower_is_better=False):
    g = pct(ours, theirs)
    mark = ''
    if lower_is_better:
        mark = ' ✅' if g <= -5 else (' ⚠️' if g >= 20 else '')
    else:
        mark = ' ✅' if g >= 5 else (' ⚠️' if g <= -10 else '')
    return f'{g:+.1f}%{mark}'



GCS = 'https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs'
GCSF = 'https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs'
GH = 'https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main'
RUNLINKS = {
    'p1': ('dsv4-point1', None, 'dsv4-sweep/manifests/point1-1p1d-tp4-mtp.yaml'),
    'p2': ('dsv4-p2', 'dsv4-bench-p2-RECORD.log', 'dsv4-sweep/manifests/p2.yaml'),
    'p3': ('dsv4-p3-rerecord', 'dsv4-bench-p3-rerecord.log', 'dsv4-sweep/manifests/p3.yaml'),
    'p4': ('dsv4-p4', 'dsv4-bench-p4.log', 'dsv4-sweep/manifests/p4.yaml'),
    'p5': ('dsv4-p5', 'dsv4-bench-p5.log', 'dsv4-sweep/manifests/p5.yaml'),
    'p6': ('dsv4-p6', 'dsv4-bench-p6.log', 'dsv4-sweep/manifests/p6.yaml'),
    'p7': ('dsv4-p7', 'dsv4-bench-p7.log', 'dsv4-sweep/manifests/p7.yaml'),
    'p8': ('dsv4-p8', 'dsv4-bench-p8.log', 'dsv4-sweep/manifests/p8.yaml'),
}


def linkcell(pid):
    if pid not in RUNLINKS: return ''
    srv, cl, cfg = RUNLINKS[pid]
    out = ' [logs]({0}/server-logs/{1})'.format(GCS, srv)
    if cl: out += ' [bench]({0}/client-logs/{1})'.format(GCSF, cl)
    out += ' [cfg]({0}/{1}) [cfg-gcs]({2}/configs/{1})'.format(GH, cfg, GCSF)
    return out


def summary_table(theirs, ours):
    """Dual-source table (InferenceMax / GCP bench with GKE); GCP cells: value (gap%)."""
    doc = ['\n## Summary: InferenceMax vs GCP bench with GKE\n']
    doc.append('GCP cells show `value (gap vs InferenceMax)`. Throughput: higher is better; '
               'E2EL/ITL: lower is better. `*` = drift-run data (superseded points excluded).\n')
    doc.append('| Pareto region | Prefill-Decode topology | Conc | Sources | Req plane | KV transfer | Total Tput (tok/s/GPU) | Input Tput (tok/s/GPU) | Output Tput (tok/s/GPU) | Median E2EL (s) | Median ITL (s) |')
    doc.append('|---|---|---|---|---|---|---|---|---|---|---|')
    # region labels verbatim from InferenceX nvidia-master.yaml @5885a434 search-space comments
    REGION = {(4, 4): 'Low latency', (4, 24): 'Low latency',
              (4, 8): 'Mid curve', (4, 16): 'Mid curve', (8, 8): 'Mid curve', (16, 8): 'Mid curve',
              (24, 8): 'High concurrency', (32, 8): 'High concurrency'}
    def gp(a, b): return f'{a:,.1f} ({100*(a-b)/b:+.1f}%)'
    def gs(a, b): return f'{a:.2f} ({100*(a-b)/b:+.1f}%)'
    import json as _json, os as _os
    p8lat = None
    f8 = _os.path.join(HERE, 'results-summary', 'p8-rerun-latency-only.extracted.json')
    if _os.path.exists(f8):
        p8lat = _json.load(open(f8))
    for r in sorted(theirs, key=lambda x: (x['pg'], x['dg'], x['conc'])):
        key = (r['conc'], r['pg'], r['dg'])
        topo = f"{r['pg']}P/{r['dg']}D"
        doc.append(f"| {REGION.get((r['pg'], r['dg']), '')} | {topo} | {r['conc']} | InferenceMax | NATS | MNNVL (mooncake) | {r['total_per_gpu']:,.1f} | "
                   f"{r['in_per_gpu']:,.1f} | {r['out_per_gpu']:,.1f} | "
                   f"{r['median_e2el_ms']/1000:.2f} | {r['median_itl_ms']/1000:.2f} |")
        rec = ours.get(key)
        if rec and not rec[2]:  # clean run with full data
            pid, d, _ = rec
            kv = 'RDMA (mooncake)' if pid.endswith('r') else 'MNNVL (mooncake)'
            o_tot = d['total_token_throughput'] / r['total_gpus']
            o_in = d['total_input_tokens'] / d['duration'] / r['pg']
            o_out = d['output_throughput'] / r['dg']
            doc.append(f"| | | | GCP bench with GKE ({pid}){linkcell(pid)} | NATS | {kv} | {gp(o_tot, r['total_per_gpu'])} | "
                       f"{gp(o_in, r['in_per_gpu'])} | {gp(o_out, r['out_per_gpu'])} | "
                       f"{gs(d['median_e2el_ms']/1000, r['median_e2el_ms']/1000)} | "
                       f"{gs(d['median_itl_ms']/1000, r['median_itl_ms']/1000)} |")
        elif key == (8192, 32, 8) and p8lat:
            m = p8lat.get('_measured_throughput_record_20260807')
            if m:
                doc.append(f"| | | | GCP bench with GKE (p8; tput = 2026-08-07 record, SUSPECT-JIT conservative; latency = 2026-08-01 rerun){linkcell('p8')} | NATS | MNNVL (mooncake) | "
                           f"{gp(m['total_per_gpu'], r['total_per_gpu'])} | {gp(m['in_per_prefill_gpu'], r['in_per_gpu'])} | {gp(m['out_per_decode_gpu'], r['out_per_gpu'])} | "
                           f"{gs(p8lat['median_e2el_ms']/1000, r['median_e2el_ms']/1000)} | "
                           f"{gs(p8lat['median_itl_ms']/1000, r['median_itl_ms']/1000)} |")
            else:
                doc.append(f"| | | | GCP bench with GKE (p8, latency-only){linkcell('p8')} | NATS | MNNVL (mooncake) | — | — | — | "
                           f"{gs(p8lat['median_e2el_ms']/1000, r['median_e2el_ms']/1000)} | "
                           f"{gs(p8lat['median_itl_ms']/1000, r['median_itl_ms']/1000)} |")
        else:
            doc.append("| | | | GCP bench with GKE | — | — | — | — | — | — | — |")
    return doc


def build_sections(standalone=True):
    """Return the comparison document as a list of markdown lines.

    standalone=True  -> full document with its own H1 (written to its own report file)
    standalone=False -> embeddable section starting at H2, for gen-dsv4-report.py to append
                        to reports/benchmark-report-dsv4-8k1k.md
    """
    theirs = load_theirs()
    ours = load_ours()
    doc = []

    if standalone:
        doc.append('# DeepSeek-V4-Pro 8k1k GB300 — Our results vs InferenceX\n')
    else:
        doc.append('# Comparison vs InferenceX published results\n')
    doc.append('Generated by `dsv4-sweep/compare_inferencex.py`. Theirs: '
               '`dsv4-sweep/reference/inferencex-published-dsv4-gb300.csv` '
               '(InferenceX, run date 2026-06-03, Apache-2.0, © 2026 SemiAnalysis LLC). '
               'Ours: `dsv4-sweep/results-summary/`.\n')
    doc.append("""> **Two corrections applied in this document** (both affect how earlier numbers read):
>
> 1. **Unit fix.** The InferenceX CSV labels TTFT/TPOT/ITL/E2E as `(ms)` but the values are
>    **seconds**. Proven by `Interactivity == 1/TPOT`, exact on all 10 rows only under that
>    reading (e.g. `1/0.008444514 = 118.42` = the stated Mean Interactivity). All their latency
>    figures below are converted to true milliseconds.
> 2. **Mean-vs-median fix.** `gen-dsv4-report.py` compared **our mean-derived** interactivity
>    (`1000/mean_tpot_ms`) against **their median** interactivity, because the JSON reference
>    stored only the median column. That is apples-to-oranges and understated us on every point.
>    Here mean is compared to mean and median to median.
""")

    doc.extend(summary_table(theirs, ours))
    doc.append("""
## Headline conclusions

- **Throughput: we match on every point that ran with clean config parity.** p4 is exact to 4
  significant figures (+0.0%); **p3 re-record (2026-08-01, warm DeepGEMM cache + filler pass,
  assertion VALID) landed at +0.3%** — the earlier −8.0% was entirely the JIT-in-measured-window
  defect, and the +5-day image deviation is exonerated. p1 is +11.7% in our favour but on a
  10-prompt run (see caveat below). The two large deficits (p5 −18.2%, p8 −17.9%) were **drift
  runs**; both since replaced: **p5's clean make-up (2026-08-01) landed at +1.8%** (1650.2 vs
  1620.9 out/decode-GPU; TPOT 13.22 vs 13.05 ms; TTFT p99 44.1 vs 49.6 s) — verdict SUSPECT
  (one late JIT compile + ~31 s of prefill stalls in the record window), so the number is
  *conservative*: contamination only understates us, and it beats parity anyway. p8's corrected
  rerun reached latency parity (TPOT −2.1%, TTFT p99 +3.7%); log-plateau analysis puts its
  throughput at ≈−2.5% measured / +1.4% steady-state (the earlier ≈−10% estimate was
  calibration error; see results-summary/p8-rerun-latency-only.extracted.json).
- **Latency: TPOT is at or better than parity everywhere** (−31% at p8, −9.5% at p7, ~−1 to −2%
  at p3/p4). Our weakness is entirely **TTFT tail**, not decode: p99 TTFT is +71% at p3, +109% at
  p8 (both since improved by reruns — see throughput bullet). **ROOT-CAUSED (2026-08-01)**:
  DeepGEMM JIT compile sessions froze the prefill worker
  *inside* p3's measured window (12.3s + 12.5s at 12:07, ~6.8% of the 366s run ≈ the −8%
  throughput gap; 12s freezes ≈ the p99 TTFT tail). The DeepGEMM cache was cold every pod —
  only the flashinfer cache was persisted. Evidence: `results-summary/p3-jit-stall-evidence.txt`;
  fix (dg-cache mount + double-bench protocol) in PLAN.md. **Re-record confirms the fix:
  +0.3% throughput, TTFT p99 gap +70.7% -> +6.6%, zero JIT/stalls in the measured window.**
- **Interactivity, compared like-for-like, is better than theirs on every point** (+1.9% at p4 up
  to +45% at p8). The previous report showed us losing on this metric purely because of the
  mean-vs-median mismatch corrected above.
- **The p8 signature is now unambiguous**: −17.9% throughput but **+45% interactivity and −31%
  TPOT**. We spent compute on deeper speculation (3-step/4-draft where the recipe says
  1-step/2-draft at that concurrency) and bought per-user latency with it. This confirms the
  MTP-depth diagnosis in the existing report using latency evidence the JSON reference could not
  supply.
- **p2 verification rerun (2026-08-07, `p2b-*` in results-summary)**: an independent PASSES=2
  rerun on the rebuilt np-3 pool re-measured conc 8/32/64 at +15.8/+5.2/+3.2% vs InferenceMax
  (prior clean record: +18.6/+9.5/+8.0%, which remains final). Verdict SUSPECT (one JIT
  compile + 8.1 s prefill stall in a conc-32 window — contamination only understates), so the
  rerun is a *conservative floor* that independently confirms the p2 leads.
- **Accuracy: not measurable from either dataset.** See §4.

**Caveat on p1**: conc 1, 10 prompts, 48 s wall time. That is far below the 10×-concurrency
methodology used everywhere else, so the +11.7% is within run-to-run noise and should not be
reported as an advantage without a longer rerun.
""")

    # ---------------- throughput ----------------
    doc.append('\n## 1. Throughput\n')
    doc.append('Convention (identical both sides): input tok/s ÷ prefill GPUs, '
               'output tok/s ÷ decode GPUs, total ÷ all GPUs.\n')
    doc.append('| Point | Topology | P/D GPUs | Conc | Out/decode-GPU ours | theirs | gap | '
               'In/prefill-GPU ours | theirs | gap | Total/GPU ours | theirs | gap |')
    doc.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for r in sorted(theirs, key=lambda x: (x['pg'], x['dg'], x['conc'])):
        key = (r['conc'], r['pg'], r['dg'])
        rec = ours.get(key)
        if not rec:
            doc.append(f"| (not run) | — | {r['pg']}/{r['dg']} | {r['conc']} | — | "
                       f"{r['out_per_gpu']:.1f} | — | — | {r['in_per_gpu']:.0f} | — | — | "
                       f"{r['total_per_gpu']:.1f} | — |")
            continue
        pid, d, drift = rec
        tag = pid + ('*' if drift else linkcell(pid))
        o_out = d['output_throughput'] / r['dg']
        o_in = d['total_input_tokens'] / d['duration'] / r['pg']
        o_tot = d['total_token_throughput'] / r['total_gpus']
        doc.append(f"| {tag} | {POINT_TOPO.get(pid,'?')} | {r['pg']}/{r['dg']} | {r['conc']} | "
                   f"{o_out:.1f} | {r['out_per_gpu']:.1f} | {fmt_gap(o_out, r['out_per_gpu'])} | "
                   f"{o_in:.0f} | {r['in_per_gpu']:.0f} | {fmt_gap(o_in, r['in_per_gpu'])} | "
                   f"{o_tot:.1f} | {r['total_per_gpu']:.1f} | {fmt_gap(o_tot, r['total_per_gpu'])} |")
    doc.append('\n`*` = run carried recipe-parity drift (see `dsv4-sweep/PLAN.md` error log); '
               'not a clean comparison.\n')

    # ---------------- latency ----------------
    doc.append('\n## 2. Latency\n')
    doc.append('All values true milliseconds (theirs converted from seconds). '
               'Lower is better; ✅ = we are ≥5% better, ⚠️ = we are ≥20% worse.\n')
    doc.append('| Point | Conc | TTFT mean ours | theirs | gap | TTFT p99 ours | theirs | gap | '
               'TPOT mean ours | theirs | gap | E2E mean ours | theirs | gap |')
    doc.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for r in sorted(theirs, key=lambda x: (x['pg'], x['dg'], x['conc'])):
        key = (r['conc'], r['pg'], r['dg'])
        rec = ours.get(key)
        if not rec:
            continue
        pid, d, drift = rec
        tag = pid + ('*' if drift else linkcell(pid))
        doc.append(
            f"| {tag} | {r['conc']} | "
            f"{d['mean_ttft_ms']:.0f} | {r['mean_ttft_ms']:.0f} | {fmt_gap(d['mean_ttft_ms'], r['mean_ttft_ms'], True)} | "
            f"{d['p99_ttft_ms']:.0f} | {r['p99_ttft_ms']:.0f} | {fmt_gap(d['p99_ttft_ms'], r['p99_ttft_ms'], True)} | "
            f"{d['mean_tpot_ms']:.2f} | {r['mean_tpot_ms']:.2f} | {fmt_gap(d['mean_tpot_ms'], r['mean_tpot_ms'], True)} | "
            f"{d['mean_e2el_ms']:.0f} | {r['mean_e2el_ms']:.0f} | {fmt_gap(d['mean_e2el_ms'], r['mean_e2el_ms'], True)} |")

    # ---------------- interactivity ----------------
    doc.append('\n## 3. Interactivity (tok/s/user = 1/TPOT) — like-for-like\n')
    doc.append('| Point | Conc | Mean ours | Mean theirs | gap | Median ours | Median theirs | gap |')
    doc.append('|---|---|---|---|---|---|---|---|')
    for r in sorted(theirs, key=lambda x: (x['pg'], x['dg'], x['conc'])):
        key = (r['conc'], r['pg'], r['dg'])
        rec = ours.get(key)
        if not rec:
            continue
        pid, d, drift = rec
        tag = pid + ('*' if drift else linkcell(pid))
        om, omed = 1000.0 / d['mean_tpot_ms'], 1000.0 / d['median_tpot_ms']
        doc.append(f"| {tag} | {r['conc']} | {om:.1f} | {r['mean_inter']:.1f} | "
                   f"{fmt_gap(om, r['mean_inter'])} | {omed:.1f} | {r['median_inter']:.1f} | "
                   f"{fmt_gap(omed, r['median_inter'])} |")

    # ---------------- config parity ----------------
    doc.append("""
## Config parity vs InferenceX (per point)

Every GCP run used the recipe image (`lmsysorg/sglang:nightly-dev-20260601-373cadc9`),
dynamo @ `81d0555`, the **NATS** request plane (`DYN_REQUEST_PLANE=nats`, matching srtctl),
and **mooncake KV transfer over MNNVL** (`MC_FORCE_MNNVL=1`, from the recipe env). The p8r
diagnostic arm (pending) is the only RDMA-KV variant (`MC_FORCE_MNNVL=0` + DRA mrdma claims).
Platform deltas shared with the DSR1 report (GKE pods vs Slurm, in-cluster bench client,
per-pod cold JIT caches) apply here identically. Per-point flag parity and its history:

| Point | Recipe config | Flag parity vs recipe | Influence on result |
|---|---|---|---|
| p1 | `disagg-low-latency-1p1d-tp4-tp4-mtp` | Identical | +11.7%, but conc-1/10-prompt run — treat as noise (see caveat) |
| p2 | `disagg-low-latency-1p6d-dep4-tp4-mtp` | Identical **after** pre-run fix: first draft missed 5 prefill `SGLANG_OPT_*` vars (incl. `DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=9216`); caught in parity review before anything ran | +18.6/+9.5/+8.0% — clean |
| p3 | `disagg-mid-curve-1p1d-dep4-dep8-mtp` | Identical flags; environmental defect: DeepGEMM JIT compiled inside the measured window (only flashinfer cache was persisted) | −8.0% artifact → +0.3% after dg-cache + filler pass (assertion VALID) |
| p4 | `disagg-mid-curve-1p1d-dep4-dep16-mtp` | Identical | +0.0% (exact to 4 sig figs) — the anchor proving config/kernel parity |
| p5 | `disagg-mid-curve-2p1d-dep4-dep8-mtp` | Drift run inherited the p3 template (DeepGEMM tokens/rank 2048 vs 4096, ctx-len, cuda-graph-bs) | −18.2% artifact → +1.8% after parameterized gen-point fix |
| p6 | `disagg-mid-curve-4p1d-dep4-dep8-mtp` | Same drift class, corrected before its clean run | +0.5% |
| p7 | `disagg-high-conc-6p1d-dep4-dep8-mtp` | Same drift class, corrected | −0.4% |
| p8 | `disagg-high-conc-8p1d-dep4-dep8-mtp` | Worst drift: MTP 3-step/4-draft vs recipe 1/2, ctx 16384 vs 9216, plus the p5-class deltas | −17.9% artifact (compute burned on deep speculation: +45% interactivity signature) → corrected rerun at latency parity; measured tput rerun in flight (MNNVL + RDMA A/B) |

The drift mechanism (silent template inheritance in `gen-point.py`) and both root-cause
chains (JIT-in-window, MTP depth) are documented in `dsv4-sweep/PLAN.md` and
`reports/perf-tuning-walkthrough.md` (cases 3–4).
""")

    # ---------------- accuracy ----------------
    doc.append("""
## 4. Accuracy

**No accuracy comparison is possible from this data, on either side.**

- The InferenceX CSV contains throughput and latency columns only — there is no accuracy,
  quality, or task-score column anywhere in it.
- Our runs cannot produce one even in principle: sa-bench drives a **synthetic `random`
  dataset** (ISL 8192 / OSL 1024, `random_range_ratio 0.8`) with **`--ignore-eos`**. The prompts
  are random token sequences and generation is forced to a fixed length regardless of the model's
  natural stopping point, so the output text is not meaningful and cannot be scored.

These are throughput/latency harnesses by construction. Claiming accuracy parity from them would
be unfounded. To make an accuracy claim we would need a separate evaluation run:
a real task suite (e.g. MMLU / GPQA / AIME / LiveCodeBench) against the same served endpoint,
with `ignore_eos` off and a real dataset, compared against DeepSeek's published DSv4-Pro scores.
That is a distinct workstream and is **not** currently in this repo.

The one indirect signal we have is that p4 reproduces their throughput to 4 significant figures.
That tells us we are running the same checkpoint through the same kernels and serving path as
they are — it constrains the *configuration* to be identical, which makes a large accuracy
divergence unlikely. It is not itself an accuracy measurement.
""")

    if not standalone:
        # demote every heading one level so the block nests under the report's H1
        doc = [re.sub(r'^(#+)', r'#\1', ln, flags=re.M) for ln in doc]
    return doc


def main():
    doc = build_sections(standalone=True)
    with open(OUT, 'w') as f:
        f.write('\n'.join(doc) + '\n')
    theirs, ours = load_theirs(), load_ours()
    print(f'wrote {os.path.relpath(OUT, os.path.join(HERE, ".."))}')
    print(f'  their points: {len(theirs)}; ours matched: '
          f'{sum(1 for r in theirs if (r["conc"], r["pg"], r["dg"]) in ours)}')


if __name__ == '__main__':
    main()
