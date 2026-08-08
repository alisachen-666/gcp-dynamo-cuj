#!/usr/bin/env python3
"""Generate reports/benchmark-report-dsv4-8k1k.md from dsv4-sweep/results/*.json."""
import json, glob, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# slims carry every scalar the report needs (incl. precomputed percentiles) and survive
# raw pruning — raws are archived in GCS and deleted locally after verified upload
R = os.path.join(HERE, 'results-summary')
OUT = os.path.join(HERE, '..', 'reports', 'benchmark-report-dsv4-8k1k.md')

POINTS = {  # pid -> (recipe, topology description)
 'p1': ('disagg-low-latency-1p1d-tp4-tp4-mtp', '1 prefill TP4 (1 node) + 1 decode TP4 (1 node)'),
 'p3': ('disagg-mid-curve-1p1d-dep4-dep8-mtp', '1 prefill DEP4 (1 node) + 1 decode DEP8 (2 nodes)'),
 'p4': ('disagg-mid-curve-1p1d-dep4-dep16-mtp', '1 prefill DEP4 (1 node) + 1 decode DEP16 (4 nodes)'),
 'p5': ('disagg-mid-curve-2p1d-dep4-dep8-mtp', '2 prefill DEP4 (2 nodes) + 1 decode DEP8 (2 nodes)'),
 'p6': ('disagg-mid-curve-4p1d-dep4-dep8-mtp', '4 prefill DEP4 (4 nodes) + 1 decode DEP8 (2 nodes)'),
 'p7': ('disagg-high-conc-6p1d-dep4-dep8-mtp', '6 prefill DEP4 (6 nodes) + 1 decode DEP8 (2 nodes)'),
 'p8': ('disagg-high-conc-8p1d-dep4-dep8-mtp', '8 prefill DEP4 (8 nodes) + 1 decode DEP8 (2 nodes)'),
}

def pct(arr, p):
    flat = []
    for x in arr:
        flat.extend(x) if isinstance(x, list) else flat.append(x)
    s = sorted(v * 1000.0 for v in flat)
    if not s: return float('nan')
    k = (len(s) - 1) * p / 100.0
    f = int(k); c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)

rows = []
for f in sorted(glob.glob(f'{R}/*.json')):
    pid = os.path.basename(f).split('-')[0]
    if not (pid.startswith('p') and pid[1:].isdigit()):
        continue   # verification/variant records (e.g. p2b-*) are not point rows
    g = re.search(r'gpus_(\d+)_ctx_(\d+)_gen_(\d+)', f)
    if not g:
        continue   # e.g. p8-rerun-latency-only.extracted.json (handled specially elsewhere)
    d = json.load(open(f))
    tot, pg, dg = map(int, g.groups())
    rows.append((pid, d, tot, pg, dg))
rows.sort(key=lambda r: (int(r[0][1:]),))

doc = ["""# DeepSeek-V4-Pro FP4 GB300 8k1k Benchmark Report — GKE A4X Max

**Target**: InferenceX `dsv4-fp4-gb300-dynamo-sglang-mtp` (commit `5885a434`), recipes from
NVIDIA/srt-slurm `recipes/dsv4-pro/sglang/gb300-fp4/8k1k/disagg/` (vendored verbatim in
`../dsv4-sweep/reference/`).

**Setup**
- Cluster: `REDACTED-GKE-CLUSTER-OLD` (REDACTED-GCP-PROJECT), np-2 NVL72 domain, `a4x-maxgpu-4g-metal` (4x GB300/node)
- Model: `deepseek-ai/DeepSeek-V4-Pro`, **mxfp4**, staged to
  `gs://alisachen-models/deepseek-ai/DeepSeek-V4-Pro/` and seeded to all 18 nodes' local SSD
- Serving: NVIDIA Dynamo built from git hash **`81d0555`** (aarch64 wheels built in-cluster),
  NATS request plane, operator-less Deployments/StatefulSets, **mooncake** KV transfer
- Spec decode: **EAGLE MTP** (num-steps 3, eagle-topk 1, draft-tokens 4) on decode workers
- MoE: `megamoe` a2a backend (DEP points), `flashinfer_mxfp4` runner (point 1)
- Benchmark: sa-bench with `--use-chat-template` +
  `sa_bench_tokenizers.sglang_deepseek_v4.SGLangDeepseekV4Tokenizer` (recipe-required),
  ISL 8192 / OSL 1024, range-ratio 0.8, ignore-eos, req_rate inf
- **Methodology (per directive, exceeds srt-slurm defaults)**: warmup 3x conc @ rate 250 ->
  long-run warm 5x conc (discarded) -> measured **10x conc**
- Metrics convention: input tok/s / prefill GPUs, output tok/s / decode GPUs, total / all GPUs

**IMAGE DEVIATION (documented)**: the recipe pins `lmsysorg/sglang:nightly-dev-20260527-14f81a67`,
which has been **deleted from Docker Hub**. Nearest surviving nightly
**`nightly-dev-20260601-373cadc9`** was used (+5 days of sglang master). See
`../dsv4-sweep/PLAN.md` error log.

## Summary
""",
"| Point | Topology | GPUs (P/D) | Conc | Total tok/s/gpu | In tok/s per prefill GPU | Out tok/s per decode GPU | TPOT mean | TTFT med | E2E mean |",
"|---|---|---|---|---|---|---|---|---|---|"]

for pid, d, tot, pg, dg in rows:
    intps = d['total_input_tokens'] / d['duration']
    doc.append(f"| {pid} | {POINTS.get(pid,('','?'))[1]} | {tot} ({pg}/{dg}) | {d['max_concurrency']} | "
               f"{d['total_token_throughput']/tot:.1f} | {intps/pg:.0f} | {d['output_throughput']/dg:.1f} | "
               f"{d['mean_tpot_ms']:.2f} ms | {d['median_ttft_ms']/1000:.1f} s | {d['mean_e2el_ms']/1000:.1f} s |")

doc.append('\n## Detailed results\n')
for pid, d, tot, pg, dg in rows:
    intps = d['total_input_tokens'] / d['duration']
    recipe, topo = POINTS.get(pid, ('?', '?'))
    doc.append(f"### {pid} — `{recipe}`\n")
    doc.append(f"{topo}; {tot} GPUs ({pg} prefill / {dg} decode); concurrency {d['max_concurrency']}; "
               f"{d['completed']}/{d['num_prompts']} completed in {d['duration']:.0f}s\n")
    doc.append('| Throughput | Value |')
    doc.append('|---|---|')
    doc.append(f"| Total token throughput per GPU | **{d['total_token_throughput']/tot:.1f} tok/s/gpu** |")
    doc.append(f"| Input token throughput per prefill GPU | **{intps/pg:.1f} tok/s/gpu** |")
    doc.append(f"| Output token throughput per decode GPU | **{d['output_throughput']/dg:.1f} tok/s/gpu** |")
    doc.append(f"| (aggregate total / input / output) | {d['total_token_throughput']:.0f} / {intps:.0f} / {d['output_throughput']:.0f} tok/s |")
    doc.append(f"| Request throughput | {d['request_throughput']:.2f} req/s |")
    doc.append('')
    doc.append('| Latency | mean | median (p50) | p90 | p95 | p99 | std |')
    doc.append('|---|---|---|---|---|---|---|')
    t = d.get('ttfts'); i_ = d.get('itls')
    p90t = f"{pct(t,90):.0f}" if t else '-'; p95t = f"{pct(t,95):.0f}" if t else '-'
    p90i = f"{pct(i_,90):.1f}" if i_ else '-'; p95i = f"{pct(i_,95):.1f}" if i_ else '-'
    doc.append(f"| TTFT (ms) | {d['mean_ttft_ms']:.0f} | {d['median_ttft_ms']:.0f} | {p90t} | {p95t} | {d['p99_ttft_ms']:.0f} | {d['std_ttft_ms']:.0f} |")
    doc.append(f"| TPOT (ms) | {d['mean_tpot_ms']:.2f} | {d['median_tpot_ms']:.2f} | - | - | {d['p99_tpot_ms']:.2f} | {d['std_tpot_ms']:.2f} |")
    doc.append(f"| ITL (ms) | {d['mean_itl_ms']:.1f} | {d['median_itl_ms']:.1f} | {p90i} | {p95i} | {d['p99_itl_ms']:.1f} | {d['std_itl_ms']:.1f} |")
    doc.append(f"| E2E latency (s) | {d['mean_e2el_ms']/1000:.2f} | {d['median_e2el_ms']/1000:.2f} | - | - | {d['p99_e2el_ms']/1000:.2f} | {d['std_e2el_ms']/1000:.2f} |")
    doc.append('')

doc.append("""## Notes

- **MTP effect**: point 1 (conc 1) shows TPOT **4.47 ms** (~224 tok/s single-user) vs DSR1's
  6.25 ms at the equivalent low-latency point — EAGLE speculative decoding delivering ~40%
  faster per-token latency.
- **Per-GPU efficiency vs DSR1**: DSv4 point 3 reaches ~805 out tok/s per decode GPU with only
  12 GPUs at conc 256, comparable to DSR1 mid_curve's 816 at 72 GPUs / conc 2048.
- All runs used the 3-stage warm methodology; raw JSONs in `../dsv4-sweep/results/`, server and
  client logs in the GCS bucket under `server-logs/dsv4-*` and `client-logs/dsv4-bench-*`.
- Errors encountered and fixed during bring-up are catalogued in `../dsv4-sweep/PLAN.md`.
""")


# --- InferenceX comparison (shared implementation) ---
# Uses reference/inferencex-published-dsv4-gb300.csv, which carries full latency distributions.
# Supersedes the old inline block, which had two defects:
#   (a) it read the JSON reference (throughput + MEDIAN interactivity only), and
#   (b) it compared OUR mean-derived interactivity against THEIR median -> understated us.
# See compare_inferencex.py for the unit fix (their "(ms)" columns are seconds).
from compare_inferencex import build_sections

doc.append('')
doc.extend(build_sections(standalone=False))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, 'w').write('\n'.join(doc))
print(f"wrote {OUT} ({len(rows)} points)")
