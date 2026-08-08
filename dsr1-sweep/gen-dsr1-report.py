#!/usr/bin/env python3
"""Generate reports/benchmark-report-dsr1-8k1k.md from results/*.json."""
import json, glob, os
import statistics

R = os.path.expanduser('~/dsr1-pareto/dsr1-sweep/results-summary')
OUT = os.path.expanduser('~/dsr1-pareto/reports/benchmark-report-dsr1-8k1k.md')

def pct(arr, p):
    flat = []
    for x in arr:
        flat.extend(x) if isinstance(x, list) else flat.append(x)
    s = sorted(v * 1000.0 for v in flat)
    k = (len(s) - 1) * p / 100.0
    f = int(k); c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)  # s -> ms

def row(d, prefill_gpus, decode_gpus, note=''):
    total_gpus = prefill_gpus + decode_gpus
    in_tps = d['total_input_tokens'] / d['duration']
    lines = []
    lines.append(f"#### Concurrency {d['max_concurrency']}{note} — {d['completed']}/{d['num_prompts']} completed, duration {d['duration']:.0f}s")
    lines.append('')
    lines.append('| Metric | Value |')
    lines.append('|---|---|')
    lines.append(f"| Total token throughput per GPU (all {total_gpus}) | **{d['total_token_throughput']/total_gpus:.1f} tok/s/gpu** |")
    lines.append(f"| Input token throughput per prefill GPU ({prefill_gpus}) | **{in_tps/prefill_gpus:.1f} tok/s/gpu** |")
    lines.append(f"| Output token throughput per decode GPU ({decode_gpus}) | **{d['output_throughput']/decode_gpus:.1f} tok/s/gpu** |")
    lines.append(f"| (aggregate: total / input / output tok/s) | {d['total_token_throughput']:.0f} / {in_tps:.0f} / {d['output_throughput']:.0f} |")
    lines.append('')
    lines.append('| Latency | mean | median (p50) | p90 | p95 | p99 | std |')
    lines.append('|---|---|---|---|---|---|---|')
    t = d.get('ttfts')
    p90t = f"{pct(t,90):.0f}" if t else '-'
    p95t = f"{pct(t,95):.0f}" if t else '-'
    lines.append(f"| TTFT (ms) | {d['mean_ttft_ms']:.0f} | {d['median_ttft_ms']:.0f} | {p90t} | {p95t} | {d['p99_ttft_ms']:.0f} | {d['std_ttft_ms']:.0f} |")
    lines.append(f"| TPOT (ms) | {d['mean_tpot_ms']:.2f} | {d['median_tpot_ms']:.2f} | - | - | {d['p99_tpot_ms']:.2f} | {d['std_tpot_ms']:.2f} |")
    i = d.get('itls')
    p90i = f"{pct(i,90):.1f}" if i else '-'
    p95i = f"{pct(i,95):.1f}" if i else '-'
    lines.append(f"| ITL (ms) | {d['mean_itl_ms']:.1f} | {d['median_itl_ms']:.1f} | {p90i} | {p95i} | {d['p99_itl_ms']:.1f} | {d['std_itl_ms']:.1f} |")
    lines.append(f"| E2E latency (s) | {d['mean_e2el_ms']/1000:.2f} | {d['median_e2el_ms']/1000:.2f} | - | - | {d['p99_e2el_ms']/1000:.2f} | {d['std_e2el_ms']/1000:.2f} |")
    lines.append('')
    return '\n'.join(lines)

def summary_table(rows, prefill_gpus, decode_gpus, notes):
    total = prefill_gpus + decode_gpus
    out = ['| Conc | Total tok/s/gpu | In tok/s per prefill GPU | Out tok/s per decode GPU | TPOT mean | TTFT median | TTFT p99 | E2E mean |',
           '|---|---|---|---|---|---|---|---|']
    for d, note in rows:
        in_tps = d['total_input_tokens'] / d['duration']
        out.append(f"| {d['max_concurrency']}{note} | {d['total_token_throughput']/total:.1f} | {in_tps/prefill_gpus:.0f} | "
                   f"{d['output_throughput']/decode_gpus:.1f} | {d['mean_tpot_ms']:.2f} ms | {d['median_ttft_ms']:.0f} ms | "
                   f"{d['p99_ttft_ms']/1000:.1f} s | {d['mean_e2el_ms']/1000:.1f} s |")
    return '\n'.join(out)

ll = []
for f in sorted(glob.glob(f'{R}/ll-results_concurrency_*_gpus_20_*.json')):
    d = json.load(open(f))
    note = ' (mult10)' if 'mult10' in f else ''
    ll.append((d, note))
ll.sort(key=lambda x: (x[0]['max_concurrency'], x[1]))

doc = []
doc.append("""# DeepSeek-R1 FP4 GB300 8k1k Benchmark Report — GKE A4X Max

**Setup** (strict InferenceX parity — see `../reference/COMPARISON.md`):
- Cluster: `REDACTED-GKE-CLUSTER-OLD` (project REDACTED-GCP-PROJECT), np-2 NVL72 domain, `a4x-maxgpu-4g-metal` (4x GB300/node)
- Image: `lmsysorg/sglang:v0.5.8.post1-cu130-runtime` + `ai-dynamo==0.8.1` wheels (= InferenceX)
- Model: `nvidia/DeepSeek-R1-0528-NVFP4-v2` (FP4, served as `deepseek-ai/DeepSeek-R1`)
- Serving: NVIDIA Dynamo disaggregated prefill/decode, operator-less (etcd+NATS), NIXL KV transfer
- Benchmark: sa-bench (srt-slurm vendored `benchmark_serving.py`), random dataset ISL 8192 / OSL 1024,
  range-ratio 0.8, ignore-eos; warmup before every measured point
- Workload: 8k1k. Metrics convention: throughput normalized per role
  (input / prefill GPUs, output / decode GPUs; total / all GPUs).

## Config 1: 8k1k LOW_LATENCY (official InferenceX point)

Recipe: `srt-slurm recipes/gb300-fp4/8k1k/low_latency.yaml` (verbatim flags).
Topology: 1 prefill worker (1 node, TP=4) + 4 decode workers (1 node TP=4 each) = 5 nodes / 20 GPUs
(4 prefill GPUs, 16 decode GPUs). Spec decode: none (STP). req_rate 300; warmup rate 250.
Base points: warmup 1x conc, measured 4x conc. `(mult10)` points: srt-slurm defaults
(warmup 2x conc, measured 10x conc) — methodology-matched to InferenceX.

### Summary
""")
doc.append(summary_table(ll, 4, 16, None))
doc.append('\n### Detailed results\n')
for d, note in ll:
    doc.append(row(d, 4, 16, note))

mc = []
for f in sorted(glob.glob(f'{R}/m4-results_concurrency_*_gpus_72_*.json')):
    mc.append((json.load(open(f)), ''))
mc.sort(key=lambda x: x[0]['max_concurrency'])
doc.append("""## Config 2: 8k1k MID_CURVE (official InferenceX point)

Recipe: `srt-slurm recipes/gb300-fp4/8k1k/mid_curve.yaml` (verbatim flags).
Topology: 6 prefill workers (1 node TP=4 each) + 1 decode worker (12 nodes, TP=EP=DP=48,
DeepEP low_latency, dp-attention) + 10 frontends = 18 nodes / 72 GPUs (24 prefill, 48 decode).
Spec decode: none (STP). Methodology: warmup 2x conc @ rate 250, measured 10x conc @ rate 700.
Points 2048/4096 gated on conc-512 review.

### Summary
""")
if mc:
    doc.append(summary_table(mc, 24, 48, None))
    doc.append('\n### Detailed results\n')
    for d, note in mc:
        doc.append(row(d, 24, 48, note))
doc.append("""
## Mid_curve vs InferenceX + tuning conclusions (2026-07-30)

- InferenceX conc-2048 reference: **838.8 out tok/s/gpu**. Ours: 816.5 (base) / 813.2 (tuned)
  at conc 2048 (-2.7%), but **837.4 at conc 4096 — matching their ceiling**.
- Environment tuning experiment (frontends moved from 2 small system nodes to GPU-node spare
  cores; `--reuse-http-connections`; 32-CPU client): output throughput UNCHANGED (-0.4%,
  within noise) while **TPOT improved 5.3%** (18.71 -> 17.72 ms). Interpretation: token
  delivery got faster but conc-2048 completion rate is bounded by prefill/queue dynamics —
  the residual gap vs InferenceX is a **throughput-knee position difference**, not a decode
  deficiency; our system reaches the same ceiling one concurrency step later.
- Remaining candidates if the conc-2048 point must match exactly: run-to-run variance band
  (repeat 3x), prefill-side profiling (their TTFT curve unknown), hostNetwork frontends.

## KV-transfer transport: MNNVL (ours) vs RDMA (InferenceX Slurm) — conclusion

**Context.** Our cluster's RoCE RDMA fabric is non-functional (missing GKE multi-network
config; see `gke-rdma-issue-REDACTED-GKE-CLUSTER-OLD.md`), so NIXL/UCX carries prefill->decode KV cache over
**cuda_ipc across MNNVL (NVLink)**. InferenceX's Slurm NVL72 racks have working IB HCAs and no
UCX restrictions, so their NIXL/UCX most plausibly selects **GPUDirect RDMA (rc_mlx5)** for
inter-node CUDA transfers (UCX does not enable cross-node cuda_ipc by default;
`UCX_CUDA_IPC_ENABLE_MNNVL` is opt-in, which we set explicitly).

**Measured evidence (ours).** UCX protocol tables (UCX_PROTO_INFO=y) during live benchmark
traffic: CUDA-memory transfers of ALL sizes select `zero-copy cuda_ipc/cuda`; tcp appears only
on control lanes (`rma_am/amo_am/keepalive`). Archived: `../dsr1-sweep/results/m2-transport-evidence-ucx-proto.txt`.
MNNVL fabric measured at 723-730 GB/s busbw (2-node and full 72-GPU clique).

**Impact analysis: transport choice does NOT materially affect output throughput.**
1. KV traffic is negligible vs either fabric: at conc 64 we complete ~4.3 req/s x ~280 MB
   MLA KV = ~1.2 GB/s total ingress (~330 MB/s per decode worker) — <0.5% of NVLink
   (700+ GB/s measured) or of 8x400G RoCE (~400 GB/s). Per-request transfer: ~1-3 ms on
   either path inside a 6+ s request.
2. Neither path consumes decode compute: cuda_ipc rides GPU copy engines (DMA), RDMA rides
   the NIC — no SM contention. NVLink contention with TP allreduce from KV ingress is ~0.02%
   of per-GPU NVLink bandwidth.
3. Transport latency lands in TTFT (once per request), not TPOT; output throughput at fixed
   concurrency is governed by TPOT + slot refill. If anything MNNVL's higher bandwidth/lower
   latency marginally FAVORS our setup for the KV hop.

**The observed conc-64 gap vs InferenceX is better explained by measurement methodology**:
our base points measured 4x conc prompts vs srt-slurm's default 10x conc (+2x conc warmup) —
shorter runs overweight ramp/drain edges and understate steady-state throughput.
**CONFIRMED (2026-07-30): the conc-64 mult10 rerun (640 prompts, srt-slurm-exact methodology)
measured +11.1% output tok/s/decode-gpu (272.9 -> 303.1) with TPOT flat (+1.4%)** — the gap
was measurement methodology, not transport. TPOT invariance is the empirical proof that the
MNNVL KV path costs no decode throughput. Residual gaps after methodology matching would
point to prefill-side differences (TTFT curves), not the KV transport.

**Caveat for scale-out**: at mid_curve (72 GPUs, conc 4096, req_rate 700) KV ingress grows
~10-20x; still low single-digit % of fabric capacity — conclusion expected to hold, and will
be re-verified with the same UCX_PROTO_INFO evidence on the mid_curve run.
""")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, 'w').write('\n'.join(doc))
print(f"wrote {OUT}")
