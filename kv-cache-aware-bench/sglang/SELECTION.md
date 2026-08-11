# Selected data points — round-robin vs KV-aware routing (DynoSim)

Kimi-K2.5-NVFP4 · GB300 · session-interleaved Weka 256k trace (4,000 requests) ·
policies: RR vs best KV variant per cell (defaults + tuned scale/credit-decay grid).
trtllm constants silicon-calibrated (v5 lineage); sglang constants = sgl-sim v1 seed
(AIC-ratio transfer, see WORKPLAN Phase S2) — recalibrate after first live sglang points.

Selection rules (inherited from the trtllm 24-GPU study):
1. No SLA gate on the sweep itself; the 5 s TTFT p95 recipe goodput line is used only
   to locate the knee (pre-knee = bounded queue, p95 under budget).
2. **Headline** = the cell that maximizes KV throughput while KV stays pre-knee,
   contrasted with RR's own knee cell — this exposes the *knee shift*, which is where
   the routing impact concentrates at scale.
3. **Conservative** = best cell with BOTH policies pre-knee — honest same-conditions
   comparison; at 72 GPUs this shows throughput parity and isolates the latency win.

## Disaggregated

| Scale | Backend | Point | KV (best variant) | RR | Gain |
|---|---|---|---|---|---|
| 24 GPU | trtllm | headline 3:3 / conc 48 | 2,220 tok/s · 0.41 s p95 · 79% reuse | 2,109 tok/s · 4.65 s p95 · 9% reuse | 1.05× thr, 11× p95 |
| 24 GPU | sglang | headline 3:3 / conc 96 | 4,347 tok/s · 4.01 s p95 · 78% reuse | RR knee is conc 32 → 2,292 tok/s | **1.90× thr, 3× conc** at 5 s budget |
| 24 GPU | sglang | conservative 4:2 / conc 64 | 3,484 tok/s · 0.26 s p95 · 84% reuse | 3,405 tok/s · 3.66 s p95 · 38% reuse | 1.02× thr, 14× p95 |
| 72 GPU | trtllm | headline 6:12 / conc 384 | 5,011 tok/s · 0.84 s p95 (kv-t decay 0.6) | RR knee is conc 96 → 3,533 tok/s (at 384: 4,068 · 56.8 s) | **1.42× thr, 4× conc** at 5 s budget |
| 72 GPU | trtllm | conservative 12:6 / conc 192 | 4,166 tok/s · 0.32 s p95 · 84% reuse | 4,116 tok/s · 2.83 s p95 · 27% reuse | 1.01× thr, 8.8× p95 |
| 72 GPU | sglang | headline 6:12 / conc 384 | 6,993 tok/s · 1.47 s p95 (kv defaults) | RR knee is conc 96 → 4,867 tok/s | **1.44× thr, 4× conc** at 5 s budget |
| 72 GPU | sglang | conservative 12:6 / conc 192 | 5,811 tok/s · 0.27 s p95 · 84% reuse | 5,729 tok/s · 2.96 s p95 · 27% reuse | 1.01× thr, 11× p95 |

Rationale notes:
- **Why the headline is a knee-shift, not a same-cell ratio.** Pre-knee, prefill capacity
  absorbs RR's re-prefill overhead, so throughput gains are small (1.01–1.05×). The
  impact of KV-aware routing is that it *keeps the system pre-knee 3–4× longer*: reuse
  stays at 79–84% vs RR's 9–38% (fragmentation ~1/P across prefill workers), so the
  prefill queue that drives RR past its knee never builds. Comparing each policy at its
  own best bounded-TTFT operating point is the deployment-relevant comparison — you
  would never run RR at conc 384 (57 s TTFT).
- **72-GPU regime.** Aggregate KV capacity (18 workers × 11.9M tokens ≈ 214M) exceeds
  the trace's unique working set (119M tokens): under KV-aware partitioning virtually
  the whole trace is cacheable, while RR still fragments — the KV-vs-RR concurrency
  gap *widens* with scale (counterintuitive but mechanical).
- **Split choice.** Bounded-TTFT optimum is prefill-lean (6:12) for the headline —
  KV routing's reuse makes heavy prefill capacity unnecessary; the conservative
  both-pre-knee cells sit at 12:6 / 9:9 because RR needs the extra prefill workers to
  stay under budget. 3:15 gives the global unbounded max (7,152 tok/s sglang) but at
  19–25 s TTFT — reported as ceiling, not selected.
- **sglang vs trtllm.** ~1.4× higher bounded-TTFT ceilings from the seed constants
  (prefill 78k vs 65k tok/s/worker, decode base 10.3 vs 14.3 ms). Same knee physics;
  at 72 GPUs KV *defaults* already capture most of the win (kv-nvda wins several
  cells) — router tuning matters most at 24 GPU where cache pressure forces evictions.

## Aggregated (24 GPU, 6 × TP4 workers)

| Backend | Point | KV | RR | Gain |
|---|---|---|---|---|
| trtllm | conc 48 (pre-crossover) | 1.25× RR throughput | — | RR wins above conc ~128 (crossover) |
| sglang | **conc 16** (peak of both curves) | 1,172 tok/s · 0.86 s p95 | 775 tok/s · 6.30 s p95 | **1.51× thr, 7.3× p95** |

Rationale notes:
- **sglang conc 16 is the throughput peak for BOTH policies** — the AIC-derived batch
  cliff (TPOT ≈2.5× at per-worker batch ≥ 8, 137k ISL) makes deeper concurrency
  counterproductive, so unlike trtllm there is no deep-batch regime to trade against.
- **No RR crossover on sglang** (trtllm agg had one above conc ~128): the cliff
  penalizes RR's recompute-inflated batches more than KV's concentrated ones.
- **RR never meets the 5 s p95 goodput line at any concurrency** on sglang agg —
  its ~6.3 s p95 is recompute cost (137k cold prefill at 23.7k tok/s), not queueing.
  On this backend, KV-aware routing is what makes agg serving of this trace
  SLA-feasible at all.

## Curves

All panels on the learnings dashboard (artifact `d54ddb6a`): trtllm 24-GPU panels 1–8,
aggregated A1–A6, 72-GPU trtllm D1–D4, sglang S1–S7 (S7 = cross-backend frontier,
same simulator + trace with backend constants swapped).

Data: `results/dynosim_sgl_disagg24_v1.csv`, `results/dynosim_sgl_disagg72_v1.csv`,
`results/dynosim_sgl_agg_v1.csv`, `../aic-results/dynosim_disagg72_v1.csv`.
