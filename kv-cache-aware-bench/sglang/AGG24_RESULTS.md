# 24-GPU Aggregated Serving: KV-Aware vs Round-Robin — Results, Configs, Reproduction

The first fully-validated silicon comparison of the study (2026-08-14). sglang
backend, native Weka-trace replay, KV routing verified active. This document is
the self-contained record: results + analysis + every config/script that
produced them.

## Setup (fixed across both arms)

- **Hardware**: 6 GB300 nodes on one NVL72 pool (nodepool-2), 4 GPUs each = 24 GPUs.
- **Serving**: 6 independent single-node TP4/EP4 sglang workers
  (`lmsysorg/sglang:v0.5.14-cu130-runtime` + `ai-dynamo[sglang]==1.3.1`),
  NVFP4 Kimi-K2.5, `--mem-fraction-static 0.85 --context-length 262144
  --page-size 64 --chunked-prefill-size 16384 --max-running-requests 16
  --kv-events-config '{"publisher":"zmq",...}' --request-plane nats --host 0.0.0.0
  --watchdog-timeout 1000000` (full args in `manifests/sgl-agg-*.yaml`).
- **Network**: TP collectives on intra-node NVLink; frontend↔worker over NATS
  (16 MB max_payload); KV events over ZMQ; RDMA NICs allocated but unused (no
  KV transfer in agg).
- **Router (the only difference between arms)**:
  - RR: `--router-mode round-robin`
  - KV: `--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs`
    (sweep-selected: defaults beat every tuned variant at agg operating points)
- **Workload**: `--public-dataset semianalysis_cc_traces_weka_062126_256k`
  (393 traces / 68.3k requests, deterministic per-(trace_id, hash_id) synthesis),
  `--num-dataset-entries 393`, concurrency-driven (`--ignore-trace-delays`),
  `ignore_eos` for OSL fidelity, `--max-context-length 262144`.
- **Protocol**: 900 s trace-replay cache warmup (artifacts discarded), then
  1800 s measured window per concurrency, ladder 8→16→24→32, seed 42.
  Zero request errors at every point, both arms.
- **Runs**: `gs://alisachen-models/perf/1786664268_alisachen-sgl-agg-rr-bench/`,
  `.../1786665995_alisachen-sgl-agg-kv-bench/`; summaries in `results/silicon/`.

## Results

| conc | KV tok/s | KV tok/s/GPU | KV TTFT p50/p90/p95/p99 (s) | RR tok/s | RR tok/s/GPU | RR TTFT p50/p90/p95/p99 (s) | thr gain |
|---|---|---|---|---|---|---|---|
| 8  | 1,140 | 47.5 | 0.38 / 1.23 / 1.94 / 7.1  | 964   | 40.2 | 0.60 / 2.60 / 4.65 / 11.4 | 1.18× |
| 16 | 1,748 | 72.8 | 0.40 / 1.11 / 1.82 / 4.1  | 1,253 | 52.2 | 0.60 / 3.54 / 6.18 / 11.7 | 1.39× |
| 24 | 1,881 | 78.4 | 0.42 / 1.40 / 2.02 / 4.3  | 1,379 | 57.5 | 0.63 / 4.78 / 7.86 / 12.5 | 1.36× |
| 32 | 2,119 | 88.3 | 0.44 / 1.51 / 2.57 / 5.6  | 1,398 | 58.2 | 0.72 / 6.73 / 9.52 / 27.2 | **1.52×** |

Decode interactivity (mean-ITL-derived): KV 84/60/50/45 tok/s/user at conc
8/16/24/32; RR 82/62/47/37 — near-identical until RR's queueing bleeds into ITL
at conc 32 (mean 27.0 ms vs KV 22.1 ms).

## Analysis

1. **Throughput/GPU: KV scales, RR saturates.** KV grows 47.5 → 88.3 tok/s/GPU
   across the ladder (86% scaling for 4× concurrency) and is still climbing at
   conc 32. RR flatlines at ~58 from conc 24 — its knee — because re-prefilling
   ~5/6 of every session's context consumes the prefill budget that KV spends
   on new work. The gap therefore *widens* with load: 1.18× → 1.52×.
2. **TTFT: two different regimes, visible in the percentile shape.** KV's p50 is
   flat 0.38–0.44 s across the entire ladder (cache-hit prefill: only new turn
   tokens computed) and its p90 stays ≤1.5 s. RR's p50 grows 0.60 → 0.72 s
   (always paying near-full recompute) while its p90/p95 blow out 2.6→6.7 s /
   4.7→9.5 s — queue-wait stacking on recompute. At conc 32 RR's p99 is 27 s:
   the queue-divergence signature of a policy past its knee.
3. **Goodput framing (5 s TTFT p95 threshold)**: RR is compliant only at conc 8;
   KV is compliant at every measured point. At the SLA boundary the honest
   summary is: **KV serves 4× the concurrency and 2.2× the throughput of RR's
   last compliant point** (2,119@32 vs 964@8).
4. **Sim vs silicon**: RR-side prediction was excellent (p95 at conc 16: 6.18 s
   measured vs 6.30 s simulated; knee location right). KV-side, the sim was
   directionally right but conservative on throughput scaling (predicted a
   conc-16 peak; silicon still climbs at 32 — real reuse cuts per-request
   prefill more than the seed constants assumed) and optimistic on tail
   (predicted 0.86 s p95, measured 1.8–2.6 s). Calibration v2 items: raise
   sub-cliff throughput slope, add tail dispersion.
5. **Context vs NVIDIA's GB200 recipe (24 GPUs)**: not directly comparable
   (trtllm + Eagle3 + synthetic trace there), but structurally: their table
   credits KV routing with +10% throughput and *negative* interactivity impact;
   with faithful reuse-bearing traffic and no speculative-decode confound, we
   measure +36–52% with KV winning every latency percentile. Workload fidelity
   is the difference — see `INFERENCEX_PIPELINE.md` §3.
6. **Open item**: extend the ladder to conc 48/64 to find KV's true knee (both
   curves unsaturated at 32); re-select the headline operating point there.

## How these points were chosen: the AIC → DynoSim → silicon chain

This section records the tuning/sweeping work *behind* the results table — how the
operating points were selected in simulation, how silicon validated them, and how
the live sweep was designed to isolate the KV-routing effect.

### 1. AIC (engine level): what should one agg worker be?

`aiconfigurator cli default … --backend sglang --total-gpus 24 --isl 137000 --osl
1100 --database-mode SILICON` (warm bracket `--prefix 133000`; cold bracket no
prefix, TTFT relaxed to 30 s). Outcomes that shaped everything downstream:

- **Worker shape**: 6 independent single-node TP4/EP4 workers (chunked prefill,
  page 64) — the agg fleet both arms run.
- **Cold infeasibility**: a cold 137k-token prefill cannot meet a 5 s TTFT on any
  24-GPU agg configuration → this workload is servable only through prefix reuse.
  That negative result is the *thesis* of the KV-vs-RR comparison: the entire
  question is which router converts the trace's 84.6% block reuse into actual hits.
- **The sglang batch cliff**: AIC's TPOT tables show a ≈2.5× per-token cost jump
  at per-worker batch ≥ 8 for 137k ISL (ratio 0.70 → 2.47 vs trtllm). Encoded in
  the simulator as a piecewise TPOT model; it predicts deep-concurrency agg
  operation is counterproductive — which bounded the ladder.
- **Seed constants for DynoSim** (AIC-ratio transfer onto live-calibrated trtllm
  values): agg prefill 23.7k tok/s/worker (45k × 0.527 cold-floor ratio), TPOT
  7.0 + 1.6·bs below the cliff, 28.4 + 5.68·bs above.

### 2. DynoSim (deployment level): which router, at what load?

Full policy × concurrency grid over the real trace's hash_ids (6 workers,
per-worker radix caches, LRU at measured KV capacity; exact dynamo v1.3.1
`worker_logit` scoring): 11 policies (rr, least-loaded, kv-defaults, kv-tuned ×
credit {0.5…1.0}, scale {1.5, 3.0}) × conc {8…256} — `results/dynosim_sgl_agg_v2.csv`.
Findings that selected the silicon points:

- **Both policies peak at conc ≈ 16** (the batch cliff makes deeper concurrency
  counterproductive for *both* — unlike trtllm agg, there is no deep-batch regime
  where RR catches up, and no RR crossover at any load). Predicted cell: KV 1,172
  tok/s @ 0.86 s p95 vs RR 775 @ 6.30 s — 1.51×.
- **RR is recompute-bound, not queue-bound**: its ~6.3 s p95 at the peak is the
  cost of re-prefilling ~5/6 of each session at 23.7k tok/s, present at *every*
  concurrency — so the sim predicted RR latency-infeasibility as a structural
  property, not an overload artifact. Silicon confirmed (RR p95 4.7–9.5 s
  across the ladder).
- **Router-flag grid: defaults win at every agg operating point.** Tuned variants
  pay only post-knee or in prefill-starved topologies; mis-tuning costs up to
  50% throughput on sglang agg (aggressive prefill-load-scale fights the cliff).
  This is why the silicon KV arm runs defaults + temperature 0 and why the live
  flag-sweep budget was spent on the 72-GPU disagg cell instead, where the grid
  showed tuning *could* matter.
- **Point selection**: ladder 8/16/24/32 — brackets the predicted joint peak at
  16 from both sides, so silicon can confirm or refute both the peak's location
  and the knee shape without trusting the sim's absolutes.

### 3. Silicon validation: how the real jobs test the sim's claims

Two identical 6×TP4 arms (nodepool-2), differing in exactly one flag
(`--router-mode`); native deterministic trace replay; per-point 900 s trace
warmup + 1800 s measurement; KV-routing *verified active* (router-predicted hit
0.80 mean via `dynamo_component_router_kv_hit_rate`, 92% engine-measured reuse)
so a silent load-routing fallback cannot masquerade as a KV result. Verdicts:

| Sim claim | Silicon verdict |
|---|---|
| KV/RR ratio 1.11–1.63× over the ladder | **Confirmed within ~10%** (1.18–1.52×) |
| RR p95 ≈ 6.3 s at conc 16 | **Confirmed** (6.18 s) |
| RR latency-infeasible at every point | Confirmed from conc 16 up (4.65 s at conc 8) |
| Joint throughput peak at conc 16 | **Refuted for KV** — silicon KV still climbs at 32 (real reuse cuts per-request prefill more than seeded); RR knees at 24 as predicted |
| KV tail p95 ≈ 0.9 s | Optimistic — measured 1.8–2.6 s (dispersion missing from sim) |

The two refuted rows became calibration-v2 items (raise sub-cliff throughput
slope, add tail dispersion) — the simulate → verify → refit loop working as
designed, and the reason the sim is trusted for *ratios and knee locations*, not
absolutes.

### 4. The live sweep design: why the ladder isolates KV-routing impact

The silicon sweep axis is concurrency (8 → 32) with everything else frozen — the
router flag is the only difference between arms, so the widening gap (1.18× →
1.52×) is attributable to routing alone. The ladder highlights the impact in
three distinct regimes: at light load (conc 8) both policies are healthy and KV's
gain is pure recompute savings; at mid load RR crosses its knee (conc 24
flatline) while KV keeps scaling; at conc 32 the gap is maximal and RR's p99
(27 s) shows queue divergence. Per-variant cache warmup before every measured
point ensures each policy is measured against cache state *it* produced — an RR
arm measured on KV-warmed caches (or vice versa) would confound the comparison.
Router-flag variants were deliberately *not* swept live at this scale: the sim
grid's verdict (defaults optimal, tuning harmful up to 50%) made live agg
flag-sweeping negative-expected-value; the live flag sweep ran at the 72-GPU
disagg cell where the grid predicted sensitivity.

## Reproduction inventory (paths in this repo)

| Layer | File(s) |
|---|---|
| Cluster bootstrap | `manifests/bootstrap/00-platform.yaml` (ns + etcd + NATS 16MB), `manifests/operatorless/00-storage-static.yaml` (gcsfuse PV/PVC), `manifests/bootstrap/mrdma-claim-template.yaml` |
| Serving arms | `sglang/manifests/sgl-agg-rr.yaml`, `sgl-agg-kv.yaml` (generated by `sglang/scripts/gen_sglang_arms.py`) |
| Bench jobs | `manifests/perf/sgl-agg-{rr,kv}-bench.yaml` (generated by `scripts/gen_perf_jobs.py`; embeds dataset staging + cache warmup + ladder) |
| Orchestration | `sglang/scripts/rebootstrap_bench.sh` (readiness→gates→launch), `monitor_sgl.sh` |
| Verification | `sglang/scripts/verify_kv_routing.sh` + `sglang/KV_ROUTING_VERIFICATION.md` |
| Dataset staging | processed HF cache build: see `gen_perf_jobs.py` comments + `INFERENCEX_PIPELINE.md` |
| Results | `sglang/results/silicon/{KV,RR}-c{8,16,24,32}.json` |
