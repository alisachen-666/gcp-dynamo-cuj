# 24-GPU Aggregated Serving: KV-Aware vs Round-Robin — Results, Configs, Reproduction

The first fully-validated silicon comparison of the study (2026-08-14). sglang
backend, native Weka-trace replay, KV routing verified active. This document is
the self-contained record: results + analysis + every config/script that
produced them.

## Setup (fixed across both arms)

- **Hardware**: 6 GB300 nodes on one NVL72 pool (np-2), 4 GPUs each = 24 GPUs.
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
