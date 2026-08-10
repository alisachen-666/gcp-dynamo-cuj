# Kimi 2.5 KV-Aware Routing Execution Plan Prompts

## System & Model Architecture Parameters
- **Model:** Kimi 2.5 (1T MoE, 32B active, FP8 precision, ~35 KB/token MLA compressed KV cache, 256k max context length)
- **Target Dataset:** `semianalysisai/cc-traces-weka-062126-256k`
- **NVIDIA Reference Recipe:** `recipes/kimi-k2.5`
- **Dynamo KV Router Formula (verified against source, `ai-dynamo/dynamo` v1.3.1, `lib/kv-router/src/scheduling/selector.rs::worker_logit`):**

  ```
  worker_score = PREFILL_LOAD_SCALE × max(0, prefill_blocks − OVERLAP_CREDIT × decay × overlap_blocks)
              + decode_blocks                                          # lower score wins
  decay = 1 / (1 + CREDIT_DECAY × normalized_excess_prefill_load)      # 1.0 when CREDIT_DECAY=0
  ```

  where the CAPITALIZED terms are our to-be-tuned flags:

  | Flag | Symbol | Default (= KV-NVDA arm) | Strategy C |
  |---|---|---|---|
  | `--router-prefill-load-scale` | `PREFILL_LOAD_SCALE` | 1.0 | 2.0 (weight prefill/TTFT cost 2×) |
  | `--router-kv-overlap-score-credit` | `OVERLAP_CREDIT` | 1.0 | sweep 0.7–1.0 (Step 1), 0.5–0.85 (Step 2); **hard-validated ≤ 1.0** — frontend refuses to start above it |
  | `--router-kv-overlap-score-credit-decay` | `CREDIT_DECAY` | 0.0 (off) | **primary Step 2 lever (2026-08-09):** sweep {0.5, 1, 2, 4} — deterministic hot-worker cache-vs-load tradeoff; needs `--router-track-prefill-tokens` (default true) |
  | `--router-temperature` | — | 0.0 | **pinned 0.0 in all steps** (2026-08-09: deterministic argmin; stochastic spreading dropped in favor of credit decay) |
  | `--router-queue-policy` | — | fcfs | fcfs (Steps 1–2); wspt as optional ablation on the Step-2 winner (orders queue by `(1+priority)/new_tokens` after overlap subtraction) |

  Notes: `prefill_blocks`/`overlap_blocks` are in KV block units (block size 32 for us); `decode_blocks` is the worker's potential active decode blocks; host/disk cache-hit weights (defaults 0.75/0.25) are inert in our runs — KV offloading is disabled. Semantics: a worker holding more of the request's prefix gets a lower score and wins; `PREFILL_LOAD_SCALE > 1` makes un-cached prefill work costlier, pushing the router harder toward cache hits and lighter-loaded workers.
- **Strategy C Tuning Logic:**
  - **Step 1 (TTFT SLA Lock):** Set `--router-prefill-load-scale 2.0`, `--router-queue-policy fcfs`, `--router-temperature 0.0`, and sweep `--router-kv-overlap-score-credit` (0.7 to 1.0).
  - **Step 2 (Throughput Maximization, revised 2026-08-09):** Keep `--router-temperature 0.0` (deterministic selection throughout — no stochastic spreading) and instead use `--router-kv-overlap-score-credit-decay` as the load-balance lever: sweep {0.5, 1.0, 2.0, 4.0} at the Step-1-winning credit. Decay shrinks the cache-affinity credit only on workers whose active prefill backlog exceeds the least-loaded worker (decay=1 halves the credit at one request-equivalent of excess load), so cache locality yields to balance exactly when a worker is hot — a deterministic alternative to temperature sampling. Requires `--router-track-prefill-tokens` (default true — no extra flag). Optionally ablate `--router-queue-policy wspt` vs fcfs on the decay winner.
- **Routing Comparison Requirement:** Every topology (24-GPU Aggregated, 24-GPU Disaggregated, 72-GPU Disaggregated) runs BOTH `--router-mode round-robin` (baseline) and `--router-mode kv` on the same Weka trace, so KV-aware perf gain is directly measurable per topology.
- **KV-Aware Config Variants (Phases 1 & 2):** KV-aware routing runs in TWO variants so recipe-default vs. self-tuned gain is separable:
  - **KV-NVDA:** `--router-mode kv` with router parameters exactly as shipped in `recipes/kimi-k2.5` (no tuning).
  - **KV-Tuned:** `--router-mode kv` with our Strategy C self-tuned parameters (Steps 1–2 sweeps).
- **Execution Hardware:** CMCS cluster, `a4x-max` machine type (GB300 NVL72). Live benchmarks and DynoSim simulation runs both execute on this cluster.
- **No Speculative Decoding (2026-08-08):** Eagle3 is stripped from every arm — nvfp4 base model only. The NVDA-recipe arms use the recipe's engine configs minus `speculative_config`; this is a tracked deviation recorded in `environment_lock.json`.
  - **Exception (2026-08-09): recipe-direct 1A arms.** Two aggregated arms run NVDA's config AS SHIPPED, Eagle3 included, to anchor the study against the recipe's own published comparison: **1A-rr** (`agg-eagle-round-robin`) and **1A-kv** (`agg-eagle-kv-router`), both at the recipe's concurrency 24. Only environment adaptations applied (image pin, local weight paths, tolerations, RDMA claims, `max_seq_len` 262144 for trace compatibility). Prerequisite: `nvidia/Kimi-K2.5-Thinking-Eagle3` draft weights staged at `/model-cache/alisachen/Kimi-K2.5-Thinking-Eagle3` (gated on HF — mirror to the bucket like the base model). Eagle-vs-no-Eagle (1A-rr vs 1B, 1A-kv vs 1C) also isolates the speculative-decoding contribution.
- **No KV Offloading (2026-08-08):** host-memory KV offload (`host_cache_size`, `secondary_offload_min_priority` on disagg prefill) is stripped from every arm — KV lives in GPU memory only. `enable_block_reuse` (on-GPU prefix cache) is retained since KV-aware routing depends on it. Tracked deviation in `environment_lock.json`.
- **Deployment Mode:** Phase 2 disagg arms run operator-less (plain Deployments in `manifests/operatorless/`, per-arm `DYN_NAMESPACE` isolation over shared etcd/NATS) since cluster RBAC blocks the Dynamo operator. Phase 1 agg arms (2-node TP8/EP8 workers) need the operator + Grove — pending admin RBAC.
- **GB300-Native First Round (2026-08-09):** first-round bench arms use engine configs derived for GB300 by aiconfigurator (memory fractions, batch/token caps, CUDA-graph lists, MoE/GEMM backends) — NOT the recipe's GB200-tuned values, which remain only in the recipe-direct 1A reference arms. Two solve brackets per topology: warm (`--prefix 133000`, trace reuse level) and cold (relaxed TTFT — the cold solve proved 137k-token cold prefill cannot meet 5 s TTFT on 24 GPUs, i.e. this workload is servable only via prefix caching).
- **Simulation Policy (2026-08-09):** ALL simulation tasks (aiconfigurator, DynoSim) run with `--database-mode SILICON` (measured rows only; coverage gaps are reported, never extrapolated over) and execute in parallel with live cluster work, never serialized behind it.

---

## Phase 0: Environment Locking & Dual Dataset Setup

```text
Act as an MLOps & Data Engineer. Your task is to prepare the environment and both benchmark datasets for Kimi 2.5 on Dynamo.

System & Model Specs:
- Model: Kimi 2.5 (1T MoE, 32B active, FP8 precision, ~35 KB/token MLA cache footprint, 256k context).
- Backend: Dynamo / vLLM / TokenSpeed engine stack.

Instructions:
1. Run `aiconfigurator cli support --model kimi-k2.5 --system gb300` to verify AIConfigurator timing database coverage. Output must return PASS.
2. Lock NVIDIA Reference Recipe Configuration: Pull parameters directly from `recipes/kimi-k2.5` (e.g., `--tensor-parallel-size 4 --enable-expert-parallel`, `--quantization nvfp4`, `--attention-backend trtllm_mla`, `--kv-cache-dtype fp8`, `--dyn-reasoning-parser kimi_k25`).
3. Generate `environment_lock.json` capturing image digests, engine flags, and CUDA runtime parameters.
4. Ingest and convert `semianalysisai/cc-traces-weka-062126-256k` into `weka_256k_trace.jsonl` using `scripts/ingest_weka_trace.py`.

Output a summary confirming both NVIDIA reference recipe parameters and Weka trace JSONL conversion status.
```

---

## Phase 0.5: Smoke Ladder (gate before ANY long benchmark run)

All stages run on GKE. Each stage must PASS before the next; stop at first failure. Any KV
transfer observed off the NVLink/RDMA path stops the run immediately (standing policy).

| Stage | What | Cost | Pass criteria |
|---|---|---|---|
| **S0** | Infra smoke (`manifests/perf/smoke0-infra.yaml`): runtime image pulls + imports on GB300 arm64, etcd/NATS reachable, PVC write/read, traces present | 1 GPU, ~5 min | job completes, `S0 PASS` in log |
| **S1** | Weights + engine up: model download complete; 1 prefill + 1 decode + frontend (arm 2B manifests, `kubectl scale` to 1/1); `/v1/models` serves; 1 manual completion via `curl` | 8 GPUs, ~40 min first time | 200 response, sane tokens |
| **S2** | Mini e2e (`smoke2-mini-e2e.yaml`): AIPerf replays 50 small requests (≤16k input) at conc 2 through the 1P+1D stack; verify KV transfer path is NVLink/RDMA (DynamoBench `common/kv-transport-guard.sh` pattern); artifacts land on PVC | 8 GPUs, ~10 min | 0 errors, artifacts parse, transfer path clean |
| **S3** | Short replay (`smoke3-short-replay.yaml`): full 3P+3D stack, 3 REAL sessions (144 reqs, p50 input 164k, max 255k) at conc 32 — rehearses 256k `max_seq_len`, KV memory headroom, ramp | 24 GPUs, ~15 min | 0 errors, no OOM/evict-thrash, p99 TTFT finite |
| **S4** | Endurance canary: arm 2B full trace at target concurrency, capped at 1h (`BENCHMARK_DURATION=3600` is already the default) — watches for slow leaks, NATS backpressure, gcsfuse stalls before multi-hour sweeps | 24 GPUs, 1h | run-to-cap without degradation trend |

Rerun the ladder from S1 after any engine-config change; S0 only after infra/image changes.

## Adopted Methodology from `pareto-benchmarking-partner.md` (2026-08-09)

The disagg phases adopt the ctx/gen rate-match methodology with one KV-aware adaptation:

1. **Phase 2.5 (new): isolated prefill/decode characterization + rate-match P:D split.**
   Before the full Phase 2 arms, measure `R_prefill(N)` (one TP4 prefill server, decode sink
   with `max_new_tokens: 1`-style negligible generation) and `R_decode(N)` (one TP4 decode
   server, overprovisioned prefill feeders) on np-1, sweeping N geometrically. Retain rows
   per the CTX-only / GEN-only table schemas. Run the Phase-3 rate-match pseudocode with
   `TOTAL_GPUS = 24` (and later 72) to derive the P:D split; cross-check against
   aiconfigurator's recommendation and the recipe's shipped 3P+3D. Validate the winner on
   silicon (= the Phase 2 arms themselves).
2. **KV-aware adaptation — R_prefill is routing-policy-dependent.** With ~97% of trace
   prefill tokens being within-session prefix reuse, effective prefill work is
   `ISL × (1 − hit_rate)`, and hit rate depends on routing and cache spread. Characterize
   prefill BOTH cold (cache flushed each point — conservative bound) and warm (steady-state
   trace replay, separately under RR and KV-aware routing). Rate-match with both bounds; the
   shift in optimal P:D between cold- and warm-matched splits is a reported result.
3. **TTFT certification at operating concurrency.** AIPerf fixed-concurrency replay is a
   closed-loop driver: certify prefill TTFT at `N_op = ceil(N_sys / n_prefill)` per the
   existence-before-gate rule (never substitute a lower-N point). Sweep prefill to just past
   the largest target `N_op`. Expect prefill admission queueing to dominate TTFT at 135k-token
   median inputs; the "high TTFT + stable TPOT" debug signal identifies it.
4. **Acceptance criteria + artifacts.** Every retained bench run must pass the validity
   checks (steady state, zero failed samples, sink/feeder non-backpressure, clean logs,
   percentiles present) and satisfy the run-folder contract (configs, server/frontend/client
   logs, reports, run-index row → GCS `perf/` artifacts).
5. **Visualizer.** Pareto page on `tok/s/user × tps/GPU` with traces: rate-match theory
   (cold + warm), DynoSim projection, and live silicon per routing arm (RR / KV-NVDA /
   KV-tuned); hover metadata + artifact links per the guide; latency-failing points visually
   distinct.

## DynoSim Sweep Methodology — Highlight-Point Selection (2026-08-09)

**Objective:** find the operating point that maximizes the *visible impact of KV-aware
routing* at the largest throughput with decent TTFT, before spending silicon hours.

**Mechanism being exploited:** with P prefill workers, RR lands a session turn on its
prefix-holding worker ~1/P of the time → RR re-prefills ≈ (1−1/P)·context per turn
(cache-size independent). KV-aware holds ~97% affinity. Routing impact therefore grows
with P; warm-traffic throughput wants small P. The highlight point is where prefill
capacity is just-sufficient for KV-aware's effective load but drastically insufficient
for RR's (RR TTFT explodes / throughput collapses while KV cruises).

**Sweep axes (full cross product in DynoSim, minutes per cell):**
- P:D split: {1:5, 2:4, 3:3} on 24 GPUs (later {n_p : 18−n_p} on 72)
- Closed-loop concurrency: {8, 16, 24, 32, 48, 64, 96}
- Policy: {rr, least-loaded, kv-nvda (scale 1.0/credit 1.0), kv-tuned (scale 2.0 ×
  credit {0.7, 0.85, 1.0})}
- Sensitivity: KV capacity ×{0.5, 1, 2} on the top candidates (robustness of the story
  to cache-size assumptions)

**Metrics per cell:** throughput (tok/s), TTFT p50/p95/p99, TPOT, prefix hit rate,
prefill utilization, worker load imbalance. Measured over the steady-state window
(second half of replay; first half warms caches).

**SLA policy (revised 2026-08-10, user decision): NO gate.** Sweep as wide as possible
and hunt max throughput; latency (TTFT p95, TPOT) is always REPORTED next to every
throughput number but never filters the sweep or the selection. The earlier
TTFT ≤ 5 s / TPOT ≤ 20 ms pair survives only as an informational annotation
(`sla_pass` column, hollow marks on the Pareto page) so readers can see where each
point sits relative to the recipe's goodput thresholds. Selection rule: max KV-aware
throughput; RR reported at the same cell for the impact ratio. Concurrency axis
extended (sim: to 384; live: 32–256 at 1800 s per point) to find the rollover.

**Two impact framings (both computed; report both):**
1. **Iso-config gap** (the dramatic one): same P:D + conc, KV vs RR — expected outcome:
   KV passes SLA, RR blows TTFT. Headline: "at this deployment, KV-aware routing is the
   difference between meeting SLA and not."
2. **Iso-SLA best-vs-best** (the rigorous one): each policy gets its own best config;
   compare max SLA-passing throughput. Headline: "KV-aware serves X× the throughput at
   equal SLA." Immune to the "you hobbled RR" critique.

**Highlight-point selection rule:** among cells where KV-aware passes the SLA, maximize
`KV_throughput × min(KV_throughput / RR_throughput_at_same_cell, cap=10)` — throughput-
weighted impact, capped so degenerate RR collapse doesn't dominate the choice. Sanity
requirement: the chosen point's RR arm must also be measurable (finite results), so both
arms produce publishable numbers.

**Calibration loop (theory ↔ silicon, per pareto-benchmarking guide):** seed rates from
aic SILICON solves → sweep → validate top-2 candidates with live S3-style short runs →
refit PREFILL_TOKRATE / TPOT curve from measured → re-sweep → lock the highlight point
for official runs. DynoSim runs on the VM, parallel to all cluster work.

## Phase 1: 24-GPU Aggregated Benchmarks (NVDA Recipe + Trace Replay)

```text
Act as an Inference Systems Engineer. Execute live benchmarking on a 24-GPU Aggregated serving cluster running Kimi 2.5, comparing NVIDIA's official reference recipe against custom trace replay.

Cluster Architecture:
- Hardware: 24 GPUs in unified Aggregated topology (all 24 GPUs process both Prefill and Decode).
- Router Scoring Formula: `worker_score = prefill_load_scale × max(0, prefill_blocks − overlap_credit × overlap_blocks) + decode_blocks` (lower wins; see verified formula + flag table at top of this document).

Instructions:
1. Arm 1A (NVIDIA Official Recipe Direct Benchmark — two routing variants, Eagle3 ON):
   - **1A-rr:** deploy `manifests/arm1a-agg-eagle-rr.yaml` (recipe `agg-eagle-round-robin` as shipped). Bench: `manifests/perf/arm1a-agg-eagle-rr-bench.yaml`, concurrency 24.
   - **1A-kv:** deploy `manifests/arm1a-agg-eagle-kv.yaml` (recipe `agg-eagle-kv-router` as shipped). Bench: `manifests/perf/arm1a-agg-eagle-kv-bench.yaml`, concurrency 24.
   - Both retain Eagle3 speculative decoding per the recipe; requires Eagle3 draft weights staged in the model cache. 1A-kv vs 1A-rr reproduces NVDA's own KV-routing comparison on our cluster/trace; 1A vs 1B/1C isolates the Eagle contribution.
   - Execute standard AIPerf workload specification as defined in `recipes/kimi-k2.5/README.md`.
   - Measure baseline TTFT, TPOT/ITL, Tokens/sec/GPU, and KV Cache Hit Rate.

2. Arm 1B (Weka Trace Replay — Round-Robin Baseline):
   - Deploy with `--router-mode round-robin` using `weka_256k_trace.jsonl`, all other engine flags identical to Arm 1C.
   - Measure TTFT (p50/p95/p99), TPOT/ITL, Tokens/sec/GPU, KV Cache Hit Rate, and load balance CV.
   - Flush host/GPU KV cache before the run.

3. Arm 1C (Weka Trace Replay — KV-Aware, NVDA Recipe Router Defaults):
   - Deploy with `--router-mode kv` using `weka_256k_trace.jsonl`, router parameters exactly as shipped in `recipes/kimi-k2.5` (no tuning).
   - Flush host/GPU KV cache before the run.

4. Arm 1D (Weka Trace Replay — KV-Aware, Self-Tuned Strategy C):
   - Deploy with `--router-mode kv` using `weka_256k_trace.jsonl`.
   - Execute Strategy C Step 1: Set `--router-prefill-load-scale 2.0`, `--router-queue-policy fcfs`, `--router-temperature 0.0`, and sweep `--router-kv-overlap-score-credit` (0.7 to 1.0) to lock p99 TTFT SLA.
   - Execute Strategy C Step 2 (revised 2026-08-09): Keep `--router-temperature 0.0`; at the Step-1-winning credit, sweep `--router-kv-overlap-score-credit-decay` {0.5, 1.0, 2.0, 4.0} to maximize throughput; optionally ablate `--router-queue-policy wspt` on the winner.
   - Flush host/GPU KV cache between runs.
   - Report perf gain vs. Arm 1B (round-robin) AND vs. Arm 1C (KV recipe defaults) on identical hardware and trace.

5. Arm 1E (DynoSim Replay Cross-Validation):
   - Replay `weka_256k_trace.jsonl` through DynoSim for 24-GPU Aggregated topology in round-robin and both kv variants to measure simulation delta vs. live Arm 1B/1C/1D results.

6. Export all metrics to `metrics_agg_24gpu.json` keyed by arm (`nvda_recipe`, `weka_round_robin`, `weka_kv_nvda`, `weka_kv_tuned`).
```

---

## Phase 2: 24-GPU Disaggregated Benchmarks & Profiler Calibration

```text
Act as a Performance Calibration Engineer. Benchmark 24-GPU Disaggregated serving on live GKE hardware using both NVIDIA reference configurations and trace replay, followed by Dynamo Profiler calibration.

Cluster Architecture:
- Topology: Disaggregated Prefill & Decode split (e.g., 8 Prefill / 16 Decode derived via `aiconfigurator`).

Instructions:
1. Run `aiconfigurator` to derive the optimal 24-GPU P/D split for Kimi 2.5:
   `aiconfigurator --model kimi-k2.5 --gpus 24 --sla-ttft 200ms --topology disaggregated`
2. Execute NIXL/RDMA KV transfer microbenchmarks to verify inter-node transfer health.
3. Arm 2A (NVIDIA Official Recipe Disaggregated Benchmark):
   - Benchmark Disaggregated topology using NVIDIA's standard recipe parameters from `recipes/kimi-k2.5`.
4. Arm 2B (Weka Trace Disaggregated — Round-Robin Baseline):
   - Run AIPerf using `weka_256k_trace.jsonl` with `--router-mode round-robin`, all other flags identical to Arm 2C. Flush KV cache prior to run.
   - Measure TTFT (p50/p95/p99), TPOT/ITL, Tokens/sec/GPU, KV Cache Hit Rate, and load balance CV.
5. Arm 2C (Weka Trace Disaggregated — KV-Aware, NVDA Recipe Router Defaults):
   - Run AIPerf using `weka_256k_trace.jsonl` with `--router-mode kv` and router parameters exactly as shipped in `recipes/kimi-k2.5` (no tuning). Flush KV cache prior to run.
6. Arm 2D (Weka Trace Disaggregated — KV-Aware, Self-Tuned Strategy C):
   - Run AIPerf using `weka_256k_trace.jsonl` with Strategy C tuned KV routing (`--router-mode kv`, `--router-prefill-load-scale 2.0`, `--router-queue-policy fcfs`, `--router-temperature 0.0`, Step-1 credit winner, then Step-2 credit-decay sweep {0.5, 1, 2, 4}). Flush KV cache prior to run.
   - Report perf gain vs. Arm 2B (round-robin) AND vs. Arm 2C (KV recipe defaults) on identical hardware and trace.
7. Arm 2E (Dynamo Profiler Calibration):
   - Attach Dynamo Profiler during live Arm 2D execution to capture:
     - Prefill execution latency vs. input sequence length curves.
     - Decode ITL distributions.
     - NIXL KV transfer latency per 64-token block.
   - Inject captured latency curves into DynoSim configuration files.
8. Export results to `metrics_disagg_24gpu.json` keyed by arm (`nvda_recipe`, `weka_round_robin`, `weka_kv_nvda`, `weka_kv_tuned`) and calibration profiles to `profiler_calibration_24gpu.json`. Also export the winning per-chip engine configuration (TP/EP layout, quantization, attention backend, KV cache dtype, KV block size, per-chip batch/KV-block budgets) to `perchip_config_24gpu.json` for reuse in Phase 3.
```

---

## Phase 3: 72-GPU Disaggregated — Sim-Derived Config + Live GPU Benchmarks

```text
Act as an Inference Simulation & Deployment Specialist. Scale Disaggregated KV-routing experiments to 72 REAL GPUs. Use DynoSim (calibrated with the Phase 2 Profiler database) to derive the 72-GPU deployment configuration, carry per-chip optimizations forward from the tuned 24-GPU config, then benchmark live with both round-robin and KV-aware routing.

Target Fleet: 72 GPUs Disaggregated (e.g., 24 Prefill / 48 Decode derived via AIConfigurator).
Calibration Data: `profiler_calibration_24gpu.json`.
Per-Chip Config: `perchip_config_24gpu.json`.

Instructions:
1. Config Derivation (DynoSim + Profiler):
   - Configure DynoSim Mocker engine for 72 GPUs, injecting the empirical kernel latency profiles and NIXL transfer delays from `profiler_calibration_24gpu.json`.
   - Execute DynoSim replays across both workload sets:
     - Run 3A-sim: NVIDIA Recipe Workload Spec on 72 GPUs.
     - Run 3B-sim: `weka_256k_trace.jsonl` on 72 GPUs across Round-Robin, Least-Loaded, and Strategy C KV-Aware routing.
   - Sweep P/D splits and router parameters in simulation; select the deployment config that holds the p99 TTFT SLA at maximum Tokens/sec/GPU.
   - Export simulated metrics to `metrics_disagg_72gpu_sim.json` and the chosen deployment config to `config_disagg_72gpu.json`.
2. Per-Chip Optimization Carry-Over:
   - Reuse the per-chip engine settings locked in the winning 24-GPU disaggregated config (`perchip_config_24gpu.json`): TP/EP layout, `--quantization nvfp4`, `--attention-backend trtllm_mla`, `--kv-cache-dtype fp8`, KV block size, per-chip batch/KV-block budgets.
   - Do NOT retune per-chip parameters at 72-GPU scale — only fleet-level parameters (P/D split, router settings) come from the DynoSim sweep.
3. Live 72-GPU Benchmarks (real GPUs, replaying `weka_256k_trace.jsonl` with the config from `config_disagg_72gpu.json`):
   - Verify NIXL/RDMA KV transfer health across all prefill→decode pairs before each run.
   - Run 3A-live: `--router-mode round-robin` (baseline).
   - Run 3B-live: `--router-mode kv` with Strategy C tuned parameters from Phase 2.
   - Flush host/GPU KV cache between runs.
   - Report KV-aware perf gain vs. round-robin at 72-GPU scale.
4. Export live metrics to `metrics_disagg_72gpu_live.json` keyed by arm (`weka_round_robin`, `weka_kv`) containing TTFT (p50/p95/p99), TPOT/ITL, Tokens/sec/GPU, KV hit rate %, and load balance CV. Report the sim-vs-live delta per metric to validate DynoSim scaling fidelity.
```

---

## Phase 4: Synthesis, Pareto Curve & Crossover Matrix Generation

```text
Act as a Systems Performance Analyst. Synthesize benchmarking artifacts from Phase 1, Phase 2, and Phase 3 into a comparative report and Pareto decision matrix.

Instructions:
1. Execute `scripts/generate_summary.py` to summarize results across NVDA Recipe Baselines, Round-Robin baselines, and KV-Aware (Strategy C) replays for all four result sets (24-GPU Agg, 24-GPU Disagg, 72-GPU Sim, 72-GPU Live).
2. The script also emits a KV-aware vs. round-robin perf-gain matrix per topology (Δ TTFT p99, Δ Tok/s/GPU, Δ KV hit rate) — this is the headline comparison.
3. Report DynoSim sim-vs-live delta at 72 GPUs to establish simulator trustworthiness for future capacity planning.
```
