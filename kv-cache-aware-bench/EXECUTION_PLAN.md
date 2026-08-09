# Kimi 2.5 KV-Aware Routing Execution Plan Prompts

## System & Model Architecture Parameters
- **Model:** Kimi 2.5 (1T MoE, 32B active, FP8 precision, ~35 KB/token MLA compressed KV cache, 256k max context length)
- **Target Dataset:** `semianalysisai/cc-traces-weka-062126-256k`
- **NVIDIA Reference Recipe:** `recipes/kimi-k2.5`
- **Dynamo KV Router Formula:** `Worker Score = prefill_load_scale * (prefill_blocks - cache_overlap_credit) + decode_blocks`
- **Strategy C Tuning Logic:**
  - **Step 1 (TTFT SLA Lock):** Set `--router-prefill-load-scale 2.0`, `--router-queue-policy fcfs`, `--router-temperature 0.0`, and sweep `--router-kv-overlap-score-credit` (0.7 to 1.0).
  - **Step 2 (Throughput Maximization):** Set `--router-queue-policy wspt`, `--router-temperature 0.3`, and sweep credit down through 0.5–0.85.
- **Routing Comparison Requirement:** Every topology (24-GPU Aggregated, 24-GPU Disaggregated, 72-GPU Disaggregated) runs BOTH `--router-mode round-robin` (baseline) and `--router-mode kv` on the same Weka trace, so KV-aware perf gain is directly measurable per topology.
- **KV-Aware Config Variants (Phases 1 & 2):** KV-aware routing runs in TWO variants so recipe-default vs. self-tuned gain is separable:
  - **KV-NVDA:** `--router-mode kv` with router parameters exactly as shipped in `recipes/kimi-k2.5` (no tuning).
  - **KV-Tuned:** `--router-mode kv` with our Strategy C self-tuned parameters (Steps 1–2 sweeps).
- **Execution Hardware:** CMCS cluster, `a4x-max` machine type (GB300 NVL72). Live benchmarks and DynoSim simulation runs both execute on this cluster.
- **No Speculative Decoding (2026-08-08):** Eagle3 is stripped from every arm — nvfp4 base model only. The NVDA-recipe arms use the recipe's engine configs minus `speculative_config`; this is a tracked deviation recorded in `environment_lock.json`.
- **No KV Offloading (2026-08-08):** host-memory KV offload (`host_cache_size`, `secondary_offload_min_priority` on disagg prefill) is stripped from every arm — KV lives in GPU memory only. `enable_block_reuse` (on-GPU prefix cache) is retained since KV-aware routing depends on it. Tracked deviation in `environment_lock.json`.
- **Deployment Mode:** Phase 2 disagg arms run operator-less (plain Deployments in `manifests/operatorless/`, per-arm `DYN_NAMESPACE` isolation over shared etcd/NATS) since cluster RBAC blocks the Dynamo operator. Phase 1 agg arms (2-node TP8/EP8 workers) need the operator + Grove — pending admin RBAC.

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

## Phase 1: 24-GPU Aggregated Benchmarks (NVDA Recipe + Trace Replay)

```text
Act as an Inference Systems Engineer. Execute live benchmarking on a 24-GPU Aggregated serving cluster running Kimi 2.5, comparing NVIDIA's official reference recipe against custom trace replay.

Cluster Architecture:
- Hardware: 24 GPUs in unified Aggregated topology (all 24 GPUs process both Prefill and Decode).
- Router Scoring Formula: Worker Score = prefill_load_scale * (prefill_blocks - cache_overlap_credit) + decode_blocks.

Instructions:
1. Arm 1A (NVIDIA Official Recipe Direct Benchmark):
   - Deploy using official `recipes/kimi-k2.5` configuration flags directly.
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
   - Execute Strategy C Step 2: Set `--router-queue-policy wspt`, `--router-temperature 0.3`, and sweep credit (0.5 to 0.85) to maximize throughput.
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
   - Run AIPerf using `weka_256k_trace.jsonl` with Strategy C tuned KV routing (`--router-mode kv`, `--router-prefill-load-scale 2.0`, `--router-queue-policy fcfs`, then wspt/temperature 0.3 sweep). Flush KV cache prior to run.
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
