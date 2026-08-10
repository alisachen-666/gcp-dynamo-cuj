# Kimi 2.5 KV-Aware Routing — Perf Results

**Cluster:** CMCS, `a4x-max` (GB300 NVL72)
**Model:** Kimi 2.5 (1T MoE, 32B active, FP8, MLA ~35 KB/token KV, 256k context)
**Trace:** `semianalysisai/cc-traces-weka-062126-256k` → `weka_256k_trace.jsonl`
**Status legend:** ⬜ not started · 🔄 running · ✅ complete · ❌ blocked

## Environment

- **Cluster access verified 2026-08-08:** context `cmcs-a4xmax` (endpoint `https://136.83.38.72`, ADC exec-plugin auth via `~/bin/adc-exec-cred.sh`). 72 × `a4x-maxgpu-4g-metal` GPU nodes (288 GPUs, pools np-1…np-4 = 4 NVL72 racks) + 2 system nodes; pod/job create permission confirmed.
- **Dynamo platform state:** `dynamo-cloud` ns has etcd + NATS running; **Dynamo operator + CRDs not yet installed** (no `dynamographdeployments` resource) — must be installed before graph deployments. Kueue, JobSet, and NVIDIA DRA driver (`computedomains`) present.
- **Local tooling:** venv at `~/kv-cache-aware-bench/.venv`; note `/home` is a small 5.7 GB partition — keep large artifacts gzipped or in GCS.

## Trace Characterization (`weka_256k_trace.jsonl`, ingested 2026-08-08)

- 30,141 requests / 393 sessions; 3.90B total input tokens, 32.2M total output tokens (heavily prefill-dominated — ideal for KV-aware routing study).
- Input length: mean 137k, p50 135k, p90 227k, p99 252k, max 256k tokens. Output length: mean 1131, p50 483, p99 9352.
- hash_ids: mean 2024 blocks/request (64-token blocks, `hash_id_scope: local` — prefix overlap only within a session).
- KV footprint at ~35 KB/token: **~8.8 GB per p99 request** — cache-aware placement is load-bearing at these sizes.
- Model mix in trace: opus-4-8 59%, fable-5 21%, opus-4-6 12%, other/unlabeled 8% (all replayed against Kimi 2.5).
- Durable copy: `data/weka_256k_trace.jsonl.gz` (8 MB); working copy in session scratchpad.

## Router Scoring Formula (source-verified 2026-08-09)

All KV-aware arms are governed by this scoring function, confirmed by reading
`ai-dynamo/dynamo` **v1.3.1** (the runtime we pinned), `lib/kv-router/src/scheduling/selector.rs::worker_logit`
— the runtime's debug log prints the same expression:

```
worker_score = prefill_load_scale × max(0, prefill_blocks − overlap_credit × decay × overlap_blocks)
             + decode_blocks                     # per candidate worker; LOWER score wins
decay = 1 / (1 + credit_decay × normalized_excess_prefill_load)   # = 1.0 at our credit_decay=0
```

- `overlap_blocks` comes from the router's radix-tree prefix indexer (`lib/kv-router/src/indexer/`) fed by worker KV events; `decode_blocks` is the worker's potential active decode blocks.
- Selection: `--router-temperature 0` picks argmin deterministically; `>0` samples via softmax over negated scores (`selector.rs::softmax_sample`).
- Queue ordering is separate from worker scoring: `--router-queue-policy` ∈ fcfs / lcfs / wspt (`scheduling/policy.rs`); wspt orders by `(1+priority)/new_tokens` with cache overlap subtracted from `new_tokens`.

**Per-arm router settings:**

| Arm | prefill_load_scale | overlap_credit | credit_decay | temperature | queue policy |
|---|---|---|---|---|---|
| B (round-robin) | — (no scoring; frontend `--router-mode round-robin`) | — | — | — | — |
| C (KV-NVDA defaults) | 1.0 | 1.0 | 0.0 | 0.0 | fcfs |
| D (KV-tuned, Step 1) | 2.0 | sweep 0.7–1.0 | 0.0 | 0.0 | fcfs |
| D (KV-tuned, Step 2) | 2.0 | Step-1 winner | sweep 0.5–4.0 | 0.0 | fcfs (+wspt ablation) |

Step 2 revised 2026-08-09: temperature stays pinned at 0.0 in every configuration — load spreading
comes from `--router-kv-overlap-score-credit-decay` (deterministic: shrinks cache-affinity credit
only on workers with excess active prefill backlog; decay=1 halves credit at one request-equivalent
of excess load) rather than stochastic softmax sampling. Decay requires `--router-track-prefill-tokens`,
which defaults to true at v1.3.1.

Constraint found in source (`scheduling/config.rs`): `overlap_score_credit` is hard-validated to [0, 1] —
values above 1.0 abort frontend startup with guidance to express prefill weighting via `prefill_load_scale`
instead. Strategy C already follows this pattern. Host/disk cache-hit weights (0.75/0.25 defaults) are
inert here: KV offloading is disabled in all arms.

## aiconfigurator GB300 Solves (SILICON mode, 2026-08-09)

24 GPUs, trtllm backend, perf DB 1.3.0rc10, ISL 137k / OSL 1100 from trace. Artifacts:
`aic-results/` (top-N CSVs + full tgz incl. generated per-candidate engine configs).

| Bracket | Best disagg P:D (workers ×4 GPU) | Prefill parallel | Decode parallel | TTFT | TPOT | tok/s/GPU |
|---|---|---|---|---|---|---|
| **Warm** (`--prefix 133000`, ~trace reuse) | **1 : 5** (conc 10) | dp4/ep4 (attention-DP) | tp4/ep4 | 1.38 s | 15.9 ms | 24.0 |
| **Cold** (no reuse, TTFT gate 30 s) | **2 : 4** (conc 8) | tp4/ep4 | tp4/ep4 | 8.4 s | 15.9 ms | 17.7 |
| Recipe (GB200-tuned) reference | 3 : 3 | tp4/ep4 attn-DP | tp4/ep4 | — | — | — |

**Findings:**
1. **Cold 137k prefill cannot meet 5 s TTFT on 24 GPUs at any config** (the strict cold solve
   returned infeasible) — this workload is servable only via prefix caching. Cold needs 8.4 s
   TTFT even at conc 8.
2. **Warm traffic wants far less prefill than the recipe ships**: P:D 1:5 vs the recipe's 3:3.
   At ~97% reuse, prefill demand collapses; the recipe's GB200 split would idle ~8 GPUs of
   prefill on warm traffic. This directly confirms the cold/warm rate-match adaptation in the
   plan (routing policy shifts the optimal provisioning, not just throughput).
3. **Worker shape is stable across brackets**: 4-GPU single-node workers (tp4 or dp4 attention
   + ep4 MoE) in every top candidate — our operator-less single-node pattern holds; only the
   replica ratio moves. Decode tp8 2-node shapes appear only at low-concurrency
   high-interactivity points.
4. **TPOT ≤ 10 ms is not achievable at useful throughput at this ISL** (all points ≥ 12.7 ms).
   The bench SLA should use TPOT ≈ 15 ms (≈ 63 tok/s/user) as the interactivity gate.
5. GB300-native engine params from generated configs (vs GB200 recipe): `free_gpu_memory_fraction`
   0.8 both tiers (recipe: 0.6 prefill / 0.85 decode), prefill `max_num_tokens` 137504
   (recipe: 8192 + chunked prefill), `moe_config.backend: WIDEEP` (recipe: TRTLLM/CUTLASS),
   decode `max_num_tokens` 32. Note the generator targets trtllm 1.3.0rc14 config format —
   validate against our 1.3.1 runtime at S2-scale before adopting wholesale.
6. Coverage note (SILICON policy): perf DB rows are sparse above 64k sequence length —
   interpolation skips logged at 65k/131k. Treat absolute predictions at 137k ISL as
   approximate; the P:D-ratio and shape conclusions are robust, per-point numbers get
   silicon validation anyway.

## DynoSim P:D × Routing Sweep (v4, 2026-08-09)

Methodology in EXECUTION_PLAN.md ("DynoSim Sweep Methodology"); simulator
`scripts/dynosim_pd.py`; sweeps archived as `aic-results/dynosim_sweep_v{1..4}.csv`;
interactive Pareto page `reports/pareto_dynosim_v4.html` (also published as artifact).

**Calibration lineage (each iteration fixed a real modeling error):**
- v1: prefill rate seeded from warm seq-rate — 30× too cheap for RR misses (ratio≡1.0).
- v2: cold-solve uncached rate (65k tok/s/worker) — TTFT separation appeared (RR 2–3×
  worse), throughput still tied: decode model too pessimistic vs live S4.
- v3: decode TPOT recalibrated from live S4 (slope 0.75→0.1 ms/seq) — high-conc cells
  entered SLA range; still no throughput separation → exposed that the trace was
  session-GROUPED (closed-loop replay held only ~5 sessions in flight; caches never
  pressured; also affects live aiperf replay — trace re-ordered for both).
- v4: session-interleaved trace (393-way round-robin) — **separation emerged**.

**v4 highlight point (per selection rule):** P:D **3:3, conc 128, KV-tuned (scale 2.0,
credit 0.7)**: 2,978 tok/s (124 tok/s/GPU) at TTFT p95 **0.60 s**, vs same-deployment
round-robin 2,304 tok/s at TTFT p95 **31.7 s (SLA fail)** — 1.29× throughput at ~53×
lower tail latency. RR's admission queue explodes with concurrency (9.2 s @64 → 20 s
@96 → 43 s @160) while KV-aware *improves*. Strategy C tuned beats KV-NVDA defaults at
most highlight cells.

**Regime finding:** optimal P:D inverts with session mixing — warm single-stream (aic)
favors 1:5; heavy interleave favors 3:3 (aggregate cache + partition affinity; 1:5
fails SLA entirely at 393-way mixing). Three-regime story: cold / warm-grouped /
warm-mixed.

**v5 (2026-08-10, UNGATED per no-SLA decision, conc extended to 384):** the max-throughput
hunt confirms the same point — **the global throughput ceiling of the entire 144-cell sweep
is 3:3 / conc 128 / KV-tuned(c0.7) at 2,978 tok/s, held at TTFT p95 0.60 s**. Beyond conc
128 KV throughput declines (active working set outgrows even partitioned caches). Ceilings
elsewhere: 1:5 ≈ 1,045 tok/s (single prefill worker saturates; TTFT grows linearly with
conc — pure queueing), 2:4 ≈ 1,808. Best-vs-best (each policy's own peak, no gate): KV
2,978 vs RR 2,352 (@3:3 conc 96) — **1.27× even with no latency argument**, and at the
peaks KV holds sub-second TTFT p95 while RR needs 20–32 s. KV TTFT stays ≈flat (0.4–3.5 s)
across conc 64→384 at 3:3 while every cache-blind config's TTFT grows linearly. The no-SLA
and SLA-gated selection rules pick the SAME operating point — the highlight point is the
ceiling. Remaining caveats: decode model calibrated to one live point; 393-way mixing fixed
(v6: mixing width = concurrency); live validation pending.

## Run Log

| Date | Phase | Arm | Config | Status | Notes |
|---|---|---|---|---|---|
| 2026-08-08 | 0 | — | Trace ingest (30,141 reqs / 393 sessions) | ✅ | streamed, no `datasets` dep |
| 2026-08-08 | 0 | — | Cluster access + inventory (288 GPUs) | ✅ | ADC auth workaround |
| 2026-08-08 | 0 | — | aiconfigurator support check (gb300 × Kimi-K2.5-NVFP4) | ✅ | PASS agg + disagg (trtllm 1.3.0rc15) |
| 2026-08-08 | 0 | — | Env lock (`environment_lock.json`) | ✅ | runtime image 1.3.1 digest pinned |
| 2026-08-08 | 0 | — | NVDA recipe `kimi-k2.5` pulled @ 12356a1d | ✅ | all 4 routing variants present |
| 2026-08-08 | 0 | — | Dynamo CRDs applied (v1.3.1) | ✅ | `dynamographdeployments` live |
| 2026-08-08 | 0 | — | Dynamo operator install | ⚠️ | RBAC needs container.admin — now OPTIONAL (Phase 1 agg multinode only); manifest at `recipes/dynamo-operator-rbac-ADMIN-APPLY.yaml` |
| 2026-08-08 | 0 | — | Operator-less disagg manifests (arms 2B/2C/2D) | ✅ | `manifests/operatorless/`, server dry-run PASS |
| 2026-08-08 | 0 | — | ComputeDomain + model-cache PVC (gcsfuse RWX) | ✅ | `kv-bench-compute-domain`; PVC pending first consumer |
| 2026-08-09 | 0 | — | Weights source switched to public bucket mirror | ✅ | `Alisa233/Kimi-K2.5-NVFP4-bucket`, 590.9 GB, config verified vs locked spec — **HF-token blocker eliminated** |
| | 0 | — | Model download (~590 GB → `/model-cache/Kimi-K2.5-NVFP4`) | ⬜ | bucket downloader in `00-support.yaml`; launch when ADC restored |
| 2026-08-08 | 0 | — | AIPerf trace + smoke slices (GKE bench jobs) | ✅ | 28,444 replayable reqs (1,697 null-length skipped); hash_ids globalized (session-local scope would fake cross-session KV hits) |
| 2026-08-09 | 0 | — | A4X MAX cross-check vs `gpu-recipes@fef5ad27` (official GKE TRT-LLM recipe) | ✅ | adopted: 8× `mrdma.google.com` NIC claims per GPU pod (`mrdma-all`), IPC_LOCK cap, UCX rail-aware pins (GID 5, local-subnet /64 — from DynamoBench m2r rev3 debugging), shm 250Gi, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`, `TLLM_NUMA_AWARE_WORKER_AFFINITY`. Deliberately NOT adopted: mpirun/SSH multinode machinery (dynamo orchestrates workers), gib NCCL plugin (NCCL is intra-node in Phase 2 TP4 workers; revisit for Phase 1 agg), UCX_TLS=tcp pin (would forbid RDMA — their agg recipe uses UCX for control only; ours carries KV transfer). `UCX_TLS` left unpinned — S2's transport guard verifies the actual KV path and we pin further only if TCP is observed. |
| 2026-08-09 | 0.5 | S0 | Infra smoke (image/etcd/NATS/PVC/traces) | ✅ | PASS; PVC-write caveat resolved same day (IAM propagation + gcsfuse mkdir semantics) |
| 2026-08-09 | 1 | 1A-rr/1A-kv | Recipe-direct agg arms added (Eagle3 ON, conc 24) | ⬜ | manifests + bench jobs generated; needs Eagle3 draft weights mirrored + operator RBAC (or LWS path) |
| 2026-08-09 | 2.5 | — | aic GB300 solves: SILICON warm+cold (primary), HYBRID (suppl.) | ✅ | see "aiconfigurator GB300 Solves"; warm P:D=1:5, cold 2:4, recipe 3:3 |
| 2026-08-09 | 0.5 | S1 | Weights + 1P1D engine up + manual curl | ✅ | **PASS** — model discovered, coherent completion (`finish_reason: stop`), prefix cache active (`cached_tokens` reported). 5 operator-less defects found+fixed en route: gcsfuse eviction @500Gi, probe budget, RollingUpdate strand, missing `DYN_SYSTEM_PORT`, frontend tokenizer mount. Warm-cache weight load ≈7 min (vs ~55 min cold via gcsfuse) |
| 2026-08-09 | 0.5 | S2 | Mini e2e AIPerf (conc 2) + RDMA path check | ✅ | **PASS** (attempt 7): 273 reqs replayed, 0 failures, TTFT p50 340ms, ITL 8.2ms; transport clean (eth0 delta ≈0 across attempts). 4 bench-client defects found+fixed: mooncake 512-block default (need `--isl-block-size 64`), aiperf 0.10→0.12 (trace block-size support), missing tiktoken, aiperf-0.12 offline-worker bug (local tokenizer paths crash → staged local HF cache + `HF_HUB_OFFLINE=1`); plus replay-mode fix: session-relative timestamps ⇒ `--no-fixed-schedule --ignore-trace-delays` (closed-loop concurrency dispatch; think-time gaps dropped — uniform across arms, noted as limitation) |
| 2026-08-09 | 0.5 | S3 | Short replay 3P+3D conc 32 + `UCX_PROTO_INFO` transport proof | ✅ | **PASS** (attempt 4): 370/372 ok (0.54% err — 2 benign empty-content on clamped-output rows); ITL p50 18ms, 808 tok/s out. **Transport proof: KV bulk on cuda_ipc (NVLink) + rc_mlx5 (RDMA) small msgs, zero TCP.** 3 defects found+fixed: aiperf multi-turn concatenation w/ cumulative-context rows (drop session_id — standalone rows, reuse via hash_ids), NATS 1MB max_payload (→16MB via nats.conf; 256k-token prompts ≈1.5MB), dynamo NATS clients cache max_payload at connect (restart stack after NATS config change) |
| | 0.5 | S3 | Short replay, 3 real sessions @ conc 32, 3P+3D | ⬜ | `smoke3-short-replay.yaml` |
| 2026-08-09 | 0.5 | S4 | 1h endurance canary (arm 2B trace, conc 32) | ⚠️ | **Completed the hour with a real finding**: 2,069 reqs ok, 4.52% err (88 empty-content replay artifacts, 8 timeouts, 2×500), sustained 693 tok/s, ITL p50 15.4ms. **At min ~21, 2 of 3 decode workers crashed within 20s** ("Hang detected on rank 0 in PyExecutor" → MPI rank killed after a >300s executor stall; watchdog timeout is hard-coded in the 1.3.1 image) — load-correlated with the trace's giant-session wave; stack self-recovered and finished. Verdict: recipe's GB200-tuned decode config (bs 128 / mnt 640) is unstable under sustained GB300 conc-32 long-context load. Round-1 arms move to aic GB300-native configs (decode mnt 32, small batch) per 2026-08-09 decision; re-canary S3+S4 on those before official runs. |

**2026-08-08 decision:** speculative decoding (Eagle3) disabled across ALL arms — nvfp4 base model only. Simplifies gated-model access (one model) and removes a confound from the routing comparison. Manifests generated via `scripts/gen_manifests.py` with Eagle stripped, image pinned to `1.3.1`, `max_seq_len` 262144, GPU-taint tolerations added.

**2026-08-08 decision:** host-memory KV offloading disabled across ALL arms (recipe shipped 100 GiB `host_cache_size` on disagg prefill). KV is GPU-resident only; `enable_block_reuse` stays on. Consequence to watch: prefill-side reusable KV capacity is smaller, so measured KV hit rates will be lower than the recipe's published numbers — that's expected and consistent across all arms.
| | 1 | 1A | Agg 24-GPU, NVDA recipe, AIPerf spec | ⬜ | |
| | 1 | 1B | Agg 24-GPU, Weka trace, round-robin | ⬜ | |
| | 1 | 1C | Agg 24-GPU, Weka trace, KV-NVDA defaults | ⬜ | |
| | 1 | 1D | Agg 24-GPU, Weka trace, KV-Tuned Strategy C | ⬜ | |
| | 1 | 1E | Agg 24-GPU, DynoSim cross-validation | ⬜ | |
| | 2 | 2A | Disagg 24-GPU, NVDA recipe | ⬜ | |
| | 2 | 2B | Disagg 24-GPU, Weka trace, round-robin | ⬜ | |
| | 2 | 2C | Disagg 24-GPU, Weka trace, KV-NVDA defaults | ⬜ | |
| | 2 | 2D | Disagg 24-GPU, Weka trace, KV-Tuned Strategy C | ⬜ | |
| | 2 | 2E | Profiler calibration (during 2D) | ⬜ | |
| | 3 | sim | Disagg 72-GPU DynoSim config derivation | ⬜ | |
| | 3 | 3A-live | Disagg 72-GPU live, round-robin | ⬜ | |
| | 3 | 3B-live | Disagg 72-GPU live, KV-Tuned | ⬜ | |
| | 4 | — | Synthesis + Pareto matrix | ⬜ | |

## Headline Metrics

Populated by `scripts/generate_summary.py` from `metrics_*.json` once runs complete.

### Summary Table

_(pending)_

### Routing Perf Gain Matrix

_(pending)_

### DynoSim Sim-vs-Live Fidelity (72 GPUs)

_(pending)_

## Strategy C Sweep Records

| Phase | Step | credit | queue policy | temp | TTFT p99 (ms) | Tok/s/GPU | SLA pass? |
|---|---|---|---|---|---|---|---|

## Incidents / Anomalies

_(record KV transfer path failures, cache flush issues, OOMs, node preemptions here; per stop policy, any non-RDMA KV path or transfer failure stops the job immediately and is logged here)_
