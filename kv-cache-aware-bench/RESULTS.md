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
| | 0.5 | S1 | Weights + 1P1D engine up + manual curl | ⬜ | scale arm 2B to 1/1 |
| | 0.5 | S2 | Mini e2e AIPerf (50 reqs, conc 2) + RDMA path check | ⬜ | `smoke2-mini-e2e.yaml` |
| | 0.5 | S3 | Short replay, 3 real sessions @ conc 32, 3P+3D | ⬜ | `smoke3-short-replay.yaml` |
| | 0.5 | S4 | 1h endurance canary (arm 2B trace) | ⬜ | gate for multi-hour sweeps |

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
