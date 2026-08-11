# Verifying KV-aware routing is REALLY active (and how we monitor it)

Motivation: on 2026-08-11 the first silicon sglang comparison point (agg, conc 8)
showed the "KV-aware" arm with TTFT **identical** to round-robin (p50 1.33 s both)
and 433/438 prefill batches at `#cached-token: 0`. Root cause: sglang publishes KV
block events only when `--kv-events-config` is set (`use_kv_events=False` otherwise)
— with an empty router index every worker scores zero overlap and the "kv" policy
silently degenerates to load balancing. The bench runs fine and produces numbers;
nothing errors. **A KV arm must therefore prove it was KV-routed, not assume it.**

## Three verification layers (`scripts/verify_kv_routing.sh <kv-arm> <rr-arm> <container>`)

### Layer 1 — config (necessary, not sufficient)
- Worker logs contain `use_kv_events=True` (dynamo glue derives it from
  `--kv-events-config`); trtllm equivalent: `--publish-events-and-metrics`.
- Frontend args contain `--router-mode kv`.

### Layer 2 — event flow (the failure we hit lives here)
- Workers register an EventPublisher for the **kv_events** topic. Seeing only
  `topic=kv_metrics` means load metrics flow but block events do NOT — the router
  can rank by load, never by overlap. This is the exact silent-degradation mode.

### Layer 2.5 — Dynamo's built-in router telemetry (Prometheus, live)
The frontend `/metrics` endpoint (port 8000) exposes first-class KV-router metrics:

| Metric | Meaning | Healthy KV run |
|---|---|---|
| `dynamo_component_router_kv_hit_rate` | histogram of the router's **predicted** hit rate per request, per worker | mean well above 0; mass in high buckets |
| `dynamo_router_overhead_indexer_find_matches_ms` | indexer lookups (count proves the index is queried) | count grows with requests |
| `dynamo_router_overhead_block_hashing_ms` / `scheduling_ms` | routing pipeline overhead breakdown | µs–ms scale |
| `dynamo_frontend_worker_last_time_to_first_token_seconds` | per-worker TTFT | skewed by affinity |
| `dynamo_frontend_model_total_kv_blocks`, `..._kv_cache_block_size` | worker-reported KV geometry | matches engine config |

Forensic value: during the broken run, `kv_hit_rate` had **all 2,248 requests in
the `le="0"` bucket** — the router predicted zero overlap for every request while
the bench "ran fine". One PromQL-style check (`sum/count > 0.3`) distinguishes
"KV routing active" from "index empty / no reuse", live, mid-run. The verifier
script's Layer 2.5 automates exactly this.

(Note: `dynamo_frontend_cached_tokens` is the frontend's L1 *tokenizer* cache —
not KV-block reuse; don't confuse the two.)

### Layer 3 — behavior (the verdict; measured during/after the bench)
1. **Engine reuse ratio**: aggregate `#cached-token / (#cached + #new)` from worker
   prefill-batch logs. Expectations on the interleaved Weka trace: KV arm ≥ 40–80%
   (sim: 78–84%); RR arm fragments to ~1/N_workers territory (sim: 9–38%).
   The KV−RR gap ≥ 20 points is the primary criterion.
2. **TTFT separation**: from aiperf `profile_export_aiperf.json`, KV TTFT p50 ≤ 0.7×
   RR at the same concurrency (recompute savings must show up; identical
   distributions = routing not active).
3. **Affinity skew**: requests-per-worker coefficient of variation. KV-aware
   routing concentrates sessions (cv high); RR is uniform by construction
   (cv ≈ 0). Guards against "events flow but scores are ignored".

All three run from logs + artifacts already collected — no extra instrumentation.

## Continuous monitoring during runs
- `scripts/monitor_sgl.sh` (background): pod crash/restart growth, bench job
  failures, chain aborts, gcsfuse mount deaths.
- First-point sanity gate: when the first concurrency point of a KV/RR pair lands,
  compare TTFT p50 before letting the ladder continue unattended (this is what
  caught the events bug after one point instead of four ladders).
- KV-transfer stop policy (disagg): NIXL/UCX transport proof — `rc_mlx5` present,
  zero `NIXL_ERR|REMOTE_DISCONNECT|Foreign traffic`, zero TCP data path — before
  any 72-GPU bench is allowed to run (`run_disagg72_chain.sh` gates on it).

## Known-good / known-bad signatures
| Signal | KV really on | KV silently off |
|---|---|---|
| worker log | `use_kv_events=True` | `use_kv_events=False` |
| event topics | `kv_events` + `kv_metrics` | `kv_metrics` only |
| `#cached-token` | mostly > 0, large | ~all 0 |
| TTFT p50 KV vs RR | ≤ 0.7× | ≈ 1.0× |
| req/worker cv | ≫ RR's | ≈ RR's |
