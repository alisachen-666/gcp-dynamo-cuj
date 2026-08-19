# 72-GPU Disaggregated Serving: KV-Aware vs Round-Robin — Campaign Status & Silicon Record

Status as of **2026-08-19 (UTC)**. This is the record of every 72-GPU sglang disagg
run in `gs://alisachen-models/perf/` (all artifacts present there; per-point
`profile_export_aiperf.json` copies mirrored under `results/silicon/disagg72/`,
one-line-per-point table in `results/silicon/disagg72/summary.csv`, regenerate with
`python3 scripts/summarize_d72.py`).

**Bottom line:** the 72-GPU campaign is *complete as scripted* (KV ladder v2,
RR down-sweep, 6-variant router-flag sweep, plus a 9P:8D probe) but it does **not**
yet yield a valid KV-vs-RR verdict at 72 GPUs: both arms sit on the same
~1.1–1.3k tok/s ceiling (15–18 tok/s/GPU — below the *24-GPU agg* result of
2,119 tok/s) with TTFT p50 of 160–370 s at conc ≥96. The stack is bottlenecked
somewhere downstream of routing; the prime suspect is the GDR bypass
(`UCX_IB_GPU_DIRECT_RDMA=n`) applied to all post-rebuild 72-GPU workers (see §5).
No bench job is running; the `sgl-d72-9x9` stack (9P + 8D, 68 GPUs) is still
deployed on np-3 (idle since the last job finished 2026-08-18 ~17:40Z).

## 1. Campaign inventory (what ran, in order)

| When (UTC) | Job(s) | Protocol | Outcome |
|---|---|---|---|
| 08-11 08:43 | `disagg72-{kv,rr}-bench` (1786437783/882) | ladder c96→c192, **mooncake-converted trace** | INVALID — pre-`9c1284f` (0% wire-level reuse) |
| 08-11 16:30 | `disagg72-{rr,kv}-bench` (1786465823/466333) | native loader, 900 s warmup, c96→c384 | pre-rebuild cluster; RR 2.7–3.0k tok/s, KV c96 **4,837 tok/s / TTFT p50 3.2 s**, KV c192 collapsed (91.9 s p50); KV c288/c384 missing |
| 08-13 | cluster rebuilt (`feb1d29`); `disagg72_relaunch.sh` transfer test → **GDR bypass applied** to all disagg72 workers | | |
| 08-17 05:36 | `disagg72-rr-bench` (1786944977) ladder v1 | single frontend, c96→c384, 1800 s/pt | 0.9–1.1k tok/s, TTFT p50 93–304 s |
| 08-17 08:37 | `disagg72-kv-bench` (1786955823) ladder v1 | single frontend, c96→c288 (c384: job Error, 0/384 responses) | 0.8–1.1k tok/s; c288 degraded to 785 tok/s → router-state stall diagnosed |
| 08-17 11:48–15:31 | `d72-flagsweep` ×9 dirs (6 with results) | frontend-only restarts, warm workers, 900 s @ c192 | all variants within ±3% (1,236–1,320 tok/s); defaults first two attempts failed (w/ warmup), re-run last |
| 08-17 18:41–20:19 | `d72-kv-p{192,288,384}` **ladder v2** | fresh frontend per point, 300 s settle, 1800 s | **canonical KV**: 1,232 / 1,227 / 1,254 tok/s |
| 08-17 21:38–08-18 00:08 | `d72-rr-p{12,24,48,72}` **down-sweep** | RR arm swapped onto np-3, fresh frontend/pt, 1800 s | **canonical RR knee search**: 655 / 850 / 880 / 1,080 tok/s |
| 08-18 04:41–17:40 | `d72-9x9-{kv,rr}-c{192,240}` ×6 dirs (2 with results, both KV) | new `sgl-d72-9x9` stack (9P, decode capped at 8 replicas) | KV c192 1,257, c240 1,192 tok/s; the RR 9x9 runs (frontend rev 06:02/07:53) produced no export |

Two leftover `alisachen-sgl-disagg72-kv-bench` pods in `Error` (47 h / 45 h old) are the
ladder-v1 c384 and a retry — "All N inference requests failed" (frontend wedged), i.e.
the router-state stall that motivated ladder v2.

## 2. Canonical results (post-rebuild, fresh-frontend protocol, 1800 s windows)

Stack for both arms: 6 prefill × TP4/EP4 (`--max-running-requests 8`,
`--chunked-prefill-size 16384`) + 12 decode × TP4/EP4 (`--max-running-requests 64`),
sglang v0.5.14 + dynamo 1.3.1, NIXL transfer, `UCX_TLS=cuda_copy,rc_x,tcp`,
**`UCX_IB_GPU_DIRECT_RDMA=n`**, np-3, NATS request plane, KV events over ZMQ.
Router is the only difference: KV = `--router-mode kv --router-temperature 0.0
--router-queue-policy fcfs`; RR = `--router-mode round-robin`.

| arm | conc | tok/s | tok/s/GPU | req/s | TTFT p50 / p90 / p95 / p99 (s) | ITL avg (ms) | ISL avg (completed) |
|---|---|---|---|---|---|---|---|
| RR | 12  | 655   | 9.1  | 0.72 | 13.2 / 41.3 / 51.2 / 79.6 | 10.5 | 81.0k |
| RR | 24  | 850   | 11.8 | 0.82 | 23.6 / 62.0 / 78.9 / 118.5 | 8.2 | 85.3k |
| RR | 48  | 880   | 12.2 | 0.92 | 47.0 / 123.2 / 162.3 / 247.6 | 8.1 | 77.2k |
| RR | 72  | 1,080 | 15.0 | 1.07 | 63.9 / 134.9 / 208.2 / 326.3 | 8.2 | 68.0k |
| RR | 96  | 1,074 | 14.9 | 1.10 | 92.9 / 181.8 / 214.1 / 276.4 | 11.1 | 60.6k |  ← ladder v1
| RR | 192 | 1,113 | 15.5 | 1.39 | 204.9 / 313.2 / 344.6 / 397.7 | 8.0 | 48.5k |  ← ladder v1
| KV | 192 | 1,232 | 17.1 | 1.52 | 202.1 / 263.2 / 271.5 / 282.7 | 8.4 | 50.3k |
| KV | 288 | 1,227 | 17.0 | 1.56 | 295.1 / 352.4 / 360.8 / 371.1 | 8.2 | 48.2k |
| KV | 384 | 1,254 | 17.4 | 1.67 | 374.0 / 451.1 / 465.4 / 485.5 | 8.3 | 45.3k |
| KV 9P:8D | 192 | 1,257 | 17.5 | 1.72 | 163.7 / 226.8 / 239.0 / 270.9 | 12.8 | 42.5k | (900 s)
| KV 9P:8D | 240 | 1,192 | 16.6 | 1.49 | 242.3 / 331.4 / 356.5 / 401.7 | 9.6 | 48.2k |

Router-flag sweep (KV, c192, 900 s, warm workers, frontend-only restarts; variant ↔ run
mapped via the `sgl-disagg72-kv-frontend` ReplicaSet revision history — the
`VARIANT.txt` markers never landed in GCS):

| variant | run | tok/s | TTFT p50 / p95 (s) |
|---|---|---|---|
| defaults (temp 0, fcfs)      | 1786980665 | 1,313 | 165.8 / 229.1 |
| prefill-load-scale 2.0       | 1786970675 | 1,288 | 167.5 / 220.7 |
| prefill-load-scale 3.0       | 1786972787 | 1,236 | 167.0 / 234.9 |
| overlap-score-credit 0.8     | 1786974821 | 1,308 | 168.4 / 229.6 |
| credit 1.0 + decay 0.8       | 1786976717 | 1,320 | 170.3 / 230.2 |
| temperature 0.5              | 1786978690 | 1,269 | 173.3 / 232.1 |

All six are within noise of each other — consistent with the router not being the
binding constraint at this operating point.

## 3. Reading the numbers

1. **A hard ceiling, not a knee.** RR climbs 655 → 1,080 tok/s from conc 12 → 72 and
   then flat-lines (1,074–1,113 at 96–192). KV is flat at 1,23x across 192 → 384 and the
   9P:8D stack lands on the same ~1.2k. Concurrency, router policy, router flags and
   prefill count all fail to move it → the binding constraint is shared capacity
   downstream of routing.
2. **TTFT is pure queueing.** p50 scales ~linearly with concurrency (RR: 13 s @12 →
   64 s @72 → 205 s @192; KV: 202 / 295 / 374 s at 192/288/384). At conc 192 the
   frontend's `queued_requests` gauge averages 307 (max 459) — more than the client's
   concurrency — so cancelled/timed-out requests are leaking into the frontend queue
   across points even with a fresh frontend per point.
3. **KV routing is active but not paying off.** Router telemetry for the canonical
   KV runs: `dynamo_component_router_kv_hit_rate` mean 0.35 (p50 0.00, p75 0.91) at
   6P:12D, mean 0.50 at 9P:8D; 1.4 M `stored`/`removed` KV events in 1800 s
   (steady-state churn ≈ eviction-bound). Half of requests are predicted cold, and
   the hits that do occur don't shorten the queue because prefill isn't what the
   system is waiting on.
4. **Completed-request ISL shrinks with load** (81k → 45k vs trace mean 137k): the
   long-context tail never finishes inside the window at high conc, so the measured
   tok/s is biased toward short requests — another reason these rows cannot be
   compared with the 24-GPU agg ladder directly.
5. **Pre-rebuild contrast.** The same 6P:12D arms on 08-11 (native loader, before the
   rebuild and before the GDR bypass) delivered RR 2.7–3.0k tok/s at c96–c384 and KV
   **4,837 tok/s / 3.2 s p50 TTFT at c96** (KV/RR = 1.8× at c96) before the KV arm
   collapsed at c192. Those runs are not canonical (different cluster incarnation,
   worker config not recorded in the repo) but they show the stack *can* do
   2.5–4× what it does now.

## 4. Verdict on the 72-GPU KV-vs-RR question

**Not answerable from these runs.** The KV ladder (192–384) and the RR down-sweep
(12–72) do not overlap, and where the arms can be compared (c96/c192, ladder v1) they
are equal within noise at ~1.1k tok/s because both are throttled by the same
non-router bottleneck. Recording the campaign as ⚠️ in `RESULTS.md` (3A-live/3B-live),
not ✅.

## 5. Suspected root cause and next steps

- **GDR bypass** (`UCX_IB_GPU_DIRECT_RDMA=n`, set by `disagg72_relaunch.sh` after the
  post-rebuild transfer test failed) forces every KV transfer through host memory.
  At ~4.8 GB of KV per 137k-token request, 1–1.5 req/s ≈ 5–7 GB/s sustained
  host-staged transfer across the stack — plausibly the ceiling we see. This is also a
  **stop-policy violation** (`common/kv-transport-guard.sh` / SWEEP_METHODOLOGY: RDMA-only
  KV path); the per-point "gate" only checks for TCP, not for GDR.
  → Fix the post-rebuild GDR path (mrdma NIC claims / GID / `UCX_IB_ROCE_*` on the
  rebuilt nodes), remove the bypass, re-run S3-style `UCX_PROTO_INFO` transport proof,
  then re-run a *single* overlapping ladder (c48/96/192) for both arms.
- Extend the guard gate to fail on `UCX_IB_GPU_DIRECT_RDMA=n` / host-staged protocols.
- Make the frontend queue leak visible: record `dynamo_frontend_queued_requests` at
  point start; restart *and* wait for queue==0 before launching the next point.
- Re-run the 9x9 RR probe so the topology comparison has both arms.
- Tear down `sgl-d72-9x9` if the GDR fix won't land soon — it is holding 68 GPUs idle.

## 6. Reproduction inventory

| Item | Location |
|---|---|
| Raw artifacts (aiperf raw/jsonl, timeslices, server metrics) | `gs://alisachen-models/perf/<epoch>_alisachen-sgl-{disagg72-*,d72-*}/` |
| Per-point summaries (mirrored) | `sglang/results/silicon/disagg72/<run>/<point>/profile_export_aiperf.json` |
| Table | `sglang/results/silicon/disagg72/summary.csv` (`scripts/summarize_d72.py`) |
| Arm manifests / generator | `sglang/scripts/gen_sglang_arms.py`, `manifests/perf/sgl-d72-flagsweep.yaml` |
| Protocol scripts | `scripts/kv_ladder_v2.sh`, `scripts/rr_downsweep.sh`, `scripts/sweep_router_flags.sh`, `scripts/disagg72_relaunch.sh` |
| Transport guard | `../common/kv-transport-guard.sh` |
| Live stack (2026-08-19) | ns `dynamo-cloud`: `sgl-d72-9x9-{frontend,prefill×9,decode×8}` on np-3; `sgl-disagg72-{kv,rr}-*` scaled to 0 |
