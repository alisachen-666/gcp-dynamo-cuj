---
name: pareto-benchmarking
description: >-
  Build a theoretical + silicon-validated Pareto curve for disaggregated LLM
  endpoint benchmarking (System TPS vs TPS/User and tps/GPU). Use when someone
  wants to run the endpoints pareto study: isolate ctx-only and gen-only curves,
  rate-match them into theoretical full-system configs, validate selected points
  on full-system silicon (x72), and visualize theory vs silicon. Covers the
  nv-sflow disaggregated harness, the run-folder contract, retained ctx/gen
  tables, the Phase-3 rate-match pseudocode, and the Pareto visualizer.
---

# Pareto Benchmarking Guide

This guide describes the experiment flow for building a theoretical and silicon-validated Pareto curve for disaggregated LLM endpoint benchmarking.

Record all retained data and rate-match results in your own summary workbook/spreadsheet — the per-phase deliverable tables below define the exact columns to track.

Final deliverables:

- summary workbook
- Pareto visualizer
- retained ctx/gen metric tables
- full-system silicon run index

## Definitions

- `N`: per-server concurrency.
- `N_ctx`: concurrency per ctx server.
- `N_gen`: concurrency per gen server.
- `T_ctx`: context server type.
- `T_gen`: generation server type.
- `n_ctx`: number of ctx servers.
- `n_gen`: number of gen servers.
- `gpus_per_ctx[T_ctx]`: GPU count consumed by one context server of type `T_ctx`.
- `gpus_per_gen[T_gen]`: GPU count consumed by one generation server of type `T_gen`.
- `R_ctx(T_ctx, N_ctx)`: retained req/s for one context server of type `T_ctx` at concurrency `N_ctx`.
- `R_gen(T_gen, N_gen)`: retained req/s for one generation server of type `T_gen` at concurrency `N_gen`. For generation, this may be capped by a TPOT/OSL-derived bound when the reported request rate is inconsistent with measured decode speed.
- `OSL`: stable output sequence length used for rate-match math. Set this from the workload or dataset.
- `TOTAL_GPUS`: full-system GPU budget for the silicon target.
- `TTFT_FILTER_MS`: max TTFT p95 allowed for theoretical source points.
- `N_ctx_op`: the concurrency one ctx server actually faces in the full-system deployment under a closed-loop concurrency-`N_sys` load = `ceil(N_sys / n_ctx)`. This is the admission backlog each ctx server must clear before a request gets its first token — **not** a freely chosen low `N_ctx`. ctx TTFT is a queueing latency and MUST be evaluated at `N_ctx_op`.
- `R_ctx_sat(T_ctx)`: saturated per-server ctx request rate — the plateau of `R_ctx(T_ctx, N_ctx)` over `N_ctx`. ctx prefill throughput saturates above a small `N_ctx`, so ctx CAPACITY is concurrency-independent and may be read from any saturated point. ctx LATENCY (TTFT) is **not** — it keeps rising with concurrency. Keep these two ctx lookups separate.

Use lowercase `n_*` for server counts and uppercase `N_*` for per-server concurrency.

Use `R_*` for retained request-rate functions. For context, retained request rate is normally measured req/s. For generation, retained request rate can be the measured req/s or a TPOT/OSL-capped value if measured request throughput is inconsistent with decode latency.

## Phase 0: Disaggregated Experiment Harness

Goal: use one nv-sflow launch to start disaggregated TRT-LLM servers, frontends, and the MLPerf endpoints client together.

Reference harness: the nv-sflow disaggregated endpoints harness in this repository.

Start from an existing per-point reproduction config in the harness, then change only the topology, server configs, client load settings, sample count, duration, and output location for each experiment. Example: any `NVIDIA/src/configs/<system_desc_id>/<benchmark_model>/point_<C>/` directory — these hold the exact configs that reproduced a submitted point (1:1 with `NVIDIA/pareto/<system>/<model>/results/point_<C>/`).

A per-point config directory contains (model-dependent):

- endpoint client config — `client.yaml`
- ctx/gen TRT-LLM server configs + env — `server-ctx.yaml`, `server-gen.yaml`, `server-env.yaml`
- (DeepSeek-R1 also) `dsr1_config.yaml`, `slurm_env.yaml`, `endpoint_env.yaml`, `source.yaml` (run lineage)
- disaggregated launch template (shared): `NVIDIA/src/sflow/templates/trtllm_disagg_endpoints.yaml`

> Note: reproduction configs live under `NVIDIA/src/configs/...` (decoupled 1:1 from `NVIDIA/pareto/.../results/`); the sflow harness lives under `NVIDIA/src/sflow/`.

### Action Steps

1. Start from a per-point config under `NVIDIA/src/configs/<system_desc_id>/<benchmark_model>/point_<C>/`.
2. Update the config values for the experiment: ctx/gen server counts, GPUs per server, frontend count, TRT-LLM YAML paths, endpoint client settings, and output folder.
3. Run the full experiment with one nv-sflow command; the template launches ctx, gen, frontend, and client tasks in order.
4. Make sure the output folder contains the full submitted/resolved YAML configs, full server logs with per-iteration stats, full frontend logs, full client logs, client reports, and summary metrics.
5. If any required artifact is missing, update the sflow template to copy or tee it into the run output folder before using that template for the study.
6. Record the run in the experiment index.

Example interactive command:

```bash
sflow run \
  -f <example_point_config.yaml> \
  -f <slurm_env.yaml> \
  -f NVIDIA/src/sflow/templates/trtllm_disagg_endpoints.yaml \
  --set ENDPOINT_CONTAINER_IMAGE=<endpoint_client.sqsh> \
  --set WORK_DIR=<repo_or_submission_dir> \
  --tui
```

Example batch command:

```bash
sflow batch \
  -f <example_point_config.yaml> \
  -f <slurm_env.yaml> \
  -f NVIDIA/src/sflow/templates/trtllm_disagg_endpoints.yaml \
  --set ENDPOINT_CONTAINER_IMAGE=<endpoint_client.sqsh> \
  --set WORK_DIR=<repo_or_submission_dir> \
  --nodes=<num_nodes> \
  --partition=<partition> \
  -o <sbatch_script_path> \
  --submit
```

### Deliverable: Run Folder Contract

Every experiment should retain the full submitted/resolved YAML configs, full ctx/gen server logs including per-iteration stats, full frontend logs, full client logs, client reports, metrics files, summary files, and one run-index row that points back to the raw output folder.

## Phase 1: CTX-Only Curves

Goal: measure `R_ctx(T_ctx, N_ctx)` for each context server type.

Starting point: pick a reasonable generation config as the gen sink baseline. A published MLPerf Inference v6.0 NVIDIA Interactive disaggregated config is a good starting reference.

### Action Steps

1. Choose the context server types to evaluate.
2. For each `T_ctx`, run one context server under test.
3. Pair it with enough gen sink capacity so decode is not limiting ctx.
4. Configure negligible generation by setting `max_new_tokens: 1` in `client.yaml`.
5. Set ctx `max_batch_size = N_ctx` and use a high ctx `max_num_tokens` value, for example `65536`, so the token cap is not the intended sweep variable.
6. Piecewise CUDA graphs can help context server performance; consider enabling them if the ctx server config does not already use them.
7. Sweep `N_ctx` starting at `1`, increasing geometrically, until TTFT becomes high. A typical sequence is `1, 2, 4, 8, ...`; stop after the first confirmed point where TTFT p95 exceeds the configured threshold or jumps sharply relative to the previous healthy point.
8. Run long enough to reach steady state.
9. Check ctx server logs for: startup readiness (`Application startup complete` or equivalent), missing workers, `ERROR`/`Traceback`/`Exception`/`RuntimeError` signatures, repeated scheduled requests equal to `max_batch_size`, repeated scheduled tokens equal to `max_num_tokens`, long ctx iteration time, growing waiting/queued request counts, KV-transfer errors, and timeout/assertion messages.
10. Keep every valid `(T_ctx, N_ctx)` point in the ctx-only table. If multiple reruns or cap shmoos exist for the same `(T_ctx, N_ctx)`, mark the retained point used for rate matching as the empirically best valid run and keep the other attempts indexed as supporting data.
11. Stop retaining higher `N_ctx` values once TTFT explodes unless a rerun removes the cause.

### Validity Checks

| Check | How to verify | Pass condition |
|---|---|---|
| Gen sink is not backpressuring ctx | Inspect gen sink logs and client latency split. With `max_new_tokens: 1`, gen TPOT should be negligible and gen-side waiting/queued requests should not grow. If adding gen sink capacity changes ctx req/s or TTFT materially, the original ctx point was sink-limited. | Ctx req/s and TTFT are stable when gen sink capacity is increased; gen logs do not show queue buildup. |
| Startup is healthy | Inspect ctx and gen logs for readiness lines and worker counts. | Every expected worker reaches startup complete; no missing workers. |
| Logs are clean | Search ctx, gen sink, and client logs for `ERROR`, `Traceback`, `Exception`, `RuntimeError`, timeout, assertion, and KV-transfer failure signatures. | No recurring error signature that affects request handling. |
| Completed samples are nonzero | Check client result summary. | `n_samples_completed > 0`. |
| Failed samples are zero | Check client report or result summary. | Failed sample count is `0`. |
| TTFT metrics are present | Check client result summary percentiles. | TTFT p50, p95, and p99 exist. |
| Duration reached steady state | Plot or inspect progress/throughput over time and server iteration logs. | Throughput and iteration timing are stable after warmup; the reported metrics are not from only startup or shutdown transients. |
| Sweep reaches deployment operating concurrency | Compare the max swept `N_ctx` against `N_ctx_op = ceil(N_sys / n_ctx)` for the largest target full-system `N_sys`. | The ctx sweep extends to (and a bit past) the largest `N_ctx_op` any target config will impose, so Phase 3 can read ctx TTFT at the operating concurrency instead of failing the existence check or under-predicting from a lower point. |

### Deliverable: CTX-Only Retained Data

Spreadsheet table: `CTX-only retained data`

| Column | How to fill |
|---|---|
| `role` | Literal `ctx`. |
| `context server type` | Context server type `T_ctx`. |
| `N per ctx server` | Retained row concurrency. |
| `run id` | Retained run id. |
| `ctx workers` | Number of ctx workers in the run. |
| `gen feeder type` | Generation server type used as the sink. |
| `gen feeder workers` | Number of gen sink workers in the run. |
| `ctx GPUs/server` | GPU count for one ctx server of this type. |
| `ctx BS` | Ctx `max_batch_size`. |
| `ctx MNT` | Ctx `max_num_tokens`. |
| `req/s` | `n_samples_completed / duration_s`; this is `R_ctx(T_ctx, N_ctx)`. |
| `TTFT p50 ms` | Client result summary TTFT p50. |
| `TTFT p95 ms` | Client result summary TTFT p95. |
| `TTFT p99 ms` | Client result summary TTFT p99. |
| `duration s` | Benchmark duration in seconds. |
| `samples completed` | Completed sample count. |
| `status` | Run status. Retained rows should be `completed`. |

## Phase 2: GEN-Only Curves

Goal: measure `R_gen(T_gen, N_gen)` and decode speed for each generation server type.

Starting point: pick a reasonable high-concurrency context config from the CTX-only study as the ctx feeder baseline. Use many context servers so gen is the system under test, not the ctx feed path; a recommended starting point is to fill two nodes completely with context servers.

### Action Steps

1. Choose the generation server types `T_gen` and the context feeder type from a high-concurrency CTX-only point.
2. For each `T_gen`, run one generation server under test with overprovisioned ctx feeders; start with enough ctx servers to fill two nodes completely.
3. For each sweep point, set gen `max_batch_size = N_gen` and gen `max_num_tokens = N_gen`.
4. Sweep `N_gen` starting at `1`, increasing geometrically. A typical sequence is `1, 2, 4, 8, ...`; run each point long enough to reach steady state and continue until TTFT explodes due to gen-side queueing delays.
5. Inspect gen logs for startup failures, runtime errors, generation iteration time, queued/waiting requests, repeated scheduled requests or tokens at configured caps, KV-transfer errors, and timeout/retry messages.
6. Inspect ctx feeder and frontend/router logs to rule out feed or routing bottlenecks when TTFT rises.
7. Keep every valid `(T_gen, N_gen)` point in the gen-only table. If multiple reruns or cap shmoos exist for the same pair, mark the retained point used for rate matching as the empirically best valid run and keep other attempts indexed as supporting data.

### Validity Checks

| Check | How to verify | Pass condition |
|---|---|---|
| Gen startup healthy | Inspect gen server logs for readiness and worker count. | All expected workers are ready; no startup traceback or crash loop. |
| Ctx feeders not bottlenecking | Inspect ctx feeder logs and compare reruns with more ctx/feed capacity when needed. | Gen req/s and TPOT do not materially improve when ctx capacity is increased. |
| Iteration time acceptable | Inspect per-iteration generation time. | Iteration time is stable at healthy points and explains TPOT when TPOT rises. |
| TPOT measured correctly | Read client metrics from completed streaming responses. | `tpot_p50_ms` and `tpot_p95_ms` are present and based on completed samples. |
| TTFT interpreted correctly | Compare TTFT with TPOT and feeder/router logs. | High TTFT with stable TPOT is not treated as gen saturation until feed/router bottlenecks are ruled out. |
| Samples completed | Read client report. | Completed samples are nonzero. |
| Failed samples | Read client report. | Failed samples are zero, or failures are explicitly classified and excluded from retained data. |
| Steady state | Review runtime window and time-series metrics. | Measurement window is long enough that req/s, TTFT, and TPOT are stable. |

### Deliverable: GEN-Only Retained Data

Spreadsheet table: `GEN-only retained data`

| Column | How to fill |
|---|---|
| `role` | Literal `gen`. |
| `generation server type` | Generation server type `T_gen`. |
| `N per gen server` | Retained row concurrency. |
| `run id` | Retained run id. |
| `context feeder type` | Context server type used by the feeders. |
| `ctx feeder workers` | Number of ctx feeder workers. |
| `gen GPUs/server` | GPU count for one gen server of this type. |
| `gen BS` | Gen `max_batch_size`. |
| `gen MNT` | Gen `max_num_tokens`. |
| `req/s` | `n_samples_completed / duration_s`; reported generation req/s. |
| `tok/s/user` | `1000 / TPOT_p50_ms`. |
| `TPOT p50 ms` | Client result summary TPOT p50. |
| `TPOT p95 ms` | Client result summary TPOT p95. |
| `TTFT p50 ms` | Client result summary TTFT p50. |
| `TTFT p95 ms` | Client result summary TTFT p95. |
| `TTFT p99 ms` | Client result summary TTFT p99. |
| `duration s` | Benchmark duration in seconds. |
| `samples completed` | Completed sample count. |
| `status` | Run status. Retained rows should be `completed`. |

## Phase 3: Theoretical Rate Matching

Goal: combine isolated ctx and gen curves into full-system theoretical configurations.

### Action Steps

1. Load retained ctx/gen rows.
2. Filter to completed rows with `TTFT p95 <= TTFT_FILTER_MS`.
3. Enumerate feasible full-system GPU splits for each `(T_gen, N_gen)` and each context server type `T_ctx`.
4. Select a context server type and ctx concurrency that can support the gen request rate.
5. Compute theoretical req/s, TPS/GPU, tok/s/user, TTFT, and TPOT using retained request-rate functions `R_ctx` and `R_gen`.
6. Keep Pareto-optimal points on `tok/s/user` versus `tps/GPU`.

### Exact Pseudocode

```text
Inputs:
  ctx_points: retained CTX-only rows
  gen_points: retained GEN-only rows
  TOTAL_GPUS: target full-system GPU budget
  OSL: stable output sequence length for the workload
  TTFT_FILTER_MS: max allowed p95 TTFT for source points

Derived functions from retained rows:
  R_ctx(T_ctx, N_ctx) -> ctx req/s for one ctx server
  R_gen(T_gen, N_gen) -> gen req/s for one gen server
  TPS_user(T_gen, N_gen) -> 1000 / TPOT_p50_ms
  TTFT_ctx(T_ctx, N_ctx) -> ctx p95 TTFT
  TTFT_gen(T_gen, N_gen) -> gen p95 TTFT

all_candidates = []

for N_sys in exact_system_concurrencies:
  for T_gen in measured generation server types:
    for n_gen in feasible generation server counts:
      if N_sys % n_gen != 0:
        continue

      N_gen = N_sys / n_gen
      if R_gen(T_gen, N_gen) is unavailable:
        continue
      if TTFT_gen(T_gen, N_gen) > TTFT_FILTER_MS:
        continue

      gen_gpus = n_gen * gpus_per_gen[T_gen]
      remaining_gpus = TOTAL_GPUS - gen_gpus
      if remaining_gpus <= 0:
        continue

      R_system = n_gen * R_gen(T_gen, N_gen)

      best_ctx = None
      for T_ctx in measured context server types:
        n_ctx = floor(remaining_gpus / gpus_per_ctx[T_ctx])
        if n_ctx <= 0:
          continue

        # CAPACITY check (throughput): ctx prefill req/s saturates above a small
        # N_ctx, so use the saturated per-server rate. Concurrency-independent.
        ctx_capacity = n_ctx * R_ctx_sat(T_ctx)
        if ctx_capacity < R_system:
          continue

        # LATENCY check (TTFT): under a closed-loop concurrency-N_sys load each ctx
        # server clears an admission backlog of ~N_sys/n_ctx requests, so TTFT must
        # be read at the OPERATING concurrency, not at the capacity point.
        N_ctx_op = ceil(N_sys / n_ctx)

        # EXISTENCE first (the point may not have been measured at all), THEN gate.
        # If no ctx run reached N_ctx_op, the config's ctx latency is uncharacterized
        # -> it CANNOT be certified. Reject it (or emit flagged ttft_unmeasured);
        # never substitute a lower-N_ctx TTFT, which severely under-predicts.
        ttft_pt = nearest_measured_ctx_at_or_above(T_ctx, N_ctx_op)
        if ttft_pt is None:
          continue
        ctx_ttft95 = TTFT_ctx(T_ctx, ttft_pt.N_ctx)
        if ctx_ttft95 is None or ctx_ttft95 > TTFT_FILTER_MS:
          continue

        candidate_ctx = {
          T_ctx,
          n_ctx,
          N_ctx_op,
          ctx_capacity,
          ctx_capacity_ratio = ctx_capacity / R_system,
          ctx_ttft95,
        }
        best_ctx = choose_better_ctx(best_ctx, candidate_ctx)

      if best_ctx is None:
        continue

      all_candidates.append({
        N_sys,
        T_gen,
        n_gen,
        N_gen,
        T_ctx = best_ctx.T_ctx,
        n_ctx = best_ctx.n_ctx,
        N_ctx = best_ctx.N_ctx,
        total_gpus = n_gen * gpus_per_gen[T_gen]
                   + best_ctx.n_ctx * gpus_per_ctx[best_ctx.T_ctx],
        req_per_s = R_system,
        tps_user = TPS_user(T_gen, N_gen),
        output_tps_per_gpu = R_system * OSL / TOTAL_GPUS,
        estimated_ttft95 = max(TTFT_gen(T_gen, N_gen), best_ctx.ctx_ttft95),
        ctx_capacity = best_ctx.ctx_capacity,
        ctx_capacity_ratio = best_ctx.ctx_capacity_ratio,
      })

pareto_points = pareto_frontier(
  all_candidates,
  x = "tps_user",
  y = "output_tps_per_gpu",
)
```

Context selection rule:

```text
choose_better_ctx(a, b):
  Prefer:
    1. lower ctx p95 TTFT (measured at N_ctx_op)
    2. then lower N_ctx_op
    3. then lower ctx capacity ratio
```

Notes:

- `exact_system_concurrencies` should include explicit target values and every exact value generated by `N_sys = n_gen * N_gen` for measured gen points.
- Use exact measured lookups only unless interpolation is intentionally added and documented.
- Rate matching CAPACITY is enforced by throughput: `n_ctx * R_ctx_sat(T_ctx) >= n_gen * R_gen(T_gen, N_gen)`. ctx prefill throughput saturates, so capacity may be read from any saturated `N_ctx`.
- **ctx TTFT must be certified at the OPERATING concurrency `N_ctx_op = ceil(N_sys / n_ctx)`, not at the capacity point.** ctx TTFT is a queueing latency: under a closed-loop concurrency-`N_sys` load the ctx tier holds an admission backlog of order `N_sys`, and TTFT ≈ backlog / ctx_capacity. Reading TTFT at a low isolated `N_ctx` (where only `N_ctx` requests ever queue) severely under-predicts the deployed value. **In practice, the actual full-system TTFT p95 can exceed a naive low-`N_ctx` projection by one to two orders of magnitude at high `N_sys`** — e.g. a projection that reads ctx TTFT at `N_ctx=64` while the deployment actually operates at `N_ctx_op≈32768` will badly under-predict.
- **Existence before gate.** Check that a ctx point was MEASURED at `>= N_ctx_op` before comparing its TTFT to the gate. If the operating concurrency is beyond the ctx sweep, the config is uncharacterized and must be rejected (or emitted flagged `ttft_unmeasured`) — never silently substitute a lower-`N_ctx` TTFT. This caps feasible `N_sys` at roughly `(max gate-passing N_ctx) * n_ctx`; serving higher `N_sys` within the TTFT target requires MORE ctx servers (more prefill capacity to keep the backlog shallow) or a faster-prefill ctx topology, which competes with the gen tier for the GPU budget.
- Sweep ctx far enough: characterize each `T_ctx` up to (and a bit past) the largest `N_ctx_op` any target `N_sys` will impose, so the latency check has real data instead of failing existence.
- Keep only configurations that respect `TOTAL_GPUS`.

### Deliverable: Theoretical Full-System Rate-Match Pareto

Spreadsheet table: `Theoretical full-system rate-match Pareto optimal data`

| Column | How to fill |
|---|---|
| `context server type` | Selected context server type `T_ctx`. |
| `generation server type` | Generation server type `T_gen`. |
| `system N` | `gen servers * N per gen server`. |
| `gen servers` | Enumerated `n_gen`. |
| `ctx servers` | Selected `n_ctx`. |
| `gen GPUs` | `gen servers * gen GPUs/server`. |
| `ctx GPUs` | `ctx servers * ctx GPUs/server`. |
| `gen GPUs/server` | GPU count for one gen server. |
| `ctx GPUs/server` | GPU count for one ctx server. |
| `N per gen server` | Retained gen concurrency. |
| `N per ctx server` | Selected retained ctx concurrency. |
| `matched req/s` | `gen servers * R_gen(T_gen, N_gen)`. |
| `tps/GPU` | `matched req/s * OSL / TOTAL_GPUS`. |
| `tok/s/user` | `1000 / gen TPOT_p50_ms`. |
| `TTFT p95 ms` | `max(ctx TTFT p95, gen TTFT p95)`. |
| `TPOT p50 ms` | Gen retained row TPOT p50. |
| `ctx overprovision x` | `ctx capacity req/s / matched req/s`. |
| `gen req/s/server used` | `R_gen(T_gen, N_gen)` used by the algorithm. |
| `ctx req/s/server` | `R_ctx(T_ctx, N_ctx)` from the selected ctx row. |
| `gen BS/MNT` | `gen max_batch_size / gen max_num_tokens`. |
| `ctx BS/MNT` | `ctx max_batch_size / ctx max_num_tokens`. |
| `gen run id` | Retained gen run id. |
| `ctx run id` | Selected ctx run id. |

## Phase 3.5: Visualizing Results

Goal: create an interactive Pareto page that makes theoretical projections and silicon results easy to compare.

### Action Steps

1. Plot Pareto data against `tps/GPU` and `tok/s/user`.
2. Add a line for theoretical rate-matched configurations from Phase 3.
3. Add a line for full-system x72 silicon runs when available.
4. Make every plotted point hoverable with enough metadata to debug or reproduce it: server types, `N_sys`, ctx/gen server counts, per-server `N`, BS/MNT, req/s, `tps/GPU`, `tok/s/user`, TTFT, TPOT, run id, status, and bottleneck note.
5. Include a full-log link for silicon points (to the raw configs, server logs, client logs, and reports) so they are reachable from hover/click details.
6. Keep failed or latency-exceeded silicon points visually distinct from valid Pareto points when they are shown.

> Optional: if you maintain your own performance simulator or a baseline (e.g. aggregated / non-disaggregated serving) curve, you can add those as extra traces for comparison. They are not required to build or validate the Pareto curve.

### Deliverable: Interactive Pareto Visualizer

The visualizer should contain:

| Trace | Meaning |
|---|---|
| Theoretical rate matching | Pareto frontier from isolated ctx/gen curves. |
| x72 silicon runs | Measured full-system results on the target allocation. |

## Phase 4: Silicon Full-System Experiments

Goal: validate selected theoretical rows on the target full-system silicon allocation.

### Action Steps

1. Select theoretical Pareto rows to validate.
2. Instantiate the row topology: context server type, generation server type, ctx servers, gen servers, per-server concurrencies, BS/MNT values, and frontend count.
3. Choose enough frontend servers to avoid request admission, routing, or KV-cache-router bottlenecks. A practical starting point is one frontend per gen server; increase frontend count if TTFT or req/s indicates frontend pressure while ctx/gen iteration times remain healthy.
4. Make sure the client endpoint list includes every frontend URL generated by the launch template.
5. Use the full target GPU budget.
6. Set total samples to an integer multiple of the dataset size.
7. Run long enough for steady-state throughput and latency.
8. Keep full client and server artifacts.
9. Confirm failed samples are zero when report data is available.
10. Compare actual req/s, TPS/GPU, tok/s/user, TTFT, and TPOT against theory.
11. If a run misses prediction, inspect ctx logs, gen logs, frontend pressure, KV-transfer behavior, and caps.
12. If the miss has a clear bottleneck, run a targeted shmoo, for example adjusting the ctx/gen server ratio, and keep the empirically best result.

### Debug Signals

- High TTFT with stable TPOT and only moderate ctx req/s utilization: **ctx admission-queue saturation**. The closed-loop driver holds `N_sys` requests in flight against a fixed prefill rate, so the backlog ≈ `N_sys` and TTFT ≈ `N_sys / ctx_capacity` even though steady-state ctx *throughput* looks healthy. This is the dominant high-`N_sys` failure mode and is exactly what the Phase-3 `N_ctx_op` latency check predicts. Fix: add ctx servers (more prefill capacity) or a faster-prefill ctx topology; do NOT certify from isolated low-`N_ctx` ctx runs.
- High TTFT with stable TPOT (other causes): ctx feed, frontend/router, or KV-cache-router queueing.
- High TPOT: gen decode saturation or gen scheduling issue.
- Low actual req/s with normal TPOT: request admission, frontend fanout, or routing bottleneck.
- Runtime failure before samples issue: startup, placement, filesystem, or worker readiness issue.
- Completed run with zero failed samples but TTFT above the target: usable for achieved frontier analysis, but latency-failing.

### Deliverable: Silicon Validation Status

Spreadsheet table: `Silicon validation status for theoretical rate-match rows`

| Column | How to fill |
|---|---|
| `context server type` | From theoretical row. |
| `generation server type` | From theoretical row. |
| `system N` | From theoretical row. |
| `gen servers` | From theoretical row. |
| `ctx servers` | From theoretical row. |
| `N per gen server` | From theoretical row. |
| `N per ctx server` | From theoretical row. |
| `theory req/s` | Theoretical `matched req/s`. |
| `theory tps/GPU` | Theoretical `tps/GPU`. |
| `theory tok/s/user` | Theoretical `tok/s/user`. |
| `theory TTFT p95 ms` | Theoretical TTFT p95. |
| `theory TPOT p50 ms` | Theoretical TPOT p50. |
| `silicon outcome` | `completed`, `runtime_failed`, `finished_uncollected`, or `not_run`. |
| `TTFT status` | `ttft_pass` if actual TTFT p95 is within target; `ttft_exceeded` if completed above target; otherwise `not_applicable`. |
| `Pareto status` | `pareto_optimal` or `pareto_non_optimal` among completed silicon points. |
| `selected silicon run` | Completed matching run with best achieved `tps/GPU`; otherwise terminal failed/uncollected attempt. |
| `attempt count` | Number of matching full-system attempts. |
| `completed attempts` | Count of matching attempts with completed metrics. |
| `runtime failed attempts` | Count of matching attempts marked runtime failed. |
| `actual req/s` | Selected run `req_per_s`. |
| `actual tps/GPU` | `output_tokens_total / duration_s / TOTAL_GPUS`; fallback `actual req/s * OSL / TOTAL_GPUS`. |
| `actual tok/s/user` | `1000 / actual TPOT_p50_ms`. |
| `actual TTFT p95 ms` | Selected run TTFT p95. |
| `actual TPOT p50 ms` | Selected run TPOT p50. |
| `actual/pred req/s` | `actual req/s / theory req/s`. |
| `frontends` | Selected run frontend count. |
| `gen BS/MNT` | Selected full-system gen `max_batch_size / max_num_tokens`. |
| `ctx BS/MNT` | Selected full-system ctx `max_batch_size / max_num_tokens`. |
| `reason / note` | Short explanation of status, bottleneck, or missing run. |
| `all silicon attempts` | Semicolon-separated matching full-system run ids. |

## Required Artifacts

Keep these for each isolated and full-system run:

- client config
- server configs
- disaggregation topology
- study metadata
- submitted run script copy
- client log
- result summary
- server logs
- run-index entry

Keep these aggregate deliverables current:

- retained ctx/gen metrics table
- theoretical rate-match Pareto table
- full-system silicon run index
- full-system silicon metrics table
- metrics summary
- summary workbook
- Pareto visualizer
