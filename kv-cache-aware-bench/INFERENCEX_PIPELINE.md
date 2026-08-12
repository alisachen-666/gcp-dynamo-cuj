# Reference Pipeline: InferenceX Agentic Benchmark (and How Ours Differs)

InferenceX (SemiAnalysis, github.com/SemiAnalysisAI/InferenceX) is the canonical
consumer of the same corpus we replay (`semianalysisai/cc-traces-weka-062126-256k`
— see `sglang/KV_ROUTING_VERIFICATION.md` for the corpus description). This
document maps their agentic-benchmark pipeline end-to-end, explains why its
replay stage is structurally immune to the fidelity failure we hit, and records
which of their mechanisms we adopted, diverged from (with rationale), or queued.

Source references (permalink commit `a007419c`):
`benchmarks/benchmark_lib.sh` (`resolve_trace_source`, `build_replay_cmd`,
`run_agentic_replay_and_write_outputs`), `benchmarks/multi_node/amd_utils/trace_replay.sh`,
`utils/agentic-benchmark/scripts/{collect_sweep_results,analyze_benchmark_distributions}.py`,
and aiperf v0.12.0 `src/aiperf/common/scenario/inferencex_agentx_mvp.py`.

## 1. Pipeline stages

```
[1] CI matrix (GitHub Actions)          one job per {hardware, engine, TP, conc, offload}
[2] Server bring-up (Slurm shells)      sglang/vllm/dynamo, agg or disagg; worker URLs recorded
[3] Trace replay per concurrency        trace_replay.sh + benchmark_lib.sh:
      a. clear_kv_caches                flush L1 radix / L2 hicache / L3 store on EVERY
                                        worker (drain-retry) -> each point measured COLD
      b. resolve_trace_source           pick Weka corpus variant by model context length;
                                        pre-download to shared HF cache
      c. build_replay_cmd               aiperf profile --scenario inferencex-agentx-mvp
                                        --public-dataset semianalysis_cc_traces_weka_*
[4] Per-run artifacts                   profile_export.jsonl + aiperf CSV + server metrics
                                        -> agentic_bench_conc<N>.json, benchmark_command.txt
[5] Sweep aggregation                   collect_sweep_results.py -> summary.csv + plots
[6] Workload-fidelity audit             analyze_benchmark_distributions.py -> ISL/OSL
                                        histograms vs published corpus distributions
```

## 2. The replay stage: a scenario as an enforced contract

`--scenario inferencex-agentx-mvp` is not a preset — it is a `ScenarioSpec`
that **validates the invocation and rejects non-conforming runs**:

| Locked rule | Value | Failure mode it makes unexpressible |
|---|---|---|
| `require_loader` | native Weka loaders only | replaying a format-converted file (our exact failure) |
| `require_cache_bust` | `FIRST_TURN_PREFIX` | reuse inflated by the first-turn prefix all traces share |
| `require_ignore_eos`, `require_streaming` | mandatory | OSL infidelity; unmeasurable TTFT/ITL |
| `forbid_ignore_trace_delays` | mandatory | discarding recorded pacing |
| `forbid_input_truncation` | mandatory | silently clipping 200k+ context tails |
| `min_benchmark_duration` | 900 s (`--unsafe-override` ⇒ `submission_valid=false`) | too-short windows passing as canonical |
| `system_idle_gap_cap_seconds` | 10 | recorded think-time stalling the run |
| `timing_mode` | `AGENTIC_REPLAY` | open-loop request spam instead of trajectory semantics |

Key semantics beyond the contract:

- **Concurrency = trajectory lanes.** `--concurrency N` runs N recorded
  sessions with turn ordering and subagent spawn-join preserved.
- **Warmup is intrinsic**: each lane starts mid-trace
  (`--trajectory-start-min/max-ratio 0.25–0.75`, snapshot-primed), then
  `--warmup-requests-per-lane` one-token requests; profiling resumes from the
  resulting live state. No separate warmup phase; no cold first point.
- **Pre-canned assistant replay** (default): recorded assistant responses build
  later turns; live server output is discarded from conversation state — the
  system under test cannot influence the workload.
- **Dynamo session affinity**: conversation-keyed `X-Dynamo-Session-ID`
  (or legacy conv-aware routing flags) with a 3600 s session lease.
- **Safety rails**: `--max-context-length` matched to the server (oversize
  requests become clean skips, not queued 400s); `--num-dataset-entries 393`
  (all traces; default 100 would silently subsample); live error-rate circuit
  breaker plus a post-run validation module that can invalidate results.
- **Observability inside the replay**: `--slice-duration 1.0` (1 s time-series)
  and `--server-metrics <every worker's Prometheus URL>` capture engine-side KV
  usage and prefix-cache hit rate into the artifacts; an opt-in gate fails the
  run if required engine metrics are missing.

## 3. Why their pipeline never had our fidelity failure

Our failure (see `sglang/KV_ROUTING_VERIFICATION.md`): converting the corpus to
aiperf's generic mooncake format put us on a synthesis path whose block content
is cached **per worker process** — the same `hash_id` became different text in
different processes, so the trace's 84.6% cross-request block reuse became ~0%
on the wire, while lengths, timing, and throughput all looked correct.

InferenceX never converted: the native loader (deterministic per-`(trace_id,
hash_id)` block content via `HashIdRandomGenerator`) was co-developed with
their pipeline, and the scenario's `require_loader` makes the converted-file
path *unexpressible*. Their distribution audit (stage 6) checks only ISL/OSL —
which our broken replay preserved perfectly — so the audit is sufficient for
them precisely because the loader guarantees content determinism by
construction. The transferable lesson, now encoded in our verification doc:
**when a dataset ships with a first-party replay path, format conversion is
where fidelity dies — and length distributions are not evidence of reuse
fidelity.**

## 4. Ours vs theirs — deliberate divergences

| Dimension | InferenceX | Ours | Why we diverge |
|---|---|---|---|
| Load model | recorded pacing preserved (`forbid_ignore_trace_delays`), scale via lanes | `--ignore-trace-delays`, sweep `--concurrency` | our study locates capacity knees per routing policy; theirs measures fixed-load realism. Numbers are NOT directly comparable across the two protocols. |
| Steady state | cold caches per point + mid-trace lane starts | 900 s trace-replay cache warmup, then measured ladder | equivalent steady-state intent; theirs is cleaner for point independence, ours matches DynoSim's steady-state window. |
| Router affinity | session-binding header (`X-Dynamo-Session-ID`) | pure KV overlap scoring (`--router-mode kv`) vs round-robin | overlap scoring *is* our study's subject; session binding is a third policy (queued as a possible arm). |
| First-turn prefix | cache-busted (required) | counted in reuse | flagged: quantify its share of our measured 92% reuse before headline claims. |
| Orchestration | Slurm + GitHub Actions CI | GKE Jobs + operator-less Deployments + chain scripts | cluster constraint (no Slurm; RBAC-limited operator). |
| Fidelity audit | ISL/OSL distributions (sufficient given loader guarantees) | 4-layer KV-routing verification (config → events → router telemetry → engine counters/payloads) | reuse is our dependent variable; it needs direct evidence. |

## 5. Adopted / queued from their playbook

Adopted already:
- `--public-dataset` native loader (after our conversion-path failure).
- Dataset pre-staging into a shared HF cache (their `resolve_trace_source`
  pattern; ours additionally stages the processed `datasets` cache because our
  bench pods run `HF_HUB_OFFLINE=1` for the private-model tokenizer).
- Warmup redesign: recipe's unbounded synthetic warmup removed (their design
  avoids a synthetic phase entirely).
- `--tokenizer-trust-remote-code` (Kimi custom tokenizer), `--random-seed` pin.

Queued (priority order):
1. `--server-metrics` + `--slice-duration 1.0` — engine cache-hit time series
   natively in artifacts, replacing log-grepping in `verify_kv_routing.sh` L3.
2. `--max-context-length 262144` — oversize tails currently 400 instead of skip.
3. First-turn cache-bust sensitivity check on our reuse numbers.
4. Optional third arm: session-binding affinity (production-realistic Dynamo mode).
5. Full `--scenario inferencex-agentx-mvp` parity arm so the study has one row
   directly comparable to published InferenceX results.
