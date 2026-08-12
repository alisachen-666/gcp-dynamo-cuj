# How We Sweep and Select Operating Points — Kimi-K2.5 (trtllm) on GB300

This document details the three-layer toolchain — **Profiler → AIConfigurator (AIC) →
DynoSim → silicon** — used to tune and select benchmark operating points for the
KV-cache-aware routing study, and the decision rules that pick the reported data points.
Companion artifacts: `aic-results/` (all solver candidates + sweep CSVs),
`reports/` (interactive Pareto/learnings pages), `RESULTS.md` (findings log),
`INFERENCEX_PIPELINE.md` (reference pipeline analysis: how InferenceX replays the
same corpus, why its scenario contract precludes our replay-fidelity failure, and
which of its mechanisms we adopted or deliberately diverged from).

## The layered toolchain

Each layer answers a different question, consumes the layer below it, and is calibrated
by the layer above it:

```
Profiler   →  measured GB300 kernel/op timings (NVIDIA's silicon profiling)
   ↓ feeds
AIC        →  engine configuration: what should ONE worker look like?
   ↓ feeds (rates)
DynoSim    →  deployment configuration: how many workers, which router, at what load?
   ↓ shortlists
Silicon    →  measured truth on cmcs-a4xmax; telemetry recalibrates DynoSim
```

This mirrors NVIDIA's own published simulate-then-verify methodology for Dynamo
(DynoSim blog): inner loop simulates thousands of configurations; outer loop validates
shortlisted candidates on the real cluster; production telemetry recalibrates the models.

### Layer 0 — Profiler (upstream, mostly implicit)

Three distinct profiling roles, only one of which we operate ourselves:

1. **NVIDIA's kernel profiler built AIC's SILICON database.** Every AIC estimate for
   `gb300/trtllm` is interpolated from measured op-timing tables (e.g. `mla_bmm_perf`
   parquet files) captured on real GB300 silicon. This is why the project mandates
   `--database-mode SILICON`: estimates stay grounded in measured kernels, never
   analytical extrapolation. Coverage gaps (sparse rows above 64k sequence length) are
   reported, not papered over.
2. **Our live runs are the system-level profiler for DynoSim.** The smoke ladder and
   canary runs supplied the calibration constants (see DynoSim lineage below).
3. **TRT-LLM's runtime autotuner** profiles kernels on the actual GPUs at every engine
   start (automatic; not a knob). `nsys`/DCGM-level manual profiling is the escalation
   path if silicon diverges from simulation inexplicably — not needed so far.

### Layer 1 — AIC: engine-level sweep (what should one worker be?)

- **Inputs:** model config (Kimi-K2.5-NVFP4: 61 layers, 384 experts, MLA), system
  `gb300`, backend `trtllm`, workload shape from the trace (ISL 137k / OSL 1.1k),
  24-GPU budget.
- **Brackets solved** (2 cache regimes × 2 architectures): warm (`--prefix 133000`,
  the trace's ~97% reuse level) and cold (no reuse), each for aggregated and
  disaggregated. SILICON mode primary; HYBRID retained as flagged supplementary data.
- **What its internal sweep determined:** parallelism per tier (prefill TP4+attention-DP,
  MoE-EP4; decode TP4 attn-DP, MoE-TP4), worker counts per bracket (P:D 1:5 warm,
  2:4 cold), batching (`max_batch_size`, `max_num_tokens` 137504 prefill / 32–64
  decode), KV fraction 0.8, MoE backend WIDEEP, per-subsystem precision (GEMM nvfp4,
  KV fp8, FMHA bf16-prefill/fp8-decode, comm half), CUDA-graph sizes, transceiver
  buffer sizing. Top-5 candidates per bracket ship as ready-made engine YAMLs.
- **Key negative results** (as valuable as the configs): cold 137k prefill cannot meet
  5 s TTFT on 24 GPUs at any configuration → the workload is servable only via prefix
  caching; TPOT ≤ 10 ms (the recipe's goodput threshold) is infeasible at this ISL
  (floor 12.7 ms) → the recipe's threshold is aspirational.
- **Documented deviations from AIC output:** `enable_chunked_prefill: true` (trace p99
  input 252k exceeds `max_num_tokens`), `max_seq_len: 262144`.
- **What AIC cannot answer:** anything involving the router — it models a worker, not a
  fleet. That's DynoSim's layer.

### Layer 2 — DynoSim: deployment-level sweep (fleet, router, load)

Custom trace-driven discrete-event simulator (`scripts/dynosim_pd.py`; NVIDIA's DynoSim
is not yet public). Fidelity anchors:

- **Router:** the *exact* scoring formula from dynamo v1.3.1 source
  (`lib/kv-router/src/scheduling/selector.rs::worker_logit`), including tuned-flag
  semantics (`prefill_load_scale`, `overlap_score_credit`).
- **Caches:** per-prefill-worker radix prefix caches over the trace's real `hash_ids`
  (64-token blocks), LRU-evicted at measured KV capacity — so reuse rates are emergent,
  not assumed.
- **Rates:** seeded from AIC SILICON solves, then recalibrated from silicon (below).

**Sweep axes:** P:D {1:5, 2:4, 3:3} × closed-loop concurrency {16…384} × policy
{round-robin, least-loaded, KV-defaults, KV-tuned × credit {0.7, 0.85, 1.0}} — 144
cells, minutes per full sweep on the VM, run concurrently with all cluster work.

**Calibration lineage (v1→v5) — every iteration fixed a real modeling error:**

| Version | Fix | Source of truth used |
|---|---|---|
| v1→v2 | prefill rate was warm-seq-rate (30× too cheap for misses) → 65k uncached tok/s/worker | AIC cold solve |
| v2→v3 | decode TPOT slope 0.75→0.1 ms/seq (model predicted 22.5 ms where silicon measured 15.4) | live S4 canary ITL |
| v3→v4 | trace was session-GROUPED → caches never pressured, policies indistinguishable → 393-way session interleave (also fixed the live replay trace) | first-principles + trace analysis |
| v4→v5 | SLA gate removed from sweep per decision; concurrency extended to 384 to find rollover | user decision |

**Known limits:** decode model calibrated to one silicon point; no engine-scheduler
detail (NVIDIA notes scheduler-aware replay matters most for high-concurrency TTFT);
mixing width fixed at 393 (v6: width = concurrency); credit-*decay* (Strategy C step 2)
not yet modeled.

### Layer 3 — Silicon: the outer loop

The smoke ladder (S0–S4) validated the stack and *supplied calibration data*; the
round-1 chain (arms 2B/2C/2D at 3:3, concurrency 32/48/64/96/128/192/256, 1800 s per
point, cold-start per arm, identical everything except the router flag) measures the
swept curves. Its telemetry (ITL vs load, TTFT percentiles, `cached_tokens` reuse
rates) feeds the next DynoSim recalibration.

## Selection rules — and how they evolved

1. **Initial:** recipe-inherited SLA gate (TTFT p95 ≤ 5 s from the recipe's goodput
   threshold; TPOT relaxed 10→20 ms after AIC proved 10 ms infeasible); select max
   KV throughput × capped KV/RR ratio among gate-passing cells.
2. **Revision 1 (no-SLA):** gate removed — sweep everything, hunt max throughput,
   always *report* latency next to throughput but never filter by it. Finding: the
   ungated global throughput ceiling (3:3 / conc 128 / KV-tuned, 2,978 tok/s @ 0.60 s)
   coincides with the gated choice — the KV operating point isn't a latency compromise.
3. **Revision 2 (current headline rule): best performance with BOTH policies pre-knee.**
   The reported comparison cell is the highest load at which round-robin's admission
   queue still drains (bounded, stable TTFT) — eliminating any "you overloaded the
   baseline" critique. Sim prediction: **3:3 at concurrency 48** — KV ≈ 2.2–2.5k tok/s
   @ 0.4–0.8 s TTFT p95 vs RR ≈ 2.1k @ 4.65 s (~6–11× tail gap, both healthy).

**The knee framing** that motivates rule 3: below the prefill-capacity knee, KV-routing
gain = recompute savings (bounded, ~1.5–4× TTFT — consistent with Baseten's published
~2× at 89% hit/50k ISL and NVIDIA's DynoSim reuse numbers); past the knee, the
cache-blind baseline's queue grows without bound (measured in sim: RR TTFT p95 9→117 s
across conc 64→384 while KV stays 0.4–3.5 s). The headline uses the pre-knee cell; the
knee curve itself is reported as the capacity story (KV moves the serviceable-load knee
~48 → ~256 sessions, ~5×); the ceiling point bounds max throughput.

## Reporting discipline

Every throughput number travels with TTFT p50/p95/p99, TPOT, reuse rate, and goodput%
against the recipe thresholds. Six standard curves (see `reports/`): TTFT-vs-concurrency
knee plot; reuse-rate-vs-concurrency; tok/s/GPU-vs-tok/s/user frontier; theory-vs-silicon
overlay; P:D frontier family; TPOT-vs-concurrency. Units are always explicit (system
tok/s vs per-GPU vs per-user) — the recipe's own results table mislabels system
throughput as tok/s/GPU, and we do not repeat that.
