# Simulation Guide — Every Process, Command, and Result (Showcase Edition)

The complete simulation record of the KV-cache-aware routing study: what we ran,
the exact commands, what came out, and how each result fed the next stage.
Companion docs: `SWEEP_METHODOLOGY.md` (decision rules), `sglang/SELECTION.md`
(selected points), `aic-results/gb200/COMPARISON.md` (external validation).
Visual showcase: the learnings dashboard (trtllm panels 1–8/A/D + sglang S1–S7)
and the standalone sglang pareto page (`sglang/reports/sglang-curves.html`).

Environment note: all AIC runs execute in a cluster pod (`python:3.12-slim` on
the system pool) because this VM's old glibc rejects manylinux wheels:

```bash
kubectl run aiconf --image=python:3.12-slim --restart=Never \
  --overrides='{"spec":{"nodeSelector":{"cloud.google.com/gke-nodepool":"system"},"tolerations":[{"operator":"Exists"}]}}' \
  --command -- sleep 14400
kubectl exec aiconf -- pip install -q aiconfigurator==0.10.0
```

---

## Stage 0 — Support check (is the model/system/backend covered?)

```bash
aiconfigurator cli support --model nvidia/Kimi-K2.5-NVFP4 --system gb300 --backend trtllm
aiconfigurator cli support --model nvidia/Kimi-K2.5-NVFP4 --system gb300 --backend sglang
```
**Result**: PASS for both backends (agg + disagg). Perf DBs: trtllm 1.3.0rc10/rc15,
sglang 0.5.9/0.5.10/0.5.12 — the version line is the API-drift early warning
(it predicted the sglang 0.5.17 glue incompatibility we later hit on silicon).

## Stage 1 — AIC SILICON solves (engine-level tuning; policy: ALWAYS `--database-mode SILICON`)

Workload encoding for the Weka trace: ISL 137k / OSL 1.1k; warm bracket models
the trace's ~97% session reuse via `--prefix 133000`; cold = no reuse
(TTFT relaxed to 30 s — a cold 137k prefill cannot meet 5 s).

```bash
# per backend (trtllm shown; sglang identical with --backend sglang):
aiconfigurator cli default --model-path nvidia/Kimi-K2.5-NVFP4 --system gb300 \
  --backend trtllm --total-gpus 24 --isl 137000 --osl 1100 --prefix 133000 \
  --ttft 5000 --tpot 10 --max-seq-len 262144 --enable-chunked-prefill \
  --database-mode SILICON --save-dir /tmp/aic-warm
# cold: drop --prefix, --ttft 30000
```

**Results** (`aic-results/silicon-*`, `sglang/results/aic-sgl-*`):

| Backend | Warm top | Cold top | Notes |
|---|---|---|---|
| trtllm | disagg P:D 1:5, TP4 workers | 2:4 / 3:3, TP4 | agg top: TP8/MoE-TP2-EP4 |
| sglang | disagg 1:5, TP4/EP4 — TPOT 9.3 ms | 3:3, TP4/EP4 | sglang meets the 10 ms ITL gate trtllm can't (12.7 ms floor) |

Key cross-backend finding: **optimal topology is workload-driven, not
backend-driven** (identical splits) — our manifests/smokes port unchanged.

## Stage 2 — Constant derivation (AIC → DynoSim seeding)

trtllm constants were fitted from AIC + live smoke telemetry (see Stage 5).
sglang constants (sgl-sim v1) via **AIC-ratio transfer** — fit both backends in
AIC space, apply the ratios to the live-calibrated trtllm constants:

- Prefill: AIC cold-TTFT floor per GPU ×4. Validation: trtllm AIC floor
  16.3k/GPU×4 = 65.1k ≈ the live-calibrated 65k → sglang 19.4k×4 = **78k**.
- Decode: min-TPOT-per-batch ratios at operating bs → base 14.3×0.72=**10.3 ms**,
  slope 0.1×0.70=**0.07 ms/seq**.
- Agg: prefill 45k×0.527=**23.7k**; piecewise TPOT with a bs≥8 cliff (from the
  AIC ratio jump 0.70→2.47).

## Stage 3 — DynoSim policy sweeps (our simulator; hash-ID-exact cache model)

DynoSim replays trace metadata directly (hash_ids + lengths — no text; see
`SWEEP_METHODOLOGY.md` for why that makes it exact w.r.t. reuse structure).
Per cell: 11 policies (rr, least-loaded, kv-defaults, scale {1.5,3.0}×credit 0.8,
tuned scale 2.0 × credit {0.5..1.0}), 4,000 requests.

```bash
T=weka_256k_aiperf_interleaved.jsonl
python3 scripts/dynosim_pd.py $T --sweep    --out aic-results/dynosim_sweep_v5.csv          # 24-GPU disagg, 5 splits x 13 conc
python3 scripts/dynosim_pd.py $T --agg      --out aic-results/dynosim_agg_v2.csv            # 6xTP4 agg, full router grid
python3 scripts/dynosim_pd.py $T --disagg72 --out aic-results/dynosim_disagg72_v1.csv       # 18 workers, 5 splits x 8 conc
python3 scripts/dynosim_pd.py $T --sweep    --backend sglang --out sglang/results/dynosim_sgl_disagg24_v1.csv
python3 scripts/dynosim_pd.py $T --disagg72 --backend sglang --out sglang/results/dynosim_sgl_disagg72_v1.csv
python3 scripts/dynosim_pd.py $T --agg      --backend sglang --out sglang/results/dynosim_sgl_agg_v2.csv
```

**Headline results per sweep**:

| Sweep | Headline cell | KV | RR | Story |
|---|---|---|---|---|
| trtllm 24-GPU | 3:3 / conc 48 | 2,220 tok/s, 0.41 s p95 | 2,109, 4.65 s | pre-knee recompute savings |
| trtllm 72-GPU | 6:12 / conc 384 | 5,011, 0.84 s | knee at conc 96 | **KV moves the knee 4×** |
| sglang 24-GPU | 3:3 / conc 96 | 4,347, 4.0 s | knee at conc 32 | 1.9× at 5 s budget |
| sglang 72-GPU | 6:12 / conc 384 | 6,993, 1.47 s | knee at conc 96 | 1.44× thr, 4× conc |
| sglang agg | conc 16 (both peak) | 1,172, 0.86 s | 775, 6.30 s | RR never goodput-feasible |

Router-flag grid outcome: **defaults (+ temperature 0) win at every selected
cell**; tuning pays only post-knee (up to 10–14%) or in prefill-starved splits
(3:15: scale 3.0/credit 0.8 +12%). Wrong tuning costs up to 50% on sglang agg.

## Stage 4 — Point selection (rules in `SWEEP_METHODOLOGY.md` §revisions)

No SLA gate on sweeps; knee = queue-drain (within-run stationarity + cross-conc
slope; 5 s p95 as the operational proxy). Report per scale: **headline** (best
bounded-TTFT KV cell vs RR's own knee) + **conservative** (both pre-knee).
Full tables: `sglang/SELECTION.md`.

## Stage 5 — Calibration loop (simulate → verify → refit)

trtllm v1→v5 corrections from smoke/canary telemetry: cold prefill rate **30×**
(2.1k→65k effective), TPOT slope **7×** (0.75→0.1 ms/seq), trace interleaving
(session-grouped replay hid cache pressure). sglang v1→v2 (pending, data in
hand from silicon): raise agg throughput slope (~1.5–2.7× under-prediction),
drop the bs≥8 cliff below conc 48 (never engages: per-worker bs ≤5.3), add tail
dispersion (KV p95 predicted 0.86 s vs 1.8–2.6 s measured).

**Sim-vs-silicon validation** (the money table, sglang agg):

| conc | sim KV/RR ratio | silicon ratio | sim RR p95 | silicon RR p95 |
|---|---|---|---|---|
| 8 | 1.11× | 1.18× | 6.22 s | 4.65 s |
| 16 | 1.51× | 1.39× | **6.30 s** | **6.18 s** |
| 24 | 1.47× | 1.36× | 6.28 s | 7.86 s |
| 32 | 1.63× | 1.52× | 6.35 s | 9.52 s |

Policy ratios transfer within ~10%; RR SLA-infeasibility confirmed; absolute
throughput under-predicted 1.5–2.7× (calibration v2 items above).

## Stage 6 — External validation showcase (AIC vs NVIDIA's published recipe)

Encoding NVIDIA's dataset manifest (ISL cap 200k, OSL lognormal mean 1,000 —
confirmed from their pinned aiperf manifest) on **gb200/trtllm/24 GPUs**:

```bash
aiconfigurator cli default --model-path nvidia/Kimi-K2.5-NVFP4 --system gb200 \
  --backend trtllm --total-gpus 24 --isl 200000 --osl 1000 --prefix 190000 \
  --ttft 5000 --tpot 10 --max-seq-len 262144 --enable-chunked-prefill \
  --database-mode SILICON --save-dir /tmp/aic-gb200-warm
# cold: drop --prefix, --ttft 30000; ISL-sensitivity: --isl 120000 --prefix 114000
```

**Results and what they proved** (`aic-results/gb200/COMPARISON.md`):
1. No config in the whole space reaches ≤11 ms TPOT at 200k context (floor
   12.6 ms even at conc 1 on 8-GPU workers) → NVIDIA's ~105 tok/s/user implies
   **true mean ISL ~100–140k** ("200k" is a cap, not a mean).
2. Warm SLA-point **total**-token throughput computes to **1,574 tok/s/GPU ≈
   their published "~1,700 tok/s/GPU"** (8%) → their undocumented column is
   total (input+output) accounting; output-based readings are impossible by 49×.
3. Reuse is worth 2× per-user even in solver space (warm 51 vs cold 28 tok/s/user).
4. Lesson encoded: AIC is a *relative-ranking* tool — absolutes run 25–100%
   conservative on decode and must never be compared to published figures
   without calibration.

## Where everything lives

| Artifact | Path |
|---|---|
| Simulator | `scripts/dynosim_pd.py` (constants + both backends + all sweep modes) |
| AIC solve outputs | `aic-results/silicon-*`, `sglang/results/aic-sgl-*`, `aic-results/gb200/` |
| Sweep CSVs | `aic-results/dynosim_*.csv`, `sglang/results/dynosim_sgl_*.csv` |
| Dashboards | learnings dashboard (artifact `d54ddb6a`), `sglang/reports/sglang-curves.html` (artifact `594b2af7`), generators in `scripts/` + `sglang/scripts/` |
| Silicon verification | `sglang/AGG24_RESULTS.md`, `sglang/results/silicon/` |
