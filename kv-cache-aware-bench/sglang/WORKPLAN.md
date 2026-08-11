# Workplan: tuning Kimi-K2.5 with SGLang on GB300 (AIC + Profiler + DynoSim)

The trtllm pipeline (../SWEEP_METHODOLOGY.md), re-instantiated for sglang. Every
command used for perf tuning is listed verbatim per phase. Policies carried over:
SILICON database mode only; sims run parallel to live work; every deviation from
generated configs documented; all jobs `alisachen-` prefixed; latency always
reported next to throughput.

## Phase S0 — AIC feasibility + engine-level tuning  ✅ started 2026-08-11

Support check (DONE — PASS agg+disagg, sglang perf DB 0.5.12):
```bash
kubectl run aiconf --image=python:3.12-slim --restart=Never \
  --overrides='{"spec":{"nodeSelector":{"cloud.google.com/gke-nodepool":"system"}}}' \
  --command -- sleep 14400
kubectl exec aiconf -- sh -c "pip install -q aiconfigurator==0.10.0 && \
  aiconfigurator cli support --model nvidia/Kimi-K2.5-NVFP4 --system gb300 --backend sglang"
```

SILICON solves — warm (trace reuse level) and cold brackets (RUNNING):
```bash
# warm: prefix 133k of ISL 137k (trace ~97% reuse), recipe TTFT threshold
aiconfigurator cli default --model-path nvidia/Kimi-K2.5-NVFP4 --system gb300 \
  --backend sglang --total-gpus 24 --isl 137000 --osl 1100 --prefix 133000 \
  --ttft 5000 --tpot 10 --max-seq-len 262144 --enable-chunked-prefill \
  --database-mode SILICON --save-dir /tmp/aic-sgl-warm
# cold: no reuse; TTFT relaxed to 30s (trtllm cold solve proved 5s infeasible)
aiconfigurator cli default --model-path nvidia/Kimi-K2.5-NVFP4 --system gb300 \
  --backend sglang --total-gpus 24 --isl 137000 --osl 1100 \
  --ttft 30000 --tpot 10 --max-seq-len 262144 --enable-chunked-prefill \
  --database-mode SILICON --save-dir /tmp/aic-sgl-cold
```

Artifact extraction (top-5 engine configs per bracket + pareto CSVs):
```bash
kubectl exec aiconf -- tar czf /tmp/aic-sgl.tgz -C /tmp aic-sgl-warm aic-sgl-cold
kubectl cp default/aiconf:/tmp/aic-sgl.tgz sglang/results/aic-sgl.tgz
tar xzf sglang/results/aic-sgl.tgz -C sglang/results/
# archive frontier CSVs for the curves:
cp sglang/results/aic-sgl-warm/*/{agg,disagg}/pareto.csv sglang/results/
```

**Outputs consumed downstream:** P:D split per bracket, worker parallelism
(tp/dp/ep/moe), batching caps, memory fractions, per-subsystem precision,
attention/MoE kernel backend choices — plus predicted (throughput, TTFT, TPOT)
per candidate. Expect version-skew triage like trtllm's WIDEEP: validate each
generated engine option against the pinned sglang image before trusting it.

## Phase S1 — Profiler layer (three roles, same as trtllm)

1. **AIC SILICON DB = NVIDIA's kernel profiling** — no command; verify coverage
   and flag gaps (>64k seqlen rows are sparse; check the solve logs):
```bash
grep -E "interpolation|low-fidelity" sglang-solve.log | sort | uniq -c
```
2. **Live system-level profiling** — smoke/canary telemetry that calibrates
   DynoSim (commands in Phase S4); extraction:
```bash
# measured ITL/TTFT percentiles per point from aiperf artifacts:
gcloud storage ls gs://alisachen-models/perf/ | grep sgl
gcloud storage cat gs://alisachen-models/perf/<run>/<point>/profile_export_aiperf.json
# engine-side: worker iteration timing + kv metrics
kubectl logs <sgl-worker> -c worker | grep -E "iteration|batch|kv"
```
3. **Escalation profilers** (only if silicon diverges inexplicably from sim):
```bash
kubectl exec <worker> -- nsys profile -d 30 -o /tmp/prof python3 ... # kernel timeline
kubectl exec <worker> -- dcgmi dmon -e 1002,1003,1005 -c 30           # SM/mem util
```

## Phase S2 — DynoSim deployment-level sweep (backend-agnostic router)

The simulator and router formula are unchanged (dynamo's worker_logit doesn't
know the engine). Re-seed the rate constants from the sglang AIC solves, then
sweep. Constants to update in `scripts/dynosim_pd.py` (sglang values from S0):
`PREFILL_TOKRATE` (cold-solve uncached rate), `TPOT_BASE_MS`/`TPOT_SLOPE_MS`
(pareto TPOT-vs-batch fit), `KV_CAPACITY_TOKENS` (sglang memory fraction),
plus `AGG_*` equivalents.

```bash
# disagg P:D x conc x policy sweep (144+ cells, minutes on the VM):
python3 scripts/dynosim_pd.py <trace>/weka_256k_aiperf_interleaved.jsonl \
  --sweep --requests 3000 --out sglang/results/dynosim_sgl_disagg_v1.csv
# aggregated sweep:
python3 scripts/dynosim_pd.py <trace>/weka_256k_aiperf_interleaved.jsonl \
  --agg --requests 3000 --out sglang/results/dynosim_sgl_agg_v1.csv
```

Selection rules (inherited): no SLA gate on the sweep; headline = best
performance with BOTH policies pre-knee; knee located via TTFT-vs-conc slope
and within-run drain test.

## Phase S3 — Manifests + stack bring-up

Generate operator-less arms (mirror of trtllm generators; to write:
`sglang/scripts/gen_sglang_arms.py`). Key differences from trtllm workers:
`python3 -m dynamo.sglang` entrypoint, sglang server-args from AIC config,
NIXL for disagg KV transfer (DynamoBench-proven env: UCX rail pins we already
carry + NIXL side channel ports), image pinned after Kimi-2.5 support check:

```bash
# image pinned (validated 2026-08-10): lmsysorg/sglang:v0.5.17-cu130-runtime
#   probe PASSED: KimiK25ForConditionalGeneration in model registry
#   (srt/models/kimi_k25.py, + kimi_k25_eagle3.py), quant methods include
#   modelopt / modelopt_fp4 / nvfp4_online. CUDA 13.0 for GB300 sm_103.
#   NOTE engine 0.5.17 vs AIC config format 0.5.11 / perf DB 0.5.12 —
#   diff generated server-args against this image before emitting manifests:
kubectl exec alisachen-sgl-probe -- python3 -m sglang.launch_server --help
python3 sglang/scripts/gen_sglang_arms.py     # emits sgl-disagg-{rr,kv,kvt}.yaml, sgl-agg1n-*.yaml
kubectl apply -n dynamo-cloud -f sglang/manifests/sgl-disagg-rr.yaml
```

## Phase S4 — Smoke ladder (reuse S0–S4 jobs against sglang stacks)

```bash
kubectl apply -n dynamo-cloud -f manifests/perf/smoke0-infra.yaml         # infra (image swap)
# S1: 1P+1D scale-down + manual completion via frontend pod curl
kubectl exec <frontend> -- curl -s http://localhost:8000/v1/models
kubectl exec <frontend> -- curl -s -X POST http://localhost:8000/v1/chat/completions -d '...'
# S2/S3: sliced-trace aiperf smokes + transport proof:
kubectl apply -n dynamo-cloud -f manifests/perf/smoke2-mini-e2e.yaml      # endpoint swap
kubectl set env deployment/<workers> UCX_PROTO_INFO=y   # per-connection transport proof
kubectl logs <prefill> -c worker | grep -E "rndv|cuda_ipc|rc_mlx|tcp/"    # NVLink/RDMA only
bash ~/DynamoBench/common/kv-transport-guard.sh                            # stop policy
# S4: 1h endurance = first bench point held for an hour, restart-count watch
```

## Phase S5 — Benchmark arms + calibration loop

```bash
# per arm: deploy -> cold restart -> bench (sliced trace, conc sweep) -> teardown
python3 scripts/gen_perf_jobs.py     # ARMS += sgl arms; alisachen- prefix, 8h deadline
kubectl apply -n dynamo-cloud -f manifests/perf/sgl-disagg-rr-bench.yaml
# calibration loop after first live points (same as trtllm v1->v5):
#   refit PREFILL_TOKRATE / TPOT slope from measured -> re-run S2 sweeps -> re-select
```

Curves regenerate with sglang traces added:
```bash
python3 scripts/gen_learnings_html.py reports/learnings.html   # + sglang series
python3 scripts/gen_static_svgs.py                             # GitHub SVGs
```

## Milestones / exit criteria

1. AIC solves complete, configs triaged against pinned image (S0)  — in flight
2. Sim sweep with sglang constants; highlight-point prediction (S2)
3. Smoke ladder green incl. transport proof (S4)
4. RR vs KV-defaults vs KV-tuned measured, disagg + agg (S5)
5. Cross-backend comparison: sglang vs trtllm at matched arms — the study's
   backend-generality claim for KV-aware routing
