# Our plan vs InferenceX GB300 FP4 dynamo-sglang (8k1k mid_curve)

Sources (fetched 2026-07-28, copies in this dir):
- Recipe: NVIDIA/srt-slurm `recipes/gb300-fp4/8k1k/mid_curve.yaml`
- Runner config: SemiAnalysisAI/InferenceX commit `3282fde` (`.github/configs/nvidia-master.yaml`
  key `dsr1-fp4-gb300-dynamo-sglang`, `runners/launch_gb300-nv.sh`)
- GKE equivalent: AI-Hypercomputer/gpu-recipes `inference/a4x/disaggregated-serving/dynamo/`

## What InferenceX actually runs (8k1k mid_curve)

| Aspect | InferenceX / srt-slurm value |
|---|---|
| Image | `lmsysorg/sglang:v0.5.8.post1-cu130-runtime` (pinned; enroot squash on Slurm) |
| Model | `nvidia/DeepSeek-R1-0528-NVFP4-v2` (**FP4**, served-model-name `deepseek-ai/DeepSeek-R1`) |
| Orchestration | NVIDIA Dynamo **0.8.1** (NATS request plane), disaggregated prefill/decode |
| Topology | **6 prefill nodes** (6 workers, each TP=4 EP=1, dp-attn off) + **12 decode nodes** (1 worker, TP=48 EP=48 DP=48, dp-attn on) = 18 nodes x 4 GPU = **72 GPUs (full NVL72 rack)** |
| Frontends | 10 dynamo frontends (1+9) behind `nginx:1.27.4` |
| Quantization | `modelopt_fp4`; kv-cache `fp8_e4m3`; attention `trtllm_mla` |
| MoE runner | prefill `flashinfer_trtllm`, decode `flashinfer_cutedsl`; decode MoE a2a `deepep` low_latency, `ep-num-redundant-experts=32`, `eplb-algorithm=deepseek` |
| Key server flags | context-length 9600; radix cache disabled; prefill mem-frac 0.95 / decode 0.83; decode cuda-graph-max-bs 512; num-reserved-decode-tokens 112; stream-interval 50 |
| Key env | `NCCL_MNNVL_ENABLE=1`, `NCCL_CUMEM_ENABLE=1`, `MC_FORCE_MNNVL=1`, `SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN=1`, `SGLANG_MOE_NVFP4_DISPATCH=1` (decode), mooncake custom mem pool, huge disagg timeouts |
| Benchmark | **sa-bench** = vendored vLLM-style `benchmark_serving.py` (srt-slurm `src/srtctl/benchmarks/scripts/sa-bench/`): random dataset, ISL 8192 / OSL 1024, `random_range_ratio 0.8`, ignore-eos, concurrencies **512 / 2048 / 4096**, req_rate 700 |
| Spec decode | none (STP only for this config) |
| Arch | aarch64 (same as our Grace nodes) |

## Gaps in our previous plan

1. **Precision/model (biggest perf gap):** we planned FP8 `deepseek-ai/DeepSeek-R1`; the GB300
   reference number is FP4 `nvidia/DeepSeek-R1-0528-NVFP4-v2` (~420 GB). GB300's headline
   throughput comes from NVFP4 + flashinfer/trtllm kernels. -> switch model + `quantization
   modelopt_fp4`, kv-cache fp8_e4m3, attention trtllm_mla.
2. **Image not pinned:** we used `lmsysorg/sglang:latest`; must pin
   `lmsysorg/sglang:v0.5.8.post1-cu130-runtime` for comparability.
3. **Aggregated vs disaggregated:** we planned one aggregated TP=4 server; InferenceX mid_curve is
   Dynamo-disaggregated P/D with wide-EP decode. This is an architectural difference, not a flag.
4. **Scale:** mid_curve needs 72 GPUs (18 nodes). Our cluster has 4 GPU nodes / 16 GPUs total
   (often shared). We cannot reproduce mid_curve as-is. Feasible on our cluster:
   - 1P1D disagg validation (2 nodes) — GKE recipe has exact configs (`deepseekr1-fp8-1p1d-*`)
   - low_latency-flavored config (1P + up to 3D)
   - true mid_curve requires reserving a full NVL72 (18 nodes) — capacity ask.
5. **Benchmark tool:** we planned `sglang.bench_serving`; InferenceX uses the sa-bench fork of
   vLLM `benchmark_serving.py`. Closest match: vendor srt-slurm's sa-bench scripts (they're
   self-contained python) and run with identical args (random dataset, rr 0.8, ignore-eos,
   percentiles ttft/tpot/itl/e2el). The GKE recipe's bench_client does the same thing.
6. **Missing perf env/flags:** MNNVL NCCL vars, mooncake mem pool, NVFP4 dispatch flags,
   radix-cache off, context-length 9600 — all absent from our baseline manifest.

## GKE (A4X Max) translation — how to run the Slurm recipe on our cluster

Per AI-Hypercomputer/gpu-recipes `inference/a4x/disaggregated-serving/dynamo/` (copy in
`gke-a4x-dynamo/`):
- Install Dynamo platform via NGC helm charts (`dynamo-crds`, `dynamo-platform`) — needs
  `NGC_API_KEY`. NOTE: our cluster already has `dynamo-system` ns with a NATS pod (34d old) and
  a `dynamo-cloud` ns — check version before reinstalling (recipe used DYNAMO 0.7.0; InferenceX
  uses 0.8.1).
- Build/push an arm64 Dynamo+SGLang image (`docker build -f container/Dockerfile.sglang
  --platform linux/arm64 --build-arg DYNAMO_VERSION=...`) OR use the pinned lmsysorg image if it
  bundles what dynamo workers need (srt-slurm uses the lmsysorg image directly as the
  `dynamo-sglang` container).
- Model storage: recipe recommends gcsfuse bucket mount (avoids HF rate limits); cluster has
  `gcsfusecsi-*` storage classes.
- Deploy via helm chart `src/helm-charts/a4x/inference-templates/dynamo-deployment` with
  `values_deepep.yaml` (multi-node, uses LWS + ComputeDomain/IMEX via `nvidia-dra-driver-gpu`
  for cross-node NVLink) or `values_wo_deepep.yaml` (1p1d test).
- Our cluster-specific gotcha (not in recipe): pods need BOTH tolerations
  `nvidia.com/gpu:NoSchedule` and `kubernetes.io/arch=arm64:NoSchedule`.
- Benchmark from an in-cluster client pod (`bench_client.yaml`) against the frontend service.

## Suggested next steps

1. Decide scale: 1P1D validation now (2 free-ish nodes needed) vs. wait for full-rack capacity.
2. Pin image `lmsysorg/sglang:v0.5.8.post1-cu130-runtime`; switch model to
   `nvidia/DeepSeek-R1-0528-NVFP4-v2` staged in a GCS bucket (gcsfuse).
3. Port srt-slurm mid_curve server flags/env into the gpu-recipes dynamo-deployment helm values
   (scaled-down P/D counts), keep flag parity where node count allows.
4. Vendor sa-bench scripts into `sweeps/` so results are metric-compatible with InferenceX.

## Addendum (2026-07-29): teammate's GKE values file vs our recipe

User provided the helm values behind the parked `dynamo-dsr1-fp4-8k1k-mid` DGD. Key facts:
- Despite the name it runs **FP8 STP** (`recipes/gb300-fp8/8k1k/stp/mid.yaml`,
  model deepseek-ai/DeepSeek-R1-0528, quantizations: fp8) on NGC image 1.1.1-cuda13 with the
  Dynamo operator — a DIFFERENT benchmark config from our FP4 mid_curve InferenceX target.
- Topology: 5x prefill workers x 2 nodes + 1 decode worker x 8 nodes + 9 frontends; rdma
  claims ENABLED (their cluster has working DRA rdma), GCS model staging (init-container
  download + gcsfuse + 2.5TB SSD), explicit deepepConfig JSON.

ADOPT into our M2/M4 templates (precision-agnostic GKE wisdom):
- UCX_CUDA_IPC_ENABLE_MNNVL=y  (REQUIRED for cross-node cuda_ipc KV transfer over MNNVL)
- UCX_PROTO_INFO=y             (per-op protocol logging - best transport evidence)
- UCX_MEMTYPE_CACHE=n, UCX_MEMTYPE_REG_WHOLE=n
- TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800
- SGLANG_DG_CACHE_DIR on shared storage (DeepGEMM JIT cache; startup win)
- generous probes (startupProbe failureThreshold 1800 x 60s), GCS staging pattern,
  deepepConfig JSON (also in srt-slurm configs/deepep_config.json)
- back-pocket: SGLANG_DECODE_BOOTSTRAP_TIMEOUT=1000, SGLANG_HACK_SEQ_BOOTSTRAP_ROOM=1

DO NOT COPY: NGC 1.1.1 image (breaks InferenceX parity), FP8 recipe/model, rc_x in UCX_TLS
(RDMA dead on REDACTED-GKE-CLUSTER-OLD), SGLANG_ENABLE_SPEC_V2 / ..._OVERLAP_SCHEDULER_SYNC_BATCH (their
newer sglang's knobs; spec-decode diverges from STP-only reference),
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=768 (srt-slurm FP4 mid_curve = 512).

## Addendum (2026-08-01): request-plane parity + max_tpt coverage

- **Request-plane parity break found and being fixed**: srtctl passes `--request-plane nats` to
  every dynamo worker (schema default "nats"; explicit in main's mid_curve since 39d2a68-era
  comment). Our manifests never passed it -> dynamo 0.8.1 defaulted to **TCP** (confirmed in
  archived dsr1-m4-decode-0.log). A/B queued: `manifests/m4n-midcurve-nats.yaml` reruns
  mid_curve conc-2048 with NATS, all else byte-identical.
- **max_tpt vendored + manifest generated** (`reference/srt-slurm-gb300-fp4-8k1k-max_tpt.yaml`
  @ deb1dfd, `manifests/mx-maxtpt.yaml`): verified identical to mid_curve except 10 prefill
  workers / decode DEP32 on 8 nodes / conc 2048. Runs with NATS from the start.
