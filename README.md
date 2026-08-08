# DeepSeek-R1 Pareto Curve Sweep — GKE A4X Max (GB300)

Workspace for sweeping DeepSeek-R1 inference configurations and building
throughput-vs-latency Pareto curves.

## ACTIVE CLUSTER (switched 2026-07-29): `REDACTED-GKE-CLUSTER-OLD`, project `REDACTED-GCP-PROJECT`, region us-east5

- kubeconfig context: `gke_REDACTED-GCP-PROJECT_us-east5_REDACTED-GKE-CLUSTER-OLD` (current-context on this VM)
- Auth: user ADC (`<user>@google.com`) copied from cloudtop — **expires ~daily**
  (`invalid_rapt` corp reauth); re-copy `~/.config/gcloud/application_default_credentials.json`
  from cloudtop or get durable SA RBAC from the cluster owner (we lack
  `container.clusterRoleBindings.create` there).
- Capacity (at switch time, ALL IDLE — zero GPU pods):
  - **np-2: 18 nodes x 4 GB300, single subblock = FULL NVL72 domain (72 GPUs)** -> mid_curve capable
  - np-1: 17 nodes, different subblock (one short of full domain)
  - same machine type (`a4x-maxgpu-4g-metal`) + same two taints as pr9
- Infra present: LWS, Kueue, JobSet, nvidia-dra-driver-gpu (ComputeDomain), networking-DRA,
  gcsfuse + lustre storage classes, managed Prometheus
- Infra MISSING vs pr9: **no Dynamo operator/platform, no NATS** — install via NGC helm charts
  (needs NGC key) or go operator-less (plain Deployments/LWS + NATS, srt-slurm-style, per IMAGE
  PLAN fallback). Also no `hf-token-secret`/`nvcr-secret` yet — create in target namespace.

(Sections below about REDACTED-GKE-CLUSTER-OLD describe the OLD cluster; smoke tests all passed there and the
manifests/tooling carry over unchanged.)

## Cluster facts (verified 2026-07-28)

- Cluster: `REDACTED-GKE-CLUSTER-OLD`, project `gke-aishared-gsc-dev`, **regional** in `us-east5` (nodes in us-east5-c)
- Access: kubeconfig on this VM (`~/.kube/config`); auth via VM service account
  `alisa-gcs-sa@gke-aishared-gsc-dev.iam.gserviceaccount.com`, granted cluster-admin via
  clusterrolebindings `alisa-sa-admin` (email) + `alisa-sa-admin-id` (numeric ID `103697796161254948063`)
- GPU nodes: 2 pools (`np-1`, `np-2`), 2 nodes each, machine `a4x-maxgpu-4g-metal`
  - **arm64** (Grace CPU, 144 cores), 983 GB RAM, 4x NVIDIA GB300 per node, ~11 TB local SSD
  - Taints: `nvidia.com/gpu=present:NoSchedule` AND `kubernetes.io/arch=arm64:NoSchedule` (both need tolerations)
  - Topology labels: `cloud.google.com/gce-topology-{block,subblock,host}` (NVLink partition via DRA available: `nvidia-dra-driver-gpu` ns)
- Cluster infra available: LWS, Kueue, JobSet, Dynamo, Lustre CSI, gcsfuse storage classes, DCGM + managed Prometheus
- Proven images on this hardware: `lmsysorg/sglang:latest` (qwen benchmark pod), `lmsysorg/sglang:kimi-k3`

## Model / serving baseline — UPDATED 2026-07-28 after InferenceX comparison

Target reference: InferenceX `dsr1-fp4-gb300-dynamo-sglang` 8k1k **mid_curve**
(see `reference/COMPARISON.md` for the full gap analysis — READ THIS FIRST).

- Model: `nvidia/DeepSeek-R1-0528-NVFP4-v2` (**FP4**, not FP8) — what InferenceX runs on GB300
- Image: `lmsysorg/sglang:v0.5.8.post1-cu130-runtime` (**pinned** — do not use :latest)
- Architecture: Dynamo disaggregated prefill/decode (mid_curve = 6P nodes + 12D nodes = 72 GPUs;
  our cluster tops out at 16 GPUs -> start with 1P1D, full mid_curve needs a capacity ask)
- GKE setup path: AI-Hypercomputer/gpu-recipes `inference/a4x/disaggregated-serving/dynamo/`
  (helm `dynamo-deployment` chart, LWS + ComputeDomain/IMEX for cross-node NVLink, gcsfuse for
  model weights) — local copy in `reference/gke-a4x-dynamo/`
- Benchmark: sa-bench (srt-slurm vendored `benchmark_serving.py`): random dataset, ISL 8192 /
  OSL 1024, range-ratio 0.8, ignore-eos, concurrencies 512/2048/4096
- OLD baseline manifest (aggregated FP8 TP=4, superseded): `manifests/dsr1-sglang.yaml`
- NOTE: a leftover ClusterIP Service `dsr1-sglang` exists in `default` ns (harmless; reuse or delete)

## GPU occupancy (as of setup — recheck before launching!)

- `np-1` both nodes: kimi-k3 (2x 4 GPUs), `np-2/cx7l`: qwen benchmark (4 GPUs)
- `np-2/zcl4`: FREE (4 GPUs) — our target node

Recheck with:
```
kubectl get pods -A -o json | jq -r '.items[] | select(.status.phase=="Running") |
  select([.spec.containers[].resources.requests["nvidia.com/gpu"] // empty] | length > 0) |
  "\(.metadata.name)\t\(.spec.nodeName)"'
```

## Smoke tests (run BEFORE any e2e attempt)

Runner: `./sweeps/run-smoke.sh [0|1|2]` — each stage prints free-GPU occupancy first.
Single-node stages need only the 1 free np-2 node (4 GPUs). Stages are ordered by
cost; stop at the first failure.

### Single-node ladder (available now) — ALL PASSED 2026-07-28

Status: smoke-0 PASS (sglang 0.5.8.post1 confirmed in image; ai-dynamo 0.8.1 aarch64 wheels OK;
4-GPU NCCL allreduce OK; note: image lacks `torchrun` entrypoint -> use
`python3 -m torch.distributed.run`). smoke-1 PASS (sa-bench deps all present in image, no venv
fallback; result: `results/smoke1-qwen3-4b-conc8.json`). smoke-2 PASS (NVFP4 ckpt downloaded
~420 GB in ~14 min via hf_transfer; server healthy in ~28 min total; fp4/trtllm_mla/flashinfer
kernels OK; results: `results/smoke2-dsr1fp4-results_concurrency_{4,16,64}_gpus_4.json`).
Smoke-2 single-node agg TP=4 8k1k numbers (untuned, no warmup — NOT comparable to mid_curve):
conc4 257 tok/s out (TPOT 11.0ms), conc16 512 tok/s (26.1ms), conc64 678 tok/s (81.6ms).

| # | Manifest | Verifies | Duration |
|---|---|---|---|
| 0 | `manifests/smoke-0-sanity.yaml` | pinned image `lmsysorg/sglang:v0.5.8.post1-cu130-runtime` pulls + runs on arm64/GB300; torch/sglang/flashinfer/sgl_kernel/deep_ep/nixl imports; 4-GPU NCCL allreduce; **Dynamo worker stack** (image ships without Dynamo — smoke installs `ai-dynamo==0.8.1` + `ai-dynamo-runtime==0.8.1` aarch64 wheels exactly like srt-slurm's `dynamo_wheels.py` does, then imports `dynamo.sglang`) | ~5-10 min (mostly image pull) |
| 1 | `manifests/smoke-1-tiny-bench.yaml` | serving path + **sa-bench tool** end-to-end against a tiny model (Qwen3-4B, 1 GPU); produces an InferenceX-format result JSON | ~15 min |
| 2 | `manifests/smoke-2-dsr1-fp4-tp4.yaml` | **NVFP4 DSR1 checkpoint** loads; `modelopt_fp4` + `trtllm_mla` + flashinfer kernels on GB300; first single-node 8k1k numbers (conc 4/16/64) | ~1-2 h (420 GB download) |

Notes:
- All smoke stages use the decided image `lmsysorg/sglang:v0.5.8.post1-cu130-runtime`
  (see IMAGE PLAN — exact InferenceX parity). Docker Hub, public, no pull secret needed.
- Stage 1+2 need the `sa-bench-scripts` ConfigMap (runner creates it from `sweeps/sa-bench/`).
- Stage 2 is aggregated TP=4 — a kernel/model smoke, NOT the mid_curve architecture.
- Jobs self-delete after 24 h (`ttlSecondsAfterFinished`); GPUs free on completion.

### Multi-node results on REDACTED-GKE-CLUSTER-OLD (2026-07-29)

- **M0 (2-node MNNVL) PASS**: ComputeDomain claim (`resource.nvidia.com/v1beta1`, channel
  resourceClaimTemplate) + 8-GPU NCCL clique, busbw **723 GB/s**. Manifest:
  `manifests/m0-nccl-2node.yaml`.
- **M0-wide (18-node/72-GPU MNNVL) PASS**: whole np-2 domain in ONE clique, busbw **730 GB/s**
  — fabric healthy at final-job scale. Manifest: `manifests/m0w-nccl-18node.yaml`.
- **RDMA finding (CONFIRMED BROKEN, report to cluster owner)**: np-2 nodes have 8x mlx5 RoCE
  NICs (400G, `gpu{0-3}rdma{0,1}`) and NCCL initializes all 8, BUT:
  (a) dra.net publishes them with `rdma=false` -> `mrdma.google.com` claims unsatisfiable;
  (b) NICs have NO IP addressing — NCCL GIDs are link-local IPv6 (fe80::) only;
  (c) forced NCCL-over-RoCE allreduce FAILS: `Got completion ... status=12` (transport retry
      exceeded, vendor err 129) on every HCA -> no data flows, collective times out.
  Root cause consistent with the GKE RDMA additional-network config being absent on this
  cluster. **Verified on BOTH domains** (np-2 AND np-1: same rdma=false DRA state, same
  status=12 data-path failure) — cluster-wide, not per-domain. Impact: single-NVL72 jobs
  unaffected (NVLink-only, matches InferenceX; KV transfer is NIXL+mooncake with
  MC_FORCE_MNNVL=1 -> NVLink, not RoCE); cross-domain scaling and RoCE fallback impossible
  until fixed. Workaround for helm charts: strip the `all-mrdma` resource claim. Test harness:
  `sweeps/run-rdma-test.sh` (POOL=np-1|np-2).
- Gotcha: pipe NCCL/torchrun output to a FILE, not through `grep | head` — head's SIGPIPE
  truncates/kills the run and fakes success.
- **M2 (Dynamo 1P1D disagg) PASS** (2026-07-30): full disagg pipeline works on GKE with the
  parity image + dynamo 0.8.1 (operator-less; chart etcd+NATS). 8k1k results
  (`results/m2-results_concurrency_{4,8}_gpus_8_disagg_1p1d.json`): conc4 437 tok/s TTFT 1.0s
  TPOT 7.5ms; conc8 817 tok/s TTFT 0.59s TPOT 8.8ms — beats single-node agg (257 tok/s / 4.0s
  / 11ms at conc4). TRANSPORT PROVEN (`results/m2-transport-evidence-ucx-proto.txt`): UCX
  proto tables show CUDA transfers "0..inf zero-copy cuda_ipc/cuda" (MNNVL), tcp control-only.
  Hard-won fixes baked into `manifests/m2-dynamo-1p1d.yaml`: local pre-download (dynamo
  fetch_llm dies on 429), NO UCX_NET_DEVICES (crashes NIXL), UCX_TLS=cuda_copy,cuda_ipc,tcp
  (tcp REQUIRED for AM), `--host 0.0.0.0` (bootstrap binds pod IP), frontend small-file
  snapshot mirror (discovery resolves worker model path), hostPath SSD cache + node pinning.
- **8k1k LOW_LATENCY OFFICIAL SWEEP COMPLETE** (2026-07-30): full InferenceX data point
  (1P + 4D workers, TP=4 each, 20 GPUs; srt-slurm low_latency.yaml verbatim; warmup rate 250
  then measured rate 300 per point). Results `results/ll-results_concurrency_{4,8,32,64}_gpus_20_ctx_4_gen_16.json`.
  METRIC CONVENTION: report tok/s/gpu normalized per role — output tok/s / decode GPUs (16),
  input tok/s / prefill GPUs (4). (Aggregate values live in the result JSONs.)
  | conc | out tok/s/gpu | in tok/s/gpu | TPOT mean | TTFT med/p99 |
  | 4  | 34.9  | 1123 | 6.25ms  | 376ms/1.0s |
  | 8  | 66.8  | 2108 | 6.62ms  | 417ms/1.3s |
  | 32 | 184.9 | 5927 | 8.84ms  | 566ms/4.6s |
  | 64 | 272.9 | 8754 | 11.31ms | 973ms/9.2s |
  Single prefill worker nears saturation at conc 64 (TTFT p99 9.2s) — the topology's design
  limit; mid_curve's 6 prefill workers address exactly this.
- **M1 (2-node TP=8 DSR1 NVFP4 serving) PASS** (2026-07-29): cross-node model load + serving
  over MNNVL works with the parity image. 8k1k: conc4 213 tok/s (TPOT 14.0ms), conc16 454 tok/s
  (TPOT 30.0ms) — results in `results/m1-dsr1fp4-tp8-*.json`. As expected, TP=8-over-2-nodes is
  NOT faster than TP=4 single-node agg (extra cross-node hops; mid_curve gets its wins from
  disagg + wide-EP, not plain TP scaling). Manifest: `manifests/m1-sglang-tp8-2node.yaml`.

### Multi-node ladder (when a full NVLink domain / more nodes free up)

Not yet scripted — reference material in `reference/gke-a4xmax-dynamo/` (A4X Max-specific
gpu-recipes copy, incl. `dynamo-configs/` 1P1D + 10P8D configs). Suggested order:

| # | Test | Verifies | Needs |
|---|---|---|---|
| M0 | ComputeDomain + IMEX channel: apply a 2-node `ComputeDomain` claim (template: `reference/.../common/templates/compute-domain.yaml` in gpu-recipes) and run NCCL `all_reduce_perf` across 2 nodes | cross-node NVLink (MNNVL) via `nvidia-dra-driver-gpu`; `NCCL_MNNVL_ENABLE=1` actually engages | 2 nodes, same subblock |
| M1 | 2-node sglang TP=8 (no Dynamo): `--dist-init-addr` leader/worker, load DSR1 NVFP4 across 8 GPUs | multi-node model load + MNNVL comm in sglang | 2 nodes |
| M2 | Dynamo 1P1D disagg (gpu-recipes `values_wo_deepep.yaml` + `dynamo-configs/deepseek-r1-fp8-1p1d-*.yaml`, helm chart `src/helm-charts/a4xmax/inference-templates/dynamo-deployment`) + sa-bench at conc 4/8 | full disagg control plane: frontend, NATS, NIXL KV transfer, P->D handoff | 2 nodes + Dynamo platform installed (NGC key) |
| M3 | DeepEP smoke: M2 topology with `values_deepep.yaml` decode worker (wide-EP, deepep low_latency) at reduced EP size | DeepEP dispatch/combine over NVL domain — the riskiest mid_curve component | 4+ nodes |
| M4 | Full 8k1k mid_curve (6P+12D, 72 GPUs) with srt-slurm flag parity, sa-bench conc 512/2048/4096 | the real thing | 18 nodes (full NVL72) |

## IMAGE PLAN (decided 2026-07-28, rev2: EXACT InferenceX parity)

**Follow InferenceX exactly — any image difference leads to perf difference:**
- Worker image: **`lmsysorg/sglang:v0.5.8.post1-cu130-runtime`** (the InferenceX pin; arm64 OK)
- Dynamo: **`ai-dynamo==0.8.1` + `ai-dynamo-runtime==0.8.1`** installed INTO that image, exactly
  as srt-slurm's `dynamo_wheels.py` does. aarch64 wheels verified on PyPI
  (`ai_dynamo_runtime-0.8.1-cp310-abi3-manylinux_2_28_aarch64.whl`, abi3 -> py3.12 OK).
- Frontend LB: `nginx:1.27.4` (InferenceX `launch_gb300-nv.sh` pins this too).
- For e2e (not smokes): bake+push a derived image so every pod is byte-identical and startup is
  fast: `FROM lmsysorg/sglang:v0.5.8.post1-cu130-runtime` +
  `RUN pip install ai-dynamo-runtime==0.8.1 ai-dynamo==0.8.1` -> Artifact Registry.
  Smoke-0 instead pip-installs at runtime (same wheels), which is fine for validation.

Rejected: `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.1-cuda13` (team's existing DGD uses it;
Dynamo baked in, but its bundled sglang differs from v0.5.8.post1 -> perf not comparable).

Compatibility risk to watch: the cluster's Dynamo operator is 1.1.1 while our workers run
dynamo 0.8.1. If the operator's DGD path fights the 0.8.1 workers (discovery/health API drift),
mirror srt-slurm's structure instead: plain Deployments/LWS running `python3 -m dynamo.sglang`
workers + `python3 -m dynamo.frontend` + nginx, with NATS request plane (srt-slurm recipe sets
`request_plane: nats`; NATS already runs in `dynamo-system`).

### Existing team infra discovered in `dynamo-cloud` ns (reuse the SCAFFOLDING, swap the image)

- DynamoGraphDeployment `dynamo-dsr1-fp4-8k1k-mid` (operator-managed, replicas 0 = parked):
  frontend + 2x prefill + 2x decode deployments. Currently configured for
  `nvidia/Kimi-K2.6-NVFP4` (configmaps `kimi-smoke-*`), i.e. it's a TEMPLATE to repoint at DSR1.
- Model weights come from gcsfuse bucket `pirillo-sct-bucket` (`huggingface_model_cache/...`)
  with parallel-download file cache — check if DSR1-NVFP4 is already staged there before
  downloading from HF.
- Worker startup script includes a base64 "canary bypass + 420s registration holdoff" patch of
  dynamo/sglang health-check internals — a known workaround for operator canary probes killing
  slow-loading giant models; keep it when adapting the template.
- Secrets in `dynamo-cloud`: `hf-token-secret`, `nvcr-secret` (image pull), `mpi-run-ssh-secret`.
- Env of note: `DYN_DISCOVERY_BACKEND=kubernetes`, `CONFIG_FILE=recipes/gb300-fp8/8k1k/stp/mid.yaml`
  (srt-slurm-style recipe path baked into their flow), GLOO_SOCKET_IFNAME=eth0.

## Layout

```
dsr1-sweep/     DeepSeek-R1 FP4 (InferenceX dsr1-fp4-gb300-dynamo-sglang)
  manifests/      smoke tests, MNNVL fabric tests, 1P1D, low_latency (1P4D), mid_curve (6P+12D)
  reference/      vendored srt-slurm recipes, InferenceX commit, gpu-recipes, COMPARISON.md
  results/        raw sa-bench JSONs (gitignored; archived in GCS)
  results-summary/  metrics-only copies tracked in git (p90/p95 precomputed)
  run-smoke.sh, seed-model-all-nodes.sh, gen-dsr1-report.py

dsv4-sweep/     DeepSeek-V4-Pro mxfp4 + EAGLE MTP (InferenceX dsv4-fp4-gb300-dynamo-sglang-mtp)
  PLAN.md         benchmarking plan + error log of every issue hit and fixed
  manifests/, reference/, results/, results-summary/
  gen-point.py, run-point.sh, seed-dsv4-nodes.sh, gen-dsv4-report.py

common/         shared tooling
  sa-bench/       benchmark client vendored from srt-slurm (+ DSv4 tokenizers)
  stage-model-to-gcs.sh, slim-results.py, run-rdma-test.sh

reports/        benchmark reports (DSR1, DSv4) + GKE RDMA issue writeup
```

Regenerate reports after new results:
```
python3 common/slim-results.py       # raw -> results-summary/
python3 dsr1-sweep/gen-dsr1-report.py
python3 dsv4-sweep/gen-dsv4-report.py
```

## Model weights (staged 2026-07-30)

**`gs://alisachen-models/deepseek-ai/DeepSeek-R1-0528-NVFP4-v2/`** — full NVFP4 checkpoint,
174 objects / 413.3 GB, verified. Uploaded from the M2 decode pod at 3.0 GiB/s
(`sweeps/stage-model-to-gcs.sh`). Use this (gcsfuse or bulk copy) for ALL future
deployments — do NOT download from HF again (anonymous 429 rate limits; and dynamo 0.8.1's
built-in fetch_llm dies on the first 429 — workers must be given a local/gcsfuse path or use
the resilient snapshot_download loop in `manifests/m2-dynamo-1p1d.yaml`).

## Benchmark measurement methodology (adopted 2026-07-30 — REQUIRED for all measured points)

Match srt-slurm/InferenceX bench.sh defaults exactly (`NUM_WARMUP_MULT=2`, `NUM_PROMPTS_MULT=10`):

1. **Bigger warmup — 2x concurrency prompts at request-rate 250** before every measured run
   (e.g. conc 64 -> 128 warmup prompts). Fully heats kernel autotuning caches, allocator
   pools, and scheduler state; measured effect: TTFT median -26% at conc 64.
2. **Longer measured run — 10x concurrency prompts** at the recipe's req_rate (e.g. conc 64
   -> 640 prompts). Throughput = tokens/wall-duration INCLUDING ramp (0->conc) and drain
   (conc->0) edges; short runs overweight those partial-load edges and understate steady
   state. Measured effect at conc 64: +11.1% output tok/s/gpu (272.9 -> 303.1) with TPOT
   flat (+1.4%) — proving it's measurement fairness, not a server change.

Any result taken with smaller multipliers must be labeled non-comparable (see `(mult10)`
labels in `reports/benchmark-report-dsr1-8k1k.md`). Applies to every config (low_latency,
mid_curve, max_tpt, DSv4 sweep).

## Transport-evidence policy (adopted 2026-07-29)

Every disaggregated worker (M2 onward, incl. the final mid_curve job) carries
**`UCX_LOG_LEVEL=info`** in its env, IN ADDITION to the recipe's own `MC_TE_METRIC=true` and
`MC_FORCE_MNNVL=1`, PLUS the CANONICAL UCX block in `manifests/worker-env-canonical.yaml`
(rev2, user decision 2026-07-29 — allowlist style from teammate's proven GKE values):
`UCX_TLS=cuda_copy,cuda_ipc,rc_x`, `UCX_CUDA_IPC_ENABLE_MNNVL=y` (REQUIRED for cross-node
NVLink KV transfer), `UCX_MEMTYPE_CACHE=n`, `UCX_MEMTYPE_REG_WHOLE=n`, `UCX_PROTO_INFO=y`
(per-op protocol evidence), `UCX_NET_DEVICES=mlx5_0..7`, `UCX_IB_GID_INDEX=5`. TCP absent
from the allowlist -> silent fallback impossible; `rc_x` inert until cluster RDMA is fixed,
then activates automatically (with the mrdma claim re-added). Embed the file's block verbatim
in every M2/M4 worker container. Purpose: NIXL (KV transfer) rides UCX — the info logs record which
transport UCX selects per endpoint, so each real workload run yields auditable proof of the
KV path. How to check after a run:

```
kubectl logs <decode-or-prefill-pod> | grep -iE "ucx.*(using|selected|transport)|cuda_ipc|(rc|dc|ud)_(mlx5|verbs)|mlx5_[0-9]:|/tcp"
```

- `cuda_ipc` / `cuda` transports -> NVLink (MNNVL/IMEX) — EXPECTED on this cluster
- ANY of `rc_mlx5|dc_mlx5|ud_mlx5|rc_verbs|ud_verbs` (often device-qualified like
  `rc_mlx5/mlx5_2:1`) -> RoCE RDMA — unexpected here (RDMA broken cluster-wide).
  Caveat: `rdmacm` alone is just connection management, NOT proof of RDMA data path.
- `tcp` for bulk data -> BAD: silent fallback, numbers invalid — investigate immediately

Layer-specific RDMA markers (different vocabularies!):
- NCCL (TP/EP collectives): `NET/IB` lines (`NCCL_DEBUG=INFO`, SUBSYS INIT,NET)
- NVSHMEM/DeepEP (MoE a2a): `ibgda` (needs `NVSHMEM_DEBUG=INFO`) — add this env on the
  first disagg validation run too, since DeepEP does not log through UCX at all

Also grep mooncake metrics (`MC_TE_METRIC`) for per-transfer bandwidth sanity (NVLink-class
= hundreds of GB/s), and optionally snapshot hardware counters around bench points:
`nvidia-smi nvlink -gt d` (should grow) vs `/sys/class/infiniband/mlx5_*/ports/1/counters/
port_xmit_data` (must stay flat). If UCX log volume proves costly at final-run scale, keep
the flag for one validation point per config, then drop it for the record run.

## Sweep plan (draft)

Fixed: model=DeepSeek-R1 FP8, 1 node, TP=4, SGLang.
Sweep axes (typical Pareto knobs):
- Request concurrency / request rate (e.g. 1, 4, 8, 16, 32, 64, 128)
- Input/output length mix (e.g. 1k/1k, 4k/1k, 1k/4k)
- Server knobs: `--max-running-requests`, `--mem-fraction-static`, chunked prefill size,
  spec-decode (MTP) on/off
Benchmark driver: `sglang.bench_serving` (or genai-bench) against the in-cluster service,
one JSON result per point -> `results/`, then plot goodput vs per-token latency in `analysis/`.

---
*Public release note: project/cluster identifiers and internal
doc links are redacted (`REDACTED-*` placeholders). The benchmark data, configs, methodology,
and reports are complete. Full provenance artifacts live in internal storage.*
