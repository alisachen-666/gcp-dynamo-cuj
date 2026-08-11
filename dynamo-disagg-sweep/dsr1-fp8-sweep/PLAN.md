# DeepSeek-R1 FP8 GB300 8k1k Sweep — Benchmarking Plan (GKE A4X Max)

Goal: reproduce **all 8 InferenceX `dsr1-fp8-gb300-dynamo-sglang` data points** on GKE A4X Max
(`REDACTED-GKE-CLUSTER-OLD`, np-2 NVL72 domain), following config, benchmark tool, topology and images strictly.

This is the third workstream, after `../dsr1-sweep/` (FP4) and `../dsv4-sweep/`. Every GKE lesson
from those two is carried over here rather than rediscovered — see "Lessons pre-applied".

## Pinned sources (copies in `reference/`)

- InferenceX matrix entry: SemiAnalysisAI/InferenceX @ `cc40942d`,
  `configs/nvidia-master.yaml` L3077 key `dsr1-fp8-gb300-dynamo-sglang`
  (`reference/inferencex-dsr1-fp8-gb300-dynamo-sglang.yaml`)
- Recipes: NVIDIA/srt-slurm branch **`sa-submission-q2-2026` @ `deb1dfd9`**,
  `recipes/gb300-fp8/8k1k/stp/{low-latency,mid,max}.yaml` (vendored verbatim)
- `reference/deepep_config.json` — srt-slurm `configs/deepep_config.json`, required by
  `--deepep-config` on every mid/max worker
- Exact SHAs in `reference/PINS.txt`

## STRICT parity requirements

| Aspect | Value (do not substitute) | Delta vs our FP4 run |
|---|---|---|
| Image | `lmsysorg/sglang:v0.5.8.post1-cu130` | **no `-runtime` suffix** (FP4 used `...-cu130-runtime`) |
| Model | `deepseek-ai/DeepSeek-R1-0528`, precision **fp8** (native ckpt) | FP4 used `nvidia/DeepSeek-R1-0528-NVFP4-v2` |
| Served name | `deepseek-ai/DeepSeek-R1` | same |
| Dynamo | **0.8.0** (srt-slurm `DynamoConfig` default; InferenceX `router: dynamo-router 0.8.0`) | **FP4 was 0.8.1** |
| Request plane | NATS (schema default) | same |
| KV transfer | `nixl` (`disaggregation-transfer-backend: nixl`) | same |
| Quantization | `quantization: fp8`, `fp8-gemm-backend: flashinfer_trtllm` | FP4: `modelopt_fp4` + `fp4-gemm-backend flashinfer_cutlass` |
| KV cache / attention | `fp8_e4m3` / `trtllm_mla` | same |
| Context length | **9300** | FP4 used 9600 |
| Spec decode | none (STP) | same |
| Frontends | 10 (1 + `num_additional_frontends: 9`, schema default) | same |
| Benchmark | sa-bench (`../common/sa-bench/`), ISL 8192 / OSL 1024, range-ratio 0.8, ignore-eos | same tool |
| req_rate | **`inf`** for all three configs | **FP4 used 300 (low-lat) / 700 (mid)** |
| Methodology | warmup 2x conc @ rate 250, measured **10x conc** (srt-slurm `bench.sh` defaults) | adopted from day 1 here |
| Transport | NVLink: `MC_FORCE_MNNVL=1`, `NCCL_MNNVL_ENABLE=1`, `NCCL_CUMEM_ENABLE=1` | same |

### Supply-chain gates — BOTH VERIFIED (2026-07-31), no forced deviation

- `lmsysorg/sglang:v0.5.8.post1-cu130`: **arm64/linux manifest present**, 15.6 GB, active,
  pushed 2026-02-05. (Unlike the DSv4 nightly, this is a release tag — not subject to nightly pruning.)
- `ai-dynamo==0.8.0` (`py3-none-any`) + `ai-dynamo-runtime==0.8.0`
  (`cp310-abi3-manylinux_2_28_aarch64`) exist on PyPI → same runtime pip-install pattern as FP4's
  0.8.1. **No source build needed** (contrast: DSv4 needed an in-cluster Rust build of dynamo@81d0555).
- `deepseek-ai/DeepSeek-R1-0528`: public, NOT gated, 163 safetensors shards, **688.6 GB**
  (vs 413 GB for the FP4 ckpt — budget ~1.7x the staging time and node SSD).

## The 8 data points (authoritative: InferenceX `search-space`)

| # | Config | Recipe | Prefill | Decode | Nodes | Conc |
|---|---|---|---|---|---|---|
| L1 | low-latency | `stp/low-latency.yaml` | 1w TP4 EP1, no dp-attn (1 node) | 1w TP4 EP1, no dp-attn (1 node) | **2** | 4 |
| L2 | low-latency | same | same | same | 2 | 8 |
| M1 | mid | `stp/mid.yaml` | **5w DEP8** (TP8/EP8/dp-attn, 2 nodes each = 10 nodes) | 1w **DEP32** (8 nodes) | **18** | 128 |
| M2 | mid | same | same | same | 18 | 256 |
| M3 | mid | same | same | same | 18 | 512 |
| M4 | mid | same | same | same | 18 | 1024 |
| X1 | max | `stp/max.yaml` | **6w DEP8** (2 nodes each = 12 nodes) | 1w **DEP24** (6 nodes) | **18** | 2048 |
| X2 | max | same | same | same | 18 | 4096 |

Node budget: 2 / 18 / 18 — all fit np-2 exactly. Only **3 server bring-ups** are needed for
8 points (concurrencies sweep against a standing deployment), so the expensive part is 3 model
loads, not 8.

**New structural challenge vs FP4**: FP8 mid/max prefill workers are **multi-node** (DEP8 = 8 GPUs
= 2 nodes per worker), where FP4 prefill workers were single-node TP4. So prefill becomes N
independent 2-node StatefulSets (one per worker, each with its own `--dist-init-addr` leader and
headless Service), not one Deployment with N replicas. Decode is a single 8-node (mid) / 6-node
(max) StatefulSet, same shape as FP4's 12-node decode.

## Execution phases

- **F0 — image/env validation** (1 node): sanity job on `v0.5.8.post1-cu130`: imports, GB300
  kernels, `pip install ai-dynamo{,-runtime}==0.8.0` + `import dynamo.sglang`, NIXL agent creation,
  4-GPU NCCL allreduce. Gate for everything else. (~10 min; mostly image pull.)
- **F1 — model staging**: `deepseek-ai/DeepSeek-R1-0528` (688.6 GB) → `gs://alisachen-models/`
  then seed to all 18 nodes' local SSD at `/mnt/stateful_partition/kube-ephemeral-ssd/dsr1-fp8-model`.
  Reuse `../common/stage-model-to-gcs.sh` + `seed-model-all-nodes.sh`.
- **F2 — L1/L2** (2 nodes): low-latency conc 4, 8. Smallest e2e; proves FP8 quantization path,
  dynamo 0.8.0, and the DG-cache decision (below) before committing 18 nodes.
- **F3 — M1..M4** (18 nodes): mid, conc 128/256/512/1024 against one standing deployment.
  First DeepEP point → carry `NVSHMEM_DEBUG=INFO` on the first run for a2a evidence.
- **F4 — X1/X2** (18 nodes): max, conc 2048/4096. Teardown of F3 required first.

Record runs: one config at a time on an otherwise-idle domain (no interference), per the DSv4
directive.

## GKE translation (every M2/DSR1/DSv4 lesson pre-applied)

1. **Operator-less Dynamo**: plain Deployments/StatefulSets + `python3 -m dynamo.frontend`;
   etcd + NATS reused from the `dynamo-cloud` chart install. No DGD CRs (RBAC-blocked).
2. **`--host 0.0.0.0`** on every worker (bootstrap server binds pod IP — M2 lesson).
3. **Model from local hostPath**, never per-pod HF download (429s; dynamo's `fetch_llm` dies on
   the first 429). Workers assert the checkpoint is seeded and fail fast if not.
4. **Frontend small-file mirror** at the *same path* workers register (`/model`), else discovery
   cannot resolve the model path.
5. **ComputeDomain** one CR per deployment sized to its node count (2 / 18 / 18); every worker pod
   carries the channel claim. **No `mrdma` claim** — cluster RDMA is broken (see
   `../reports/gke-rdma-issue-REDACTED-GKE-CLUSTER-OLD.md`) and NVLink-only matches `MC_FORCE_MNNVL` anyway.
6. **UCX env**: `UCX_TLS=cuda_copy,cuda_ipc,tcp` (tcp REQUIRED for AM), `UCX_CUDA_IPC_ENABLE_MNNVL=y`,
   `UCX_MEMTYPE_CACHE=n`, `UCX_MEMTYPE_REG_WHOLE=n`, `UCX_PROTO_INFO=y`.
   **NEVER `UCX_NET_DEVICES=mlx5...`** — crashes NIXL.
7. **Multi-node workers**: StatefulSet + headless Service, `--dist-init-addr <sts>-0.<svc>:5757
   --nnodes N --node-rank $POD_INDEX` (from `apps.kubernetes.io/pod-index`).
8. **`/configs` ConfigMap** mounting `deepep_config.json` on every mid/max worker
   (`--deepep-config /configs/deepep_config.json`).
9. **Bench pods**: `nodeSelector: np-2` (hostPath check fails otherwise) and `cp -L` when copying
   the sa-bench ConfigMap (symlink dereference — DSv4 bug).
10. **Hard readiness gate**: require **3 consecutive successful `/v1/completions`** through the
    Service before benching. Never gate on a registration count — it is `prefill_workers + 1`
    and racing it produces `Not Found` 404s (DSv4 bug).
11. **JIT cache** persisted to node hostPath so each node compiles once across all points.
12. **Log to a file, not through `grep | head`** — SIGPIPE truncation fakes success.

## Deviations / open risks

- **[D1] DeepGEMM cache dir.** All three recipes set `SGLANG_DG_CACHE_DIR=/configs/dg-10212025`,
  and low-latency additionally sets `SGLANG_ENABLE_JIT_DEEPGEMM=false`. That directory is a
  **prebuilt DeepGEMM kernel cache on their Slurm shared filesystem and is NOT in the srt-slurm
  repo** — we cannot obtain it. Plan: point `SGLANG_DG_CACHE_DIR` at a node hostPath
  (`.../dsr1-fp8-dg-cache`, shared across points) and run **JIT enabled once** to populate it,
  then honour the recipe's `false` for the recorded runs. mid/max do not set the JIT flag at all
  (JIT on by default), so this deviation is confined to low-latency and is expected to affect
  startup time, not steady-state throughput. **Must be stated in the report.**
- **[D2] Frontend load-balancing.** srt-slurm fronts the 10 dynamo frontends with `nginx`; we use
  a k8s Service. Same deviation as the FP4 run, which matched InferenceX to its ceiling — carried
  forward knowingly.
- **[D3] KV transport is MNNVL cuda_ipc, not RoCE RDMA** (cluster RDMA broken). The FP4 report
  established with TPOT-invariance evidence that this costs no decode throughput; re-verify with
  `UCX_PROTO_INFO` on the first run of each config.
- **[D4] `skip-tokenizer-init: true`** in mid/max means the frontend owns tokenization — confirm
  the small-file mirror includes `tokenizer.json` + `tokenizer_config.json` (it does).
- **[R1] Node budget**: mid and max each need the whole 18-node np-2 domain, so F3 and F4 are
  strictly serial, and any DSv4 rerun contends for the same domain.
- **[R2] 688.6 GB checkpoint** — staging and per-node seeding is the long pole (FP4's 413 GB took
  ~14 min HF→pod at hf_transfer speeds and uploaded to GCS at 3.0 GiB/s; expect ~1.7x).
- **[R3] Credentials**: this VM currently has **no kubeconfig and no ADC** (see "Blocked on" below).

## Blocked on

Cluster access. This VM has `kubectl` but no kubeconfig, no `gcloud`, and its GCE default SA
(`455207029971-compute@`) returns `ACCESS_TOKEN_SCOPE_INSUFFICIENT` for the container API.
The cluster endpoint itself IS reachable (`https://136.83.38.72` → 401, i.e. network path fine).
Per `../README.md`, auth is user ADC copied from cloudtop and expires ~daily.

## Error log (running; per goal directive)

- (none yet — GPU phases not started)

## Directory layout

- `PLAN.md` — this file
- `reference/` — 3 verbatim srt-slurm recipes, deepep_config.json, InferenceX block, PINS.txt
- `manifests/` — per-config GKE manifests + per-point bench jobs
- `results/` — raw sa-bench JSONs (gitignored; archived in GCS)
- `results-summary/` — metrics-only copies tracked in git
- `gen-fp8-point.py`, `run-point.sh`, `gen-dsr1fp8-report.py`
