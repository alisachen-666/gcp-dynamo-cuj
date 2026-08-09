# End-to-End Analysis: Disaggregated Serving Benchmarking on GKE A4X Max (GB300 NVL72)

## Motivation

In SemiAnalysis's InferenceX open-source benchmarking platform, NVIDIA Dynamo serves as a
datacenter-scale, multi-node orchestration layer. Rather than replacing execution engines like
vLLM, SGLang, or TensorRT-LLM, Dynamo sits above to coordinate split prefill/decode pools,
manage cluster-wide KV-caches, and enable wide expert parallelism across scale-up fabrics
(like NVLink on NVL72 racks).

## Existing Benchmarking Efforts

- **[External] GB200 DeepSeek R1 Benchmarks** — the GB200 NVL72 generation established the
  disaggregated DSR1 baseline that InferenceX now publishes across hardware generations.
- **GB300 bring-up on GKE** — go/nvidia-dynamo-on-a4x-max-gke: operator-less Dynamo
  (etcd + NATS + plain Deployments/StatefulSets) brought up on `a4x-maxgpu-4g-metal`
  (4× GB300/node, NVL72 scale-up domain), the substrate for everything below.
- **[Internal] Inference Scaling on GB200** — Kuo's doc on engine-level scaling behavior.

**Our value-add is a disaggregated-serving benchmark harness on GKE whose methodology,
recipes, and metrics conform to InferenceX** — so GKE numbers are directly comparable against
the published bare-metal Slurm results, point by point, with the config provenance to prove it.

## Result

The output is a Pareto curve per model (throughput per GPU vs interactivity), 7–10 points each:
https://screenshot.googleplex.com/Jb56JJPESdgu29q

Headline: **of the 18 InferenceX points reproduced (8 DSR1-FP4 + 10 DSv4), we lead or match
on 16**; the two open deficits (DSR1 max_tpt −7.6%, DSR1 mid conc-2048 −2.7%) are
root-caused to scheduler-layer effects, not configuration drift (§ analysis below).

---

## Disaggregated Serving benchmark harness

**Concurrency space** (InferenceX convention) split into 4 regions:
- **Low Latency**: concurrency 1–32 (fixed; at least 1 point required)
- **Low/Med/High Throughput**: 33 → M, split into 3 equal regions in log₂ space

Our swept points land as: DSR1 conc {4, 8, 32, 64} (low-latency), {512, 2048, 4096}
(mid-curve), {2048 @ 40P/32D} (max-throughput); DSv4 conc {1, 8, 32, 64} (low-latency),
{256, 512} (mid), {1024, 4096, 8192} (high-concurrency).

**第一步**: TODO(alisachen) — Check why xPyD changes with concurrency on the graph.
Our observation from the sweep: the published curve switches topology per region because each
recipe fixes one xPyD shape and is only Pareto-optimal inside its concurrency band — e.g.
DSR1 4P/16D wins below conc ~64, 24P/48D through the knee, 40P/32D only at saturation; the
"change" is the envelope over fixed-topology curves, not a dynamic reconfiguration.

**第二步**: Sweep the three regions from
`https://github.com/SemiAnalysisAI/InferenceX/blob/main/.github/configs/nvidia-master.yaml#L6437-L6452`
(low-latency / mid-curve / high-throughput), recipes vendored verbatim from srt-slurm
@ `sa-submission-q2-2026` (DSR1: `recipes/gb300-fp4/8k1k/`; DSv4:
`recipes/dsv4-pro/sglang/gb300-fp4/8k1k/disagg/`).

### DSR1-FP4 8k1k (GCP cells = optimal final run per point; `value (gap vs InferenceMax)`)

| Pareto region | Prefill-Decode topology | Conc | Sources | Req plane | KV transfer | Total Tput (tok/s/GPU) | Input Tput (tok/s/GPU) | Output Tput (tok/s/GPU) | Median E2EL (s) | Median ITL (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| Low latency | 4P/16D | 4 | InferenceMax | NATS | MNNVL (NIXL) | 247.4 | 1,099.5 | 34.4 | 6.40 | 0.33 |
| | | | GCP bench with GKE RDMA-KV [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-llr) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsr1-llr-bench.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsr1-sweep/manifests/llr-lowlatency-rdmakv.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/llr-lowlatency-rdmakv.yaml;tab=live_object) | NATS | RDMA (rc_mlx5) | 267.4 (+8.1%) | 1,188.1 (+8.1%) | 37.2 (+8.1%) | 6.01 (-6.1%) | 0.31 (-6.6%) |
| Low latency | 4P/16D | 8 | InferenceMax | NATS | MNNVL (NIXL) | 457.8 | 2,031.8 | 64.3 | 6.99 | 0.37 |
| | | | GCP bench with GKE RDMA-KV [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-llr) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsr1-llr-bench.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsr1-sweep/manifests/llr-lowlatency-rdmakv.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/llr-lowlatency-rdmakv.yaml;tab=live_object) | NATS | RDMA (rc_mlx5) | 495.2 (+8.2%) | 2,197.8 (+8.2%) | 69.6 (+8.2%) | 6.57 (-6.1%) | 0.33 (-11.9%) |
| Low latency | 4P/16D | 32 | InferenceMax | NATS | MNNVL (NIXL) | 1,374.7 | 6,105.1 | 192.1 | 9.09 | 0.46 |
| | | | GCP bench with GKE [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-lowlatency) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsr1-sweep/manifests/m4b-lowlatency-1p4d.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/m4b-lowlatency-1p4d.yaml;tab=live_object) | TCP | MNNVL (cuda_ipc) | 1,415.1 (+2.9%) | 6,284.2 (+2.9%) | 197.8 (+2.9%) | 8.87 (-2.5%) | 0.45 (-2.1%) |
| Low latency | 4P/16D | 64 | InferenceMax | NATS | MNNVL (NIXL) | 2,093.7 | 9,307.2 | 290.3 | 11.98 | 0.61 |
| | | | GCP bench with GKE [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-lowlatency) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsr1-sweep/manifests/m4b-lowlatency-1p4d.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/m4b-lowlatency-1p4d.yaml;tab=live_object) | TCP | MNNVL (cuda_ipc) | 2,185.7 (+4.4%) | 9,716.0 (+4.4%) | 303.1 (+4.4%) | 11.43 (-4.6%) | 0.58 (-3.7%) |
| Mid curve | 24P/48D | 512 | InferenceMax | NATS | MNNVL (NIXL) | 2,653.3 | 7,075.7 | 442.2 | 15.14 | 0.64 |
| | | | GCP bench with GKE RDMA-KV [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-m4r-rev3-3pts) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/bench-m4r-c512-rev3.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsr1-sweep/manifests/m4r-midcurve-rdmakv.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/m4r-midcurve-rdmakv.yaml;tab=live_object) | NATS | RDMA (rc_mlx5) | 2,946.9 (+11.1%) | 7,858.4 (+11.1%) | 491.1 (+11.1%) | 13.74 (-9.3%) | 0.64 (-0.5%) |
| Mid curve | 24P/48D | 2048 | InferenceMax | NATS | MNNVL (NIXL) | 5,028.8 | 13,408.8 | 838.8 | 44.28 | 0.66 |
| | | | GCP bench with GKE RDMA-KV [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-m4r-rev3-3pts) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/bench-m4r-c2048-rev3.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsr1-sweep/manifests/m4r-midcurve-rdmakv.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/m4r-midcurve-rdmakv.yaml;tab=live_object) | NATS | RDMA (rc_mlx5) | 4,967.4 (-1.2%) | 13,244.9 (-1.2%) | 828.7 (-1.2%) | 41.85 (-5.5%) | 0.69 (+4.4%) |
| Mid curve | 24P/48D | 4096 | InferenceMax | NATS | MNNVL (NIXL) | 4,594.4 | 12,252.1 | 765.5 | 92.94 | 0.68 |
| | | | GCP bench with GKE [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-midcurve) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsr1-sweep/manifests/m4-midcurve.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/m4-midcurve.yaml;tab=live_object) | TCP | MNNVL (cuda_ipc) | 5,025.6 (+9.4%) | 13,401.9 (+9.4%) | 837.4 (+9.4%) | 84.29 (-9.3%) | 0.90 (+32.5%) |
| Max throughput | 40P/32D | 2048 | InferenceMax | NATS | MNNVL (NIXL) | 7,120.5 | 11,391.7 | 1,781.6 | 26.39 | 0.86 |
| | | | GCP bench with GKE RDMA-KV [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-mxr) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsr1-mxr-bench.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsr1-sweep/manifests/mxr-maxtpt-rdmakv.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/mxr-maxtpt-rdmakv.yaml;tab=live_object) | NATS | RDMA (rc_mlx5) | 7,080.3 (-0.6%) | 11,327.1 (-0.6%) | 1,771.7 (-0.6%) | 27.44 (+4.0%) | 0.90 (+4.5%) |

<!-- BEGIN dsr1-network-tables (generated; source: benchmark-report-rdma-kv.md) -->
## DSR1-FP4: MNNVL vs RDMA — per-network tables

Format matches the main summary table: InferenceMax row = absolute values; GCP row = `value (gap vs InferenceMax)` per metric. Throughput: higher is better; TPOT/TTFT/ITL: lower is better (negative gap = we are faster).

### Table 1 — KV over MNNVL (NVLink; final per point, optimal variant)

| Region | Topology | Conc | Sources | Out/decode-GPU (tok/s) | TPOT mean (ms) | TTFT med (s) | Median ITL (s) |
|---|---|---|---|---|---|---|---|
| Low latency | 4P/16D | 4 | InferenceMax | 34.4 | 6.73 | 0.24 | 0.33 |
| | | | GCP bench with GKE NATS mult10 [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-lln) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsr1-lln-bench.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsr1-sweep/manifests/lln-lowlatency-nats.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/lln-lowlatency-nats.yaml;tab=live_object) | 36.5 (+6.1%) | 6.24 (-7.4%) | 0.25 (+6.0%) | 0.31 (-6.5%) |
| Low latency | 4P/16D | 8 | InferenceMax | 64.3 | 7.09 | 0.27 | 0.37 |
| | | | GCP bench with GKE NATS mult10 [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-lln) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsr1-lln-bench.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsr1-sweep/manifests/lln-lowlatency-nats.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/lln-lowlatency-nats.yaml;tab=live_object) | 69.3 (+7.7%) | 6.61 (-6.7%) | 0.38 (+38.1%) | 0.33 (-11.6%) |
| Low latency | 4P/16D | 32 | InferenceMax | 192.1 | 9.13 | 0.62 | 0.46 |
| | | | GCP bench with GKE [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-lowlatency) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsr1-sweep/manifests/m4b-lowlatency-1p4d.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/m4b-lowlatency-1p4d.yaml;tab=live_object) | 197.8 (+2.9%) | 8.86 (-2.9%) | 0.57 (-7.3%) | 0.45 (-2.1%) |
| Low latency | 4P/16D | 64 | InferenceMax | 290.3 | 12.02 | 0.73 | 0.61 |
| | | | GCP bench with GKE [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-lowlatency) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsr1-sweep/manifests/m4b-lowlatency-1p4d.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/m4b-lowlatency-1p4d.yaml;tab=live_object) | 303.1 (+4.4%) | 11.47 (-4.6%) | 0.72 (-1.5%) | 0.58 (-3.7%) |
| Mid curve | 24P/48D | 512 | InferenceMax | 442.2 | 13.08 | 3.39 | 0.64 |
| | | | GCP bench with GKE NATS A/B [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-midcurve) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsr1-sweep/manifests/m4-midcurve.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/m4-midcurve.yaml;tab=live_object) | 469.0 (+6.1%) | 12.95 (-1.0%) | 2.18 (-35.7%) | 0.64 (-0.2%) |
| Mid curve | 24P/48D | 2048 | InferenceMax | 838.8 | 13.25 | 32.03 | 0.66 |
| | | | GCP bench with GKE [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-midcurve) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsr1-sweep/manifests/m4-midcurve.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/m4-midcurve.yaml;tab=live_object) | 816.5 (-2.7%) | 18.71 (+41.2%) | 23.60 (-26.3%) | 0.90 (+37.0%) |
| Mid curve | 24P/48D | 4096 | InferenceMax | 765.5 | 13.57 | 80.43 | 0.68 |
| | | | GCP bench with GKE [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-midcurve) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsr1-sweep/manifests/m4-midcurve.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/m4-midcurve.yaml;tab=live_object) | 837.4 (+9.4%) | 17.90 (+31.9%) | 67.72 (-15.8%) | 0.90 (+32.5%) |
| Max throughput | 40P/32D | 2048 | InferenceMax | 1,781.6 | 17.12 | 10.24 | 0.86 |
| | | | GCP bench with GKE TCP A/B [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-mxt) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsr1-mxt-bench.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsr1-sweep/manifests/mxt-maxtpt-tcp.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/mxt-maxtpt-tcp.yaml;tab=live_object) | 1,646.7 (-7.6%) | 17.18 (+0.4%) | 15.84 (+54.7%) | 0.87 (+1.3%) |

### Table 2 — KV over GPUDirect RDMA (rev4: `UCX_NET_DEVICES` + GID 5 + rail-aware pairing)

| Region | Topology | Conc | Sources | Out/decode-GPU (tok/s) | TPOT mean (ms) | TTFT med (s) | Median ITL (s) |
|---|---|---|---|---|---|---|---|
| Low latency | 4P/16D | 4 | InferenceMax | 34.4 | 6.73 | 0.24 | 0.33 |
| | | | GCP bench with GKE RDMA-KV [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-llr) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsr1-llr-bench.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsr1-sweep/manifests/llr-lowlatency-rdmakv.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/llr-lowlatency-rdmakv.yaml;tab=live_object) | 37.2 (+8.1%) | 6.21 (-7.7%) | 0.24 (+0.1%) | 0.31 (-6.6%) |
| Low latency | 4P/16D | 8 | InferenceMax | 64.3 | 7.09 | 0.27 | 0.37 |
| | | | GCP bench with GKE RDMA-KV [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-llr) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsr1-llr-bench.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsr1-sweep/manifests/llr-lowlatency-rdmakv.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/llr-lowlatency-rdmakv.yaml;tab=live_object) | 69.6 (+8.2%) | 6.57 (-7.4%) | 0.37 (+37.1%) | 0.33 (-11.9%) |
| Low latency | 4P/16D | 32 | InferenceMax | 192.1 | 9.13 | 0.62 | 0.46 |
| | | | GCP bench with GKE RDMA-KV [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-llr) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsr1-llr-bench.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsr1-sweep/manifests/llr-lowlatency-rdmakv.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/llr-lowlatency-rdmakv.yaml;tab=live_object) | 195.1 (+1.5%) | 8.80 (-3.6%) | 0.81 (+31.4%) | 0.44 (-3.4%) |
| Low latency | 4P/16D | 64 | InferenceMax | 290.3 | 12.02 | 0.73 | 0.61 |
| | | | GCP bench with GKE RDMA-KV [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-llr) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsr1-llr-bench.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsr1-sweep/manifests/llr-lowlatency-rdmakv.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/llr-lowlatency-rdmakv.yaml;tab=live_object) | 300.2 (+3.4%) | 11.37 (-5.4%) | 0.93 (+27.7%) | 0.57 (-5.2%) |
| Mid curve | 24P/48D | 512 | InferenceMax | 442.2 | 13.08 | 3.39 | 0.64 |
| | | | GCP bench with GKE RDMA-KV [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-m4r-rev3-3pts) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/bench-m4r-c512-rev3.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsr1-sweep/manifests/m4r-midcurve-rdmakv.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/m4r-midcurve-rdmakv.yaml;tab=live_object) | 491.1 (+11.1%) | 12.70 (-2.9%) | 1.33 (-60.8%) | 0.64 (-0.5%) |
| Mid curve | 24P/48D | 2048 | InferenceMax | 838.8 | 13.25 | 32.03 | 0.66 |
| | | | GCP bench with GKE RDMA-KV [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-m4r-rev3-3pts) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/bench-m4r-c2048-rev3.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsr1-sweep/manifests/m4r-midcurve-rdmakv.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/m4r-midcurve-rdmakv.yaml;tab=live_object) | 828.7 (-1.2%) | 13.75 (+3.8%) | 29.31 (-8.5%) | 0.69 (+4.4%) |
| Mid curve | 24P/48D | 4096 | InferenceMax | 765.5 | 13.57 | 80.43 | 0.68 |
| | | | GCP bench with GKE RDMA-KV [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-m4r-rev3-3pts) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/bench-m4r-c4096-rev3.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsr1-sweep/manifests/m4r-midcurve-rdmakv.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/m4r-midcurve-rdmakv.yaml;tab=live_object) | 830.8 (+8.5%) | 13.87 (+2.2%) | 73.21 (-9.0%) | 0.69 (+2.0%) |
| Max throughput | 40P/32D | 2048 | InferenceMax | 1,781.6 | 17.12 | 10.24 | 0.86 |
| | | | GCP bench with GKE RDMA-KV [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-mxr) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsr1-mxr-bench.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsr1-sweep/manifests/mxr-maxtpt-rdmakv.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/mxr-maxtpt-rdmakv.yaml;tab=live_object) | 1,771.7 (-0.6%) | 17.85 (+4.2%) | 10.78 (+5.2%) | 0.90 (+4.5%) |

### Network delta (RDMA vs MNNVL, same recipe/methodology/cluster)

| Region | Conc | Δ Out tput | Δ TPOT (lower better) | Δ TTFT med | Δ ITL med |
|---|---|---|---|---|---|
| Low latency | 4 | +1.9% | -0.4% | -5.5% | -0.1% |
| Low latency | 8 | +0.4% | -0.7% | -0.7% | -0.3% |
| Low latency | 32 | -1.4% | -0.7% | +41.8% | -1.3% |
| Low latency | 64 | -1.0% | -0.9% | +29.7% | -1.5% |
| Mid curve | 512 | +4.7% | -1.9% | -39.0% | -0.4% |
| Mid curve | 2048 | +1.5% | -26.5% | +24.2% | -23.8% |
| Mid curve | 4096 | -0.8% | -22.5% | +8.1% | -23.1% |
| Max throughput | 2048 | +7.6% | +3.9% | -32.0% | +3.2% |

### Trend analysis: what changing the KV network does

- **Decode-side contention relief grows with concurrency.** MNNVL KV transfers ride the same
  NVLink fabric (and copy-engine/SM resources) the decode batch computes on; at fat batches
  every prefill->decode handoff steals from token generation. Moving KV to the NICs
  (GPUDirect RDMA) offloads that entirely. First confirmation at mid conc-2048: TPOT dropped
  from ~16-19 ms (MNNVL variants) to 13.75 ms — closing the long-standing +24-41% TPOT gap
  vs InferenceMax to ~3% — while throughput rose to the best value recorded on the point
  (-1.2% vs InferenceMax, vs -2.7% best MNNVL). Expect the effect to shrink toward zero at
  low-latency concurrencies (thin batches, KV traffic sparse) and to be the key question at
  max_tpt (whether dispatch-wave stalls interact with transport).
- **TTFT moves the other way at saturation.** RDMA per-transfer setup adds prefill-side
  latency and the freed decode capacity admits deeper queues; conc-2048 TTFT median rose vs
  MNNVL. At 8k1k this trades a one-time wait for steady-state speed — E2E/interactivity
  judge the net (per-metric tables in the main report).
- **InferenceX never ran this configuration** (their KV rides NVLink in every published
  GB300 result, both stacks) — so RDMA-vs-MNNVL deltas here are new information about the
  platform envelope, not a reproduction check.
- Rows marked *pending* fill in as the sweep lands (mid 512/4096, max_tpt, low-latency 4-64);
  the trend claims above will be re-evaluated against the full matrix.
<!-- END dsr1-network-tables -->

### DSv4 8k1k

| Pareto region | Prefill-Decode topology | Conc | Sources | Req plane | KV transfer | Total Tput (tok/s/GPU) | Input Tput (tok/s/GPU) | Output Tput (tok/s/GPU) | Median E2EL (s) | Median ITL (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| Low latency | 4P/4D | 1 | InferenceMax | NATS | MNNVL (mooncake) | 193.4 | 343.5 | 43.2 | 5.06 | 0.02 |
| | | | GCP bench with GKE (p1) [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsv4-point1) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsv4-sweep/manifests/point1-1p1d-tp4-mtp.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsv4-sweep/manifests/point1-1p1d-tp4-mtp.yaml;tab=live_object) | NATS | MNNVL (mooncake) | 216.0 (+11.7%) | 383.8 (+11.7%) | 48.3 (+11.7%) | 4.33 (-14.5%) | 0.02 (-0.1%) |
| Mid curve | 4P/8D | 256 | InferenceMax | NATS | MNNVL (mooncake) | 5,249.6 | 13,998.7 | 875.1 | 31.79 | 1.06 |
| | | | GCP bench with GKE (p3) [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsv4-p3-rerecord) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsv4-bench-p3-rerecord.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsv4-sweep/manifests/p3.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsv4-sweep/manifests/p3.yaml;tab=live_object) | NATS | MNNVL (mooncake) | 5,265.2 (+0.3%) | 14,040.3 (+0.3%) | 877.7 (+0.3%) | 32.31 (+1.7%) | 1.08 (+1.7%) |
| Mid curve | 4P/16D | 256 | InferenceMax | NATS | MNNVL (mooncake) | 3,055.6 | 13,580.1 | 424.5 | 32.36 | 0.80 |
| | | | GCP bench with GKE (p4) [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsv4-p4) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsv4-bench-p4.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsv4-sweep/manifests/p4.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsv4-sweep/manifests/p4.yaml;tab=live_object) | NATS | MNNVL (mooncake) | 3,055.8 (+0.0%) | 13,581.1 (+0.0%) | 424.5 (+0.0%) | 31.59 (-2.4%) | 0.79 (-0.6%) |
| Low latency | 4P/24D | 8 | InferenceMax | NATS | MNNVL (mooncake) | 327.6 | 2,035.4 | 43.0 | 6.19 | 0.02 |
| | | | GCP bench with GKE (p2) [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsv4-p2) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsv4-bench-p2-RECORD.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsv4-sweep/manifests/p2.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsv4-sweep/manifests/p2.yaml;tab=live_object) | NATS | MNNVL (mooncake) | 388.7 (+18.6%) | 2,414.8 (+18.6%) | 51.0 (+18.6%) | 5.37 (-13.2%) | 0.02 (-0.1%) |
| Low latency | 4P/24D | 32 | InferenceMax | NATS | MNNVL (mooncake) | 954.2 | 5,932.9 | 124.5 | 8.88 | 0.02 |
| | | | GCP bench with GKE (p2) [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsv4-p2) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsv4-bench-p2-RECORD.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsv4-sweep/manifests/p2.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsv4-sweep/manifests/p2.yaml;tab=live_object) | NATS | MNNVL (mooncake) | 1,044.8 (+9.5%) | 6,495.5 (+9.5%) | 136.3 (+9.5%) | 8.32 (-6.3%) | 0.02 (+2.7%) |
| Low latency | 4P/24D | 64 | InferenceMax | NATS | MNNVL (mooncake) | 1,599.4 | 9,953.6 | 207.0 | 10.56 | 0.02 |
| | | | GCP bench with GKE (p2) [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsv4-p2) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsv4-bench-p2-RECORD.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsv4-sweep/manifests/p2.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsv4-sweep/manifests/p2.yaml;tab=live_object) | NATS | MNNVL (mooncake) | 1,727.3 (+8.0%) | 10,750.1 (+8.0%) | 223.5 (+8.0%) | 10.05 (-4.9%) | 0.02 (+1.2%) |
| Mid curve | 8P/8D | 512 | InferenceMax | NATS | MNNVL (mooncake) | 7,294.6 | 12,968.2 | 1,620.9 | 32.08 | 1.32 |
| | | | GCP bench with GKE (p5) [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsv4-p5) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsv4-bench-p5.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsv4-sweep/manifests/p5.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsv4-sweep/manifests/p5.yaml;tab=live_object) | NATS | MNNVL (mooncake) | 7,426.3 (+1.8%) | 13,202.4 (+1.8%) | 1,650.2 (+1.8%) | 32.13 (+0.2%) | 1.36 (+3.3%) |
| Mid curve | 16P/8D | 1024 | InferenceMax | NATS | MNNVL (mooncake) | 9,746.1 | 12,994.7 | 3,249.0 | 32.02 | 1.68 |
| | | | GCP bench with GKE (p6) [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsv4-p6) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsv4-bench-p6.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsv4-sweep/manifests/p6.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsv4-sweep/manifests/p6.yaml;tab=live_object) | NATS | MNNVL (mooncake) | 9,790.0 (+0.5%) | 13,053.2 (+0.5%) | 3,263.6 (+0.5%) | 32.42 (+1.3%) | 1.69 (+0.4%) |
| High concurrency | 24P/8D | 4096 | InferenceMax | NATS | MNNVL (mooncake) | 11,633.9 | 13,788.8 | 5,169.3 | 84.82 | 1.71 |
| | | | GCP bench with GKE (p7) [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsv4-p7) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsv4-bench-p7.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsv4-sweep/manifests/p7.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsv4-sweep/manifests/p7.yaml;tab=live_object) | NATS | MNNVL (mooncake) | 11,593.2 (-0.4%) | 13,740.6 (-0.4%) | 5,151.2 (-0.4%) | 84.29 (-0.6%) | 1.65 (-3.3%) |
| High concurrency | 32P/8D | 8192 | InferenceMax | NATS | MNNVL (mooncake) | 12,233.5 | 13,593.0 | 6,795.2 | 132.31 | 2.21 |
| | | | GCP bench with GKE (p8; tput = 2026-08-07 record, SUSPECT-JIT conservative; latency = 2026-08-01 rerun) [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsv4-p8) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsv4-bench-p8.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dsv4-sweep/manifests/p8.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsv4-sweep/manifests/p8.yaml;tab=live_object) | NATS | MNNVL (mooncake) | 11,844.2 (-3.2%) | 13,161.0 (-3.2%) | 6,577.1 (-3.2%) | 135.03 (+2.1%) | 2.14 (-2.9%) |
Column mapping to the template: Sources → the row-pair label (InferenceMax / GCP bench with
GKE + winning-variant tag + `[logs][bench][cfg][cfg-gcs]` provenance links); Config → the
Pareto-region + topology columns; the five metric columns are identical. Full per-metric
detail (TTFT/TPOT tails, interactivity) lives in the two source reports.

## Recipes / Dynamo Setup

- **Cluster**: `REDACTED-GKE-CLUSTER` (REDACTED-GCP-PROJECT), `a4x-maxgpu-4g-metal` — 4× GB300 per node,
  NVL72 scale-up domain, 8-rail RoCE fabric; sweeps use 5–18 nodes per point (the largest
  topologies — DSR1 mid 24P/48D and max_tpt 40P/32D — occupy all 18 np-2 nodes).
- **Dynamo**: operator-less — etcd + NATS StatefulSets, workers as plain
  Deployments/StatefulSets; DSR1 on ai-dynamo 0.8.1 + sglang v0.5.8.post1-cu130 + NIXL KV
  transfer; DSv4 on dynamo @ `81d0555` + sglang nightly + mooncake KV transfer
  (`MC_FORCE_MNNVL=1`).
- **ISL/OSL 8k/1k sweep**: sa-bench, ISL 8192 / OSL 1024, `random_range_ratio 0.8`,
  `ignore-eos`; InferenceX methodology (2×conc warmup @250 + 10×conc measured @recipe rate;
  DSv4 runs 3× warmup + 5× long-warm discarded + 10× measured, exceeding the default).
- **Measurement hygiene** (what it took to make GKE numbers record-quality): DeepGEMM JIT
  cache persistence (hostPath) + cache-filler pass + JIT/stall assertion stamping each record
  VALID/SUSPECT; rotation-proof SLIM result tail-printing; extract-before-teardown.

## Recipe comparison vs InferenceX — DSR1-FP4 MNNVL, DSR1-FP4 RDMA, DSv4

### What is identical by construction

Both sides run the same models (DSR1 NVFP4 / DSv4-Pro FP4), the same sglang serving engine
with **per-point engine flags vendored from the InferenceX recipes and diffed flag-by-flag**
(the one exception — DSv4 p2's five missing prefill `SGLANG_OPT_*` vars — was caught and
fixed before the record run), the same disaggregated prefill/decode topologies per point,
and the same client methodology (sa-bench, ISL 8192/OSL 1024, `random_range_ratio 0.8`,
2×conc warmup + 10×conc measured). DSv4 p4's **+0.0% exact reproduction** is the
methodology-validation anchor: when everything matches, the platforms measure the same.

### Platform-layer differences (apply to all three arms)

| Dimension | InferenceX | Ours | Perf-relevant? |
|---|---|---|---|
| Orchestration | Bare-metal Slurm (srt-slurm), long-lived nodes | GKE pods (operator-less Dynamo: etcd/NATS StatefulSets + worker Deployments) | Indirectly (admission + JIT below) |
| Admission/queueing | srt-slurm scheduler holds a request queue | GKE Service + Dynamo frontends admit eagerly | **Yes — the systematic low-conc lead** (see analysis) |
| IMEX (NVL72) | Slurm prolog-managed | DRA `ComputeDomain` channels | No measured effect |
| UCX | Custom `/usr/local/ucx` build | NIXL-wheel UCX | **Yes — root of the MNNVL KV gap** (lane misordering under GKE netns until rev4 `UCX_NET_DEVICES`) |
| JIT caches | Warm across runs (persistent nodes) | Cold per pod → hostPath dg-cache + cache-filler pass + JIT/stall assertion (VALID/SUSPECT stamps) | Yes if unguarded (−8% once at DSv4 p3); all finals guarded |
| GPU clocks/power | GB300 NVL72 | Same (a4x-maxgpu-4g-metal) | No |

### Workload-specific recipe deltas

| Arm | KV transfer | Request plane | Engine stack | Notes |
|---|---|---|---|---|
| InferenceX DSR1-FP4 (reference) | MNNVL (NIXL/mooncake over NVLink) | NATS | sglang v0.5.8-class + dynamo | All published GB300 points |
| **Ours: DSR1-FP4 MNNVL** | MNNVL (`cuda_ipc` via NIXL-wheel UCX) | Per-point best of TCP / NATS (A/B'd; finals labeled) | ai-dynamo 0.8.1 + sglang v0.5.8.post1-cu130 | Closest analogue to theirs; same NVLink KV path, different UCX build + netns |
| **Ours: DSR1-FP4 RDMA** | **GPUDirect RDMA (`rc_mlx5`), rev4: `UCX_NET_DEVICES` 8-NIC rail allowlist + GID 5 + rail-aware pairing** | NATS | Same as MNNVL arm — only the KV network differs | **No InferenceX counterpart** — every published point runs KV on NVLink; this arm is new platform-envelope data. 100% RDMA-audited (zero TCP fallback) |
| **Ours: DSv4** | MNNVL (mooncake, `MC_FORCE_MNNVL=1`) | NATS | dynamo @ `81d0555` + sglang nightly (+5 days vs recipe pin — the pinned nightly was deleted from Docker Hub) | MTP spec-decode per recipe; the one-time MTP 3/4 template drift (p5–p8 first runs) was detected by its interactivity signature and re-run |

### Performance analysis: ours vs InferenceX, by arm

**DSR1-FP4 MNNVL (like-for-like NVLink KV).** We lead at low latency (+2.9% to +7.7%) and
past the saturation knee (mid-4096 **+9.4%**), and trailed at the two heaviest KV-traffic
points: mid-2048 (−2.7% best-plane) and max_tpt (**−7.6%**). The leads come from the
admission-path difference — eager admission gives better TTFT/E2E at thin batches and
degrades more gracefully past the knee, where their queue-holding scheduler regresses. The
deficits were initially hypothesized as scheduler/dispatch effects; the RDMA A/B **refuted
that**: they were KV-transport contention — our wheel-UCX `cuda_ipc` MNNVL path competing
with DeepEP all-to-all and decode HBM traffic. Notably, InferenceX achieves ~13.4 ms TPOT
*on* MNNVL KV — a well-tuned bare-metal NVLink path need not pay this tax; ours (GKE netns +
wheel UCX) did.

**DSR1-FP4 RDMA (our extension).** Moving KV to the NICs erased the MNNVL deficits: max_tpt
−7.6% → **−0.6%**, mid-2048 → **−1.2%**, with mid TPOT 13.75 ms matching their bare-metal
13.4. Final DSR1 board (best network per point): **6/8 leading, worst −1.2%**. The network
A/B trend (tables above): RDMA is neutral where the NVLink fabric is idle (thin batches) and
decisively better where DeepEP saturates it (TPOT/ITL −22 to −26% at mid 2048/4096, +7.6%
throughput at max_tpt) — the KV finding of this program: **at heavy expert-parallel decode,
NVLink KV contention is real and avoidable by moving KV to GPUDirect RDMA.**

**DSv4 (MNNVL, MTP).** Ten points: p4 **+0.0%** (exact), p3 +0.3% (after the JIT protocol;
it was −8% unguarded), p1 +11.7%, p2 **+18.6/+9.5/+8.0%** (independently confirmed by a
verification rerun whose conservative floor still leads +15.8/+5.2/+3.2%), p5 +1.8%, p6
+0.5%, p7 −0.4%, p8 **−3.2% conservative floor** (one JIT session inside the measured
window; clean estimate ≈ −2%) with latency at parity. The low-conc leads carry the same
admission-path signature as DSR1; mid/high-conc points sit at ±2%.

### Attribution: which recipe delta explains which gap

| Recipe delta | Observed effect | Evidence |
|---|---|---|
| Admission/queueing (eager vs queued) | **All low-conc leads** (DSR1 ll +2.9–8.2%, DSv4 p1/p2 +8–19%) and the past-knee lead (mid-4096 +9.4%) | Better TTFT/E2E at the same throughput; their curve regresses past the knee, ours doesn't |
| KV path build (wheel-UCX `cuda_ipc` vs their UCX/NVLink) | **The two former DSR1 deficits** (−7.6%, −6.0%) | RDMA A/B closes both to ≤1.2% with identical recipes — transport, not scheduler |
| JIT cold start (pods vs warm nodes) | One-time −8% (DSv4 p3); p8's −3.2% is a floor for the same reason | Assertion protocol timestamps compiles inside measured windows; guarded finals |
| Image nightly +5 days (DSv4) | No measurable effect | p4 +0.0% and p3 +0.3% exactness with the same image |
| Spec-decode template drift (MTP 3/4 vs 1/2) | −17.9% tput / +45% interactivity (p8 first run) | Now a documented drift detector; corrected reruns at parity |

**Net conclusion.** On identical engine recipes, GKE + Dynamo **matches or beats bare-metal
Slurm on 14 of 18 published-comparable points**; the four trailing points sit at −0.4%,
−0.6%, −1.2%, and −3.2% (the last a JIT-conservative floor, ≈ −2% clean). Nothing in the
data indicates a GKE platform tax; the residual deltas are measurement floors and the
(solved) KV-transport build issue. The program's transferable findings: (1) eager admission
is a real serving-curve advantage at low concurrency and past the knee; (2) at heavy EP
decode, KV-over-NVLink contention is avoidable via GPUDirect RDMA — worth knowing for any
NVL72 deployment whose UCX/NVLink path is not perfectly tuned; (3) record-quality GKE
benchmarking requires the JIT-guard protocol (cache + filler pass + in-window assertion).

## Docs

- `benchmark-report-dsr1-8k1k.md`, `benchmark-report-dsv4-8k1k.md` — full per-metric
  comparisons, config-parity sections, per-cell GCS/GitHub provenance links
- `benchmark-report-rdma-kv.md` — RDMA-KV network A/B (in progress)
- `perf-tuning-walkthrough.md` — the 8 tuning case studies behind these numbers
- `comparison-dsv4-vs-inferencex.md` — standalone DSv4 comparison
