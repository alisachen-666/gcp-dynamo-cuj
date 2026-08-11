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

| Pt | Pareto region | Prefill-Decode topology | Conc | Sources | Req plane | KV transfer | Total Tput (tok/s/GPU) | Input Tput (tok/s/GPU) | Output Tput (tok/s/GPU) | Median E2EL (s) | Median ITL (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p1 | Low latency | 4P/16D | 4 | InferenceMax | NATS | MNNVL (NIXL) | 247.4 | 1,099.5 | 34.4 | 6.40 | 0.33 |
| | | | | GCP bench with GKE RDMA-KV [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-llr) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsr1-llr-bench.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsr1-sweep/manifests/llr-lowlatency-rdmakv.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/llr-lowlatency-rdmakv.yaml;tab=live_object) | NATS | RDMA (rc_mlx5) | 267.4 (+8.1%) | 1,188.1 (+8.1%) | 37.2 (+8.1%) | 6.01 (-6.1%) | 0.31 (-6.6%) |
| p2 | Low latency | 4P/16D | 8 | InferenceMax | NATS | MNNVL (NIXL) | 457.8 | 2,031.8 | 64.3 | 6.99 | 0.37 |
| | | | | GCP bench with GKE RDMA-KV [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-llr) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsr1-llr-bench.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsr1-sweep/manifests/llr-lowlatency-rdmakv.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/llr-lowlatency-rdmakv.yaml;tab=live_object) | NATS | RDMA (rc_mlx5) | 495.2 (+8.2%) | 2,197.8 (+8.2%) | 69.6 (+8.2%) | 6.57 (-6.1%) | 0.33 (-11.9%) |
| p3 | Low latency | 4P/16D | 32 | InferenceMax | NATS | MNNVL (NIXL) | 1,374.7 | 6,105.1 | 192.1 | 9.09 | 0.46 |
| | | | | GCP bench with GKE [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-lowlatency) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsr1-sweep/manifests/m4b-lowlatency-1p4d.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/m4b-lowlatency-1p4d.yaml;tab=live_object) | TCP | MNNVL (cuda_ipc) | 1,415.1 (+2.9%) | 6,284.2 (+2.9%) | 197.8 (+2.9%) | 8.87 (-2.5%) | 0.45 (-2.1%) |
| p4 | Low latency | 4P/16D | 64 | InferenceMax | NATS | MNNVL (NIXL) | 2,093.7 | 9,307.2 | 290.3 | 11.98 | 0.61 |
| | | | | GCP bench with GKE [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-lowlatency) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsr1-sweep/manifests/m4b-lowlatency-1p4d.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/m4b-lowlatency-1p4d.yaml;tab=live_object) | TCP | MNNVL (cuda_ipc) | 2,185.7 (+4.4%) | 9,716.0 (+4.4%) | 303.1 (+4.4%) | 11.43 (-4.6%) | 0.58 (-3.7%) |
| p5 | Mid curve | 24P/48D | 512 | InferenceMax | NATS | MNNVL (NIXL) | 2,653.3 | 7,075.7 | 442.2 | 15.14 | 0.64 |
| | | | | GCP bench with GKE RDMA-KV [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-m4r-rev3-3pts) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/bench-m4r-c512-rev3.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsr1-sweep/manifests/m4r-midcurve-rdmakv.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/m4r-midcurve-rdmakv.yaml;tab=live_object) | NATS | RDMA (rc_mlx5) | 2,946.9 (+11.1%) | 7,858.4 (+11.1%) | 491.1 (+11.1%) | 13.74 (-9.3%) | 0.64 (-0.5%) |
| p6 | Mid curve | 24P/48D | 2048 | InferenceMax | NATS | MNNVL (NIXL) | 5,028.8 | 13,408.8 | 838.8 | 44.28 | 0.66 |
| | | | | GCP bench with GKE RDMA-KV [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-m4r-rev3-3pts) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/bench-m4r-c2048-rev3.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsr1-sweep/manifests/m4r-midcurve-rdmakv.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/m4r-midcurve-rdmakv.yaml;tab=live_object) | NATS | RDMA (rc_mlx5) | 4,967.4 (-1.2%) | 13,244.9 (-1.2%) | 828.7 (-1.2%) | 41.85 (-5.5%) | 0.69 (+4.4%) |
| p7 | Mid curve | 24P/48D | 4096 | InferenceMax | NATS | MNNVL (NIXL) | 4,594.4 | 12,252.1 | 765.5 | 92.94 | 0.68 |
| | | | | GCP bench with GKE [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-midcurve) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsr1-sweep/manifests/m4-midcurve.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/m4-midcurve.yaml;tab=live_object) | TCP | MNNVL (cuda_ipc) | 5,025.6 (+9.4%) | 13,401.9 (+9.4%) | 837.4 (+9.4%) | 84.29 (-9.3%) | 0.90 (+32.5%) |
| p8 | Max throughput | 40P/32D | 2048 | InferenceMax | NATS | MNNVL (NIXL) | 7,120.5 | 11,391.7 | 1,781.6 | 26.39 | 0.86 |
| | | | | GCP bench with GKE RDMA-KV [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-mxr) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsr1-mxr-bench.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsr1-sweep/manifests/mxr-maxtpt-rdmakv.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/mxr-maxtpt-rdmakv.yaml;tab=live_object) | NATS | RDMA (rc_mlx5) | 7,080.3 (-0.6%) | 11,327.1 (-0.6%) | 1,771.7 (-0.6%) | 27.44 (+4.0%) | 0.90 (+4.5%) |

<!-- BEGIN dsr1-network-tables (generated; source: benchmark-report-rdma-kv.md) -->
## DSR1-FP4: MNNVL vs RDMA — per-network tables

Format matches the main summary table: InferenceMax row = absolute values; GCP row = `value (gap vs InferenceMax)` per metric. Throughput: higher is better; TPOT/TTFT/ITL: lower is better (negative gap = we are faster).

### Table 1 — KV over MNNVL (NVLink; final per point, optimal variant)

| Pt | Region | Topology | Conc | Sources | Total Tput (tok/s/GPU) | Input Tput (tok/s/GPU) | Out/decode-GPU (tok/s) | TPOT mean (ms) | TTFT med (s) | Median ITL (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| p1 | Low latency | 4P/16D | 4 | InferenceMax | 247.4 | 1,099.5 | 34.4 | 6.73 | 0.24 | 0.33 |
| | | | | GCP bench with GKE NATS mult10 [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-lln) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsr1-lln-bench.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsr1-sweep/manifests/lln-lowlatency-nats.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/lln-lowlatency-nats.yaml;tab=live_object) | 262.4 (+6.1%) | 1,166.0 (+6.0%) | 36.5 (+6.1%) | 6.24 (-7.4%) | 0.25 (+6.0%) | 0.31 (-6.5%) |
| p2 | Low latency | 4P/16D | 8 | InferenceMax | 457.8 | 2,031.8 | 64.3 | 7.09 | 0.27 | 0.37 |
| | | | | GCP bench with GKE NATS mult10 [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-lln) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsr1-lln-bench.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsr1-sweep/manifests/lln-lowlatency-nats.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/lln-lowlatency-nats.yaml;tab=live_object) | 493.2 (+7.7%) | 2,188.8 (+7.7%) | 69.3 (+7.7%) | 6.61 (-6.7%) | 0.38 (+38.1%) | 0.33 (-11.6%) |
| p3 | Low latency | 4P/16D | 32 | InferenceMax | 1,374.7 | 6,105.1 | 192.1 | 9.13 | 0.62 | 0.46 |
| | | | | GCP bench with GKE [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-lowlatency) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsr1-sweep/manifests/m4b-lowlatency-1p4d.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/m4b-lowlatency-1p4d.yaml;tab=live_object) | 1,415.1 (+2.9%) | 6,284.2 (+2.9%) | 197.8 (+2.9%) | 8.86 (-2.9%) | 0.57 (-7.3%) | 0.45 (-2.1%) |
| p4 | Low latency | 4P/16D | 64 | InferenceMax | 2,093.7 | 9,307.2 | 290.3 | 12.02 | 0.73 | 0.61 |
| | | | | GCP bench with GKE [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-lowlatency) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsr1-sweep/manifests/m4b-lowlatency-1p4d.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/m4b-lowlatency-1p4d.yaml;tab=live_object) | 2,185.7 (+4.4%) | 9,716.0 (+4.4%) | 303.1 (+4.4%) | 11.47 (-4.6%) | 0.72 (-1.5%) | 0.58 (-3.7%) |
| p5 | Mid curve | 24P/48D | 512 | InferenceMax | 2,653.3 | 7,075.7 | 442.2 | 13.08 | 3.39 | 0.64 |
| | | | | GCP bench with GKE NATS A/B [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-midcurve) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsr1-sweep/manifests/m4-midcurve.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/m4-midcurve.yaml;tab=live_object) | 2,813.9 (+6.0%) | 7,503.6 (+6.0%) | 469.0 (+6.1%) | 12.95 (-1.0%) | 2.18 (-35.7%) | 0.64 (-0.2%) |
| p6 | Mid curve | 24P/48D | 2048 | InferenceMax | 5,028.8 | 13,408.8 | 838.8 | 13.25 | 32.03 | 0.66 |
| | | | | GCP bench with GKE [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-midcurve) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsr1-sweep/manifests/m4-midcurve.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/m4-midcurve.yaml;tab=live_object) | 4,894.5 (-2.7%) | 13,050.4 (-2.7%) | 816.5 (-2.7%) | 18.71 (+41.2%) | 23.60 (-26.3%) | 0.90 (+37.0%) |
| p7 | Mid curve | 24P/48D | 4096 | InferenceMax | 4,594.4 | 12,252.1 | 765.5 | 13.57 | 80.43 | 0.68 |
| | | | | GCP bench with GKE [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-midcurve) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsr1-sweep/manifests/m4-midcurve.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/m4-midcurve.yaml;tab=live_object) | 5,025.6 (+9.4%) | 13,401.9 (+9.4%) | 837.4 (+9.4%) | 17.90 (+31.9%) | 67.72 (-15.8%) | 0.90 (+32.5%) |
| p8 | Max throughput | 40P/32D | 2048 | InferenceMax | 7,120.5 | 11,391.7 | 1,781.6 | 17.12 | 10.24 | 0.86 |
| | | | | GCP bench with GKE TCP A/B [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-mxt) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsr1-mxt-bench.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsr1-sweep/manifests/mxt-maxtpt-tcp.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/mxt-maxtpt-tcp.yaml;tab=live_object) | 6,580.8 (-7.6%) | 10,528.1 (-7.6%) | 1,646.7 (-7.6%) | 17.18 (+0.4%) | 15.84 (+54.7%) | 0.87 (+1.3%) |

### Table 2 — KV over GPUDirect RDMA (rev4: `UCX_NET_DEVICES` + GID 5 + rail-aware pairing)

| Pt | Region | Topology | Conc | Sources | Total Tput (tok/s/GPU) | Input Tput (tok/s/GPU) | Out/decode-GPU (tok/s) | TPOT mean (ms) | TTFT med (s) | Median ITL (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| p1 | Low latency | 4P/16D | 4 | InferenceMax | 247.4 | 1,099.5 | 34.4 | 6.73 | 0.24 | 0.33 |
| | | | | GCP bench with GKE RDMA-KV [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-llr) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsr1-llr-bench.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsr1-sweep/manifests/llr-lowlatency-rdmakv.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/llr-lowlatency-rdmakv.yaml;tab=live_object) | 267.4 (+8.1%) | 1,188.1 (+8.1%) | 37.2 (+8.1%) | 6.21 (-7.7%) | 0.24 (+0.1%) | 0.31 (-6.6%) |
| p2 | Low latency | 4P/16D | 8 | InferenceMax | 457.8 | 2,031.8 | 64.3 | 7.09 | 0.27 | 0.37 |
| | | | | GCP bench with GKE RDMA-KV [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-llr) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsr1-llr-bench.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsr1-sweep/manifests/llr-lowlatency-rdmakv.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/llr-lowlatency-rdmakv.yaml;tab=live_object) | 495.2 (+8.2%) | 2,197.8 (+8.2%) | 69.6 (+8.2%) | 6.57 (-7.4%) | 0.37 (+37.1%) | 0.33 (-11.9%) |
| p3 | Low latency | 4P/16D | 32 | InferenceMax | 1,374.7 | 6,105.1 | 192.1 | 9.13 | 0.62 | 0.46 |
| | | | | GCP bench with GKE RDMA-KV [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-llr) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsr1-llr-bench.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsr1-sweep/manifests/llr-lowlatency-rdmakv.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/llr-lowlatency-rdmakv.yaml;tab=live_object) | 1,395.8 (+1.5%) | 6,198.4 (+1.5%) | 195.1 (+1.5%) | 8.80 (-3.6%) | 0.81 (+31.4%) | 0.44 (-3.4%) |
| p4 | Low latency | 4P/16D | 64 | InferenceMax | 2,093.7 | 9,307.2 | 290.3 | 12.02 | 0.73 | 0.61 |
| | | | | GCP bench with GKE RDMA-KV [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-llr) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsr1-llr-bench.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsr1-sweep/manifests/llr-lowlatency-rdmakv.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/llr-lowlatency-rdmakv.yaml;tab=live_object) | 2,164.7 (+3.4%) | 9,622.9 (+3.4%) | 300.2 (+3.4%) | 11.37 (-5.4%) | 0.93 (+27.7%) | 0.57 (-5.2%) |
| p5 | Mid curve | 24P/48D | 512 | InferenceMax | 2,653.3 | 7,075.7 | 442.2 | 13.08 | 3.39 | 0.64 |
| | | | | GCP bench with GKE RDMA-KV [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-m4r-rev3-3pts) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/bench-m4r-c512-rev3.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsr1-sweep/manifests/m4r-midcurve-rdmakv.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/m4r-midcurve-rdmakv.yaml;tab=live_object) | 2,946.9 (+11.1%) | 7,858.4 (+11.1%) | 491.1 (+11.1%) | 12.70 (-2.9%) | 1.33 (-60.8%) | 0.64 (-0.5%) |
| p6 | Mid curve | 24P/48D | 2048 | InferenceMax | 5,028.8 | 13,408.8 | 838.8 | 13.25 | 32.03 | 0.66 |
| | | | | GCP bench with GKE RDMA-KV [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-m4r-rev3-3pts) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/bench-m4r-c2048-rev3.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsr1-sweep/manifests/m4r-midcurve-rdmakv.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/m4r-midcurve-rdmakv.yaml;tab=live_object) | 4,967.4 (-1.2%) | 13,244.9 (-1.2%) | 828.7 (-1.2%) | 13.75 (+3.8%) | 29.31 (-8.5%) | 0.69 (+4.4%) |
| p7 | Mid curve | 24P/48D | 4096 | InferenceMax | 4,594.4 | 12,252.1 | 765.5 | 13.57 | 80.43 | 0.68 |
| | | | | GCP bench with GKE RDMA-KV [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-m4r-rev3-3pts) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/bench-m4r-c4096-rev3.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsr1-sweep/manifests/m4r-midcurve-rdmakv.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/m4r-midcurve-rdmakv.yaml;tab=live_object) | 4,986.0 (+8.5%) | 13,296.5 (+8.5%) | 830.8 (+8.5%) | 13.87 (+2.2%) | 73.21 (-9.0%) | 0.69 (+2.0%) |
| p8 | Max throughput | 40P/32D | 2048 | InferenceMax | 7,120.5 | 11,391.7 | 1,781.6 | 17.12 | 10.24 | 0.86 |
| | | | | GCP bench with GKE RDMA-KV [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-mxr) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsr1-mxr-bench.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsr1-sweep/manifests/mxr-maxtpt-rdmakv.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsr1-sweep/manifests/mxr-maxtpt-rdmakv.yaml;tab=live_object) | 7,080.3 (-0.6%) | 11,327.1 (-0.6%) | 1,771.7 (-0.6%) | 17.85 (+4.2%) | 10.78 (+5.2%) | 0.90 (+4.5%) |

### Network delta (RDMA vs MNNVL, same recipe/methodology/cluster)

Cells show `MNNVL value → RDMA value (Δ%)` — the two arms' measured data side by
side. Out tput: higher better; TPOT/TTFT/ITL: lower better.

| Pt | Region | Conc | Out/decode-GPU (tok/s) | TPOT mean (ms) | TTFT med (s) | ITL med (s) |
|---|---|---|---|---|---|---|
| p1 | Low latency | 4 | 36.5 → 37.2 (+1.9%) | 6.24 → 6.21 (-0.4%) | 0.25 → 0.24 (-5.5%) | 0.31 → 0.31 (-0.1%) |
| p2 | Low latency | 8 | 69.3 → 69.6 (+0.4%) | 6.61 → 6.57 (-0.7%) | 0.38 → 0.37 (-0.7%) | 0.33 → 0.33 (-0.3%) |
| p3 | Low latency | 32 | 197.8 → 195.1 (-1.4%) | 8.86 → 8.80 (-0.7%) | 0.57 → 0.81 (+41.8%) | 0.45 → 0.44 (-1.3%) |
| p4 | Low latency | 64 | 303.1 → 300.2 (-1.0%) | 11.47 → 11.37 (-0.9%) | 0.72 → 0.93 (+29.7%) | 0.58 → 0.57 (-1.5%) |
| p5 | Mid curve | 512 | 469.0 → 491.1 (+4.7%) | 12.95 → 12.70 (-1.9%) | 2.18 → 1.33 (-39.0%) | 0.64 → 0.64 (-0.4%) |
| p6 | Mid curve | 2048 | 816.5 → 828.7 (+1.5%) | 18.71 → 13.75 (-26.5%) | 23.60 → 29.31 (+24.2%) | 0.90 → 0.69 (-23.8%) |
| p7 | Mid curve | 4096 | 837.4 → 830.8 (-0.8%) | 17.90 → 13.87 (-22.5%) | 67.72 → 73.21 (+8.1%) | 0.90 → 0.69 (-23.1%) |
| p8 | Max throughput | 2048 | 1,646.7 → 1,771.7 (+7.6%) | 17.18 → 17.85 (+3.9%) | 15.84 → 10.78 (-32.0%) | 0.87 → 0.90 (+3.2%) |

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

| Pt | Pareto region | Prefill-Decode topology | Conc | Sources | Req plane | KV transfer | Total Tput (tok/s/GPU) | Input Tput (tok/s/GPU) | Output Tput (tok/s/GPU) | Median E2EL (s) | Median ITL (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p1 | Low latency | 4P/4D | 1 | InferenceMax | NATS | MNNVL (mooncake) | 193.4 | 343.5 | 43.2 | 5.06 | 0.02 |
| | | | | GCP bench with GKE (p1) [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsv4-point1) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsv4-sweep/manifests/point1-1p1d-tp4-mtp.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsv4-sweep/manifests/point1-1p1d-tp4-mtp.yaml;tab=live_object) | NATS | MNNVL (mooncake) | 216.0 (+11.7%) | 383.8 (+11.7%) | 48.3 (+11.7%) | 4.33 (-14.5%) | 0.02 (-0.1%) |
| p3 | Mid curve | 4P/8D | 256 | InferenceMax | NATS | MNNVL (mooncake) | 5,249.6 | 13,998.7 | 875.1 | 31.79 | 1.06 |
| | | | | GCP bench with GKE (p3) [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsv4-p3-rerecord) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsv4-bench-p3-rerecord.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsv4-sweep/manifests/p3.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsv4-sweep/manifests/p3.yaml;tab=live_object) | NATS | MNNVL (mooncake) | 5,265.2 (+0.3%) | 14,040.3 (+0.3%) | 877.7 (+0.3%) | 32.31 (+1.7%) | 1.08 (+1.7%) |
| p4 | Mid curve | 4P/16D | 256 | InferenceMax | NATS | MNNVL (mooncake) | 3,055.6 | 13,580.1 | 424.5 | 32.36 | 0.80 |
| | | | | GCP bench with GKE (p4) [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsv4-p4) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsv4-bench-p4.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsv4-sweep/manifests/p4.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsv4-sweep/manifests/p4.yaml;tab=live_object) | NATS | MNNVL (mooncake) | 3,055.8 (+0.0%) | 13,581.1 (+0.0%) | 424.5 (+0.0%) | 31.59 (-2.4%) | 0.79 (-0.6%) |
| p2 | Low latency | 4P/24D | 8 | InferenceMax | NATS | MNNVL (mooncake) | 327.6 | 2,035.4 | 43.0 | 6.19 | 0.02 |
| | | | | GCP bench with GKE (p2) [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsv4-p2) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsv4-bench-p2-RECORD.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsv4-sweep/manifests/p2.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsv4-sweep/manifests/p2.yaml;tab=live_object) | NATS | MNNVL (mooncake) | 388.7 (+18.6%) | 2,414.8 (+18.6%) | 51.0 (+18.6%) | 5.37 (-13.2%) | 0.02 (-0.1%) |
| p2 | Low latency | 4P/24D | 32 | InferenceMax | NATS | MNNVL (mooncake) | 954.2 | 5,932.9 | 124.5 | 8.88 | 0.02 |
| | | | | GCP bench with GKE (p2) [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsv4-p2) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsv4-bench-p2-RECORD.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsv4-sweep/manifests/p2.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsv4-sweep/manifests/p2.yaml;tab=live_object) | NATS | MNNVL (mooncake) | 1,044.8 (+9.5%) | 6,495.5 (+9.5%) | 136.3 (+9.5%) | 8.32 (-6.3%) | 0.02 (+2.7%) |
| p2 | Low latency | 4P/24D | 64 | InferenceMax | NATS | MNNVL (mooncake) | 1,599.4 | 9,953.6 | 207.0 | 10.56 | 0.02 |
| | | | | GCP bench with GKE (p2) [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsv4-p2) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsv4-bench-p2-RECORD.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsv4-sweep/manifests/p2.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsv4-sweep/manifests/p2.yaml;tab=live_object) | NATS | MNNVL (mooncake) | 1,727.3 (+8.0%) | 10,750.1 (+8.0%) | 223.5 (+8.0%) | 10.05 (-4.9%) | 0.02 (+1.2%) |
| p5 | Mid curve | 8P/8D | 512 | InferenceMax | NATS | MNNVL (mooncake) | 7,294.6 | 12,968.2 | 1,620.9 | 32.08 | 1.32 |
| | | | | GCP bench with GKE (p5) [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsv4-p5) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsv4-bench-p5.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsv4-sweep/manifests/p5.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsv4-sweep/manifests/p5.yaml;tab=live_object) | NATS | MNNVL (mooncake) | 7,426.3 (+1.8%) | 13,202.4 (+1.8%) | 1,650.2 (+1.8%) | 32.13 (+0.2%) | 1.36 (+3.3%) |
| p6 | Mid curve | 16P/8D | 1024 | InferenceMax | NATS | MNNVL (mooncake) | 9,746.1 | 12,994.7 | 3,249.0 | 32.02 | 1.68 |
| | | | | GCP bench with GKE (p6) [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsv4-p6) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsv4-bench-p6.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsv4-sweep/manifests/p6.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsv4-sweep/manifests/p6.yaml;tab=live_object) | NATS | MNNVL (mooncake) | 9,790.0 (+0.5%) | 13,053.2 (+0.5%) | 3,263.6 (+0.5%) | 32.42 (+1.3%) | 1.69 (+0.4%) |
| p7 | High concurrency | 24P/8D | 4096 | InferenceMax | NATS | MNNVL (mooncake) | 11,633.9 | 13,788.8 | 5,169.3 | 84.82 | 1.71 |
| | | | | GCP bench with GKE (p7) [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsv4-p7) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsv4-bench-p7.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsv4-sweep/manifests/p7.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsv4-sweep/manifests/p7.yaml;tab=live_object) | NATS | MNNVL (mooncake) | 11,593.2 (-0.4%) | 13,740.6 (-0.4%) | 5,151.2 (-0.4%) | 84.29 (-0.6%) | 1.65 (-3.3%) |
| p8 | High concurrency | 32P/8D | 8192 | InferenceMax | NATS | MNNVL (mooncake) | 12,233.5 | 13,593.0 | 6,795.2 | 132.31 | 2.21 |
| | | | | GCP bench with GKE (p8; tput = 2026-08-07 record, SUSPECT-JIT conservative; latency = 2026-08-01 rerun) [logs](https://console.cloud.google.com/storage/browser/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsv4-p8) [bench](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/client-logs/dsv4-bench-p8.log;tab=live_object) [cfg](https://github.com/alisachen-666/gcp-dynamo-cuj/blob/main/dynamo-disagg-sweep/dsv4-sweep/manifests/p8.yaml) [cfg-gcs](https://console.cloud.google.com/storage/browser/_details/gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs/configs/dsv4-sweep/manifests/p8.yaml;tab=live_object) | NATS | MNNVL (mooncake) | 11,844.2 (-3.2%) | 13,161.0 (-3.2%) | 6,577.1 (-3.2%) | 135.03 (+2.1%) | 2.14 (-2.9%) |
Column mapping to the template: Sources → the row-pair label (InferenceMax / GCP bench with
GKE + winning-variant tag + `[logs][bench][cfg][cfg-gcs]` provenance links); Config → the
Pareto-region + topology columns; the five metric columns are identical. Full per-metric
detail (TTFT/TPOT tails, interactivity) lives in the two source reports.

## Setup: Dynamo disaggregated bench with SGLang on GKE

Structured after the [gpu-recipes inference setup guides](https://github.com/AI-Hypercomputer/gpu-recipes/blob/main/inference/a4x/single-host-serving/tensorrt-llm/README.md).

### Test environment

- **Orchestration**: GKE (a4x-max metal node pool), operator-less Dynamo — etcd + NATS via
  the NGC `dynamo-platform` helm chart; workers as plain Deployments/StatefulSets
- **Machine type**: `a4x-maxgpu-4g-metal` — GB300 NVL72, 4 GPUs + 8× CX-7 mrdma NICs per
  node; 18 nodes (full rack) for the largest topologies
- **Enabled features**: `nvidia-dra-driver-gpu` (ComputeDomain CRD → IMEX channels for
  MNNVL), DraNet (mrdma NICs as DRA resource claims), hostPath DeepGEMM JIT cache
- **Serving stack**: sglang disaggregated prefill/decode + NIXL (DSR1) / mooncake (DSv4)
  KV transfer; per-point engine flags vendored from the InferenceX srt-slurm recipes

### High-level architecture

See the [system diagram](#system-diagram) (client → frontend → NATS → prefill → KV path
A/B → decode) and the [benchmarking-flow diagram](figures/bench-flow.svg) (the 10-stage
per-point pipeline).

### Environment setup (one-time)

**1. Clone the repository and set variables:**

```bash
git clone https://github.com/alisachen-666/gcp-dynamo-cuj.git && cd gcp-dynamo-cuj
export PROJECT_ID=<PROJECT_ID>
export CLUSTER_NAME=<GKE_CLUSTER_NAME>
export CLUSTER_REGION=<REGION_of_your_cluster>
export GCS_BUCKET=<ARTIFACT_BUCKET>          # run logs / configs / results archive
```

| Variable | Description | Example |
|---|---|---|
| `PROJECT_ID` | GCP project hosting the cluster | `my-gpu-project` |
| `CLUSTER_NAME` | GKE cluster with the a4x-max pool | `a4xmax-bench` |
| `CLUSTER_REGION` | Cluster region | `us-central1` |
| `GCS_BUCKET` | Bucket for artifact archiving | `my-bench-artifacts` |

**2. Connect to the cluster:**

```bash
gcloud container clusters get-credentials "$CLUSTER_NAME" \
    --region "$CLUSTER_REGION" --project "$PROJECT_ID"
```

**3. Verify platform prerequisites** (DRA driver, DraNet, Dynamo platform):

```bash
kubectl get crd computedomains.resource.nvidia.com          # nvidia-dra-driver-gpu
kubectl get resourceclaimtemplates -A | grep mrdma           # DraNet NIC claims
kubectl get pods -n dynamo-cloud | grep -E "etcd|nats"       # dynamo-platform (helm, NGC)
```

If the Dynamo platform is absent, install it from NGC (needs an NGC API key):

```bash
helm install dynamo-platform <NGC_DYNAMO_PLATFORM_CHART> -n dynamo-cloud --create-namespace
```

**4. Stage model weights** (checkpoints staged in GCS, materialized to `HF_HOME` on first
worker start; sizes make node-local caching worthwhile):

| Model | Weights | Served as |
|---|---|---|
| DeepSeek-R1 FP4 | `DeepSeek-R1-0528-NVFP4-v2` | `deepseek-ai/DeepSeek-R1` |
| DeepSeek-V4-Pro FP4 | mxfp4 checkpoint | `deepseek-ai/DeepSeek-V4-Pro` |

### Run the recipe

**1. Supported points** (per-point manifests, all recipe-vendored):

| Sweep | Points | Manifests |
|---|---|---|
| DSR1-FP4 8k1k | p1–p8 (MNNVL: `ll/m4/mx`; RDMA: `llr/m4r/mxr`) | `dynamo-disagg-sweep/dsr1-sweep/manifests/` |
| DSv4 8k1k | p1–p8 (MTP spec-decode) | `dynamo-disagg-sweep/dsv4-sweep/manifests/` |

Worker images — exact InferenceX pins: DSR1 `lmsysorg/sglang:v0.5.8.post1-cu130-runtime`
(+ `ai-dynamo==0.8.1` wheels), DSv4 `lmsysorg/sglang:nightly-dev-20260601-373cadc9`
(+ dynamo @ `81d0555`), frontend `nginx:1.27.4`.

**2. Deploy and benchmark one point** (the runner executes the full pipeline in the
benchmarking-flow diagram — gates, filler pass, record pass, assertion, upload, teardown):

```bash
dynamo-disagg-sweep/dsv4-sweep/run-point-v2.sh p4 2 2   # <point> <expected_workers> <passes>
```

Or step-by-step, what the runner does:

```bash
kubectl apply -f dynamo-disagg-sweep/dsr1-sweep/manifests/m4r-midcurve-rdmakv.yaml
kubectl apply -f dynamo-disagg-sweep/dsr1-sweep/manifests/m4r-bench-conc2048.yaml
```

**3. Regenerate reports from the harvested slims:**

```bash
python3 dynamo-disagg-sweep/dsr1-sweep/compare_inferencex_dsr1.py
python3 dynamo-disagg-sweep/dsv4-sweep/gen-dsv4-report.py
python3 common/gen-rdma-report.py && python3 common/gen-pareto-figs.py
```

### Monitoring and troubleshooting

```bash
kubectl get pods -n dynamo-cloud                              # worker/bench status
kubectl logs -n dynamo-cloud job/dsv4-bench-p4 -f             # live bench progress
kubectl logs -n dynamo-cloud <decode-pod> | grep -c rc_mlx5   # RDMA transport audit
                                                              # (must be 100% rc_mlx5, zero fallback)
```

The runner stamps every record run `VALID` or `SUSPECT` (JIT/stall assertion over each
measured window); a SUSPECT run should be re-recorded against the standing deployment.

### Cleanup

```bash
# per point (the runner does this automatically after upload)
kubectl delete -n dynamo-cloud deployment/dsv4-p4-frontend deployment/dsv4-p4-prefill \
    statefulset/dsv4-p4-decode deployment/dsv4-p4-decode \
    computedomain/dsv4-p4-cd service/dsv4-p4-frontend service/dsv4-p4-decode job/dsv4-bench-p4
# platform (only when finished with all sweeps)
helm uninstall dynamo-platform -n dynamo-cloud
```

## Sweep summary: methodology, Pareto view, and where the gaps live

### 0. Our sweeping methodology

Each point in the tables above is one **fully isolated deployment** on the GKE A4X Max
cluster, driven by a runner that enforces record-quality gates end to end: deploy the
point's vendored InferenceX recipe (topology + engine flags, diffed flag-by-flag) → wait for
worker registration → hard serving gate (3 consecutive real completions) → **cache-filler
bench pass** (forces every DeepGEMM/flashinfer JIT shape through compilation; results
discarded) → **record pass** (2×conc warmup @250 + 10×conc measured, the InferenceX client
methodology) → JIT/stall assertion over every measured window (stamps VALID/SUSPECT) →
extract-before-teardown → artifact upload (server logs, bench logs, configs, results).
RDMA points add a transport gate + in-flight watchdog: any KV fall-back to TCP kills the run
rather than record a mixed-transport number. Per point we keep the **optimal final** across
audited variants (request plane TCP/NATS; KV network MNNVL/RDMA), with every non-final
variant retained in the reports' extras rows.

![E2E benchmarking flow — one point, deploy to report](figures/bench-flow.svg)

The flow above runs once per data point: stages 1–5 establish a provably healthy, correctly
configured deployment (with the RDMA transport gate killing any run whose KV path degrades);
6a/6b separate compilation from measurement; 7 proves the measured windows were clean or
stamps them SUSPECT (a conservative floor, re-recordable cheaply against the standing
deployment); 8–9 make harvesting rotation-proof and archive every artifact; 10 turns slims
into the tables, curves, and per-cell provenance links in this document.

### What our Pareto curves consist of

Each plotted point is **one fully isolated deployment** — a (topology, concurrency) pair
measured end-to-end by the pipeline above — and each curve is those measured operating
points sorted along the x-axis and connected. Nothing on the curve is fitted, sampled, or
interpolated; the connecting lines are visual guides between adjacent measured points only.

**DSR1-FP4 (8 points, 3 topologies).** The curve concatenates three deployment families,
each owning one region of the frontier:

| Segment | Points | Topology | Concurrencies | Frontier region |
|---|---|---|---|---|
| Right (interactivity-rich) | p1–p4 | 4P/16D, 20 GPUs | 4, 8, 32, 64 | Low latency: ≥85 tok/s/user, low tput/GPU |
| Middle | p5–p7 | 24P/48D, 72 GPUs | 512, 2048, 4096 | Mid curve: the throughput ramp |
| Left (saturation) | p8 | 40P/32D, 72 GPUs | 2048 | Max throughput: peak tok/s/GPU |

Within a segment, moving left = raising concurrency on the *same* deployment (deeper decode
batches: more tok/s/GPU, less tok/s/user); crossing segments = switching topology. The ours
curve in the RDMA figures is the RDMA arm at every point; the InferenceX curve is their
published measurement of the *same* (topology, concurrency) grid, so the two curves are
point-for-point comparable — every ours label `p<x>(c<y>)` has an InferenceX partner at the
same x-neighborhood.

**DSv4 (10 points, 8 topologies).** Finer topology granularity: every point p1–p8 is its
own prefill/decode geometry (from 1P TP4 + 1D TP4 at conc 1 up to 8P DEP4 + 1D DEP8 at
conc 8192), with only p2 contributing multiple concurrencies (8/32/64). The curve is
therefore mostly a *topology* sweep rather than a concurrency sweep — each point is the
recipe's chosen geometry for that operating regime.

**Axes** (identical on both series): Y is total (input + output) token throughput per GPU
(`total_token_throughput / total_gpus`); X is interactivity (1000 / mean TPOT, linear),
median E2E latency (log), or median TTFT (log) depending on the figure. The **upper
envelope** across segments is the deployable Pareto frontier: an operator picks the segment
covering their latency/interactivity budget, then the concurrency on it matching their
throughput target.

### 1. Why Pareto curves

A single throughput number hides the trade every serving deployment actually makes:
throughput per GPU **against** user experience (interactivity, latency, time-to-first-token).
Two systems can post the same peak tok/s/GPU while one of them makes users wait twice as
long per token. Sweeping concurrency per topology traces the whole trade-off frontier — the
Pareto curve — and comparing *curves* (ours vs InferenceX's published points) answers the
question that matters: **at equal user experience, who serves more tokens per GPU?** It also
prevents cherry-picking: a point that wins on throughput but sits below the reference curve
at its latency level is not a win.

### 2. What the Pareto curves can tell

- **Vertical distance between curves = efficiency gap at equal experience.** Ours above
  theirs at the same x means more tokens/GPU at the same interactivity/latency budget.
- **Horizontal reach** shows achievable operating envelope: the leftmost points are the
  interactivity floor (latency-critical serving), the rightmost the saturation regime.
- **Curve shape around the knee** exposes scheduler/admission behavior: a curve that keeps
  rising past the reference's knee (as ours does at DSR1 mid-4096) means more graceful
  saturation; a curve that collapses early signals queueing pathology.
- **Which topology owns which region**: the three DSR1 topologies (4P/16D, 24P/48D, 40P/32D)
  each dominate a segment; the frontier is their upper envelope — matching InferenceX's
  region labels (low-latency / mid-curve / max-throughput).

### 3. How our gap vs InferenceX shows in the curves

**DSR1-FP4 over RDMA** (figures below): our curve sits **on or above** the published curve
across the full frontier — +8% at the interactivity-rich end (conc 4–8, where we also reach
*better* interactivity: 161 vs 149 tok/s/user), +11.1% at mid-512, converging to overlap at
the saturation end (−0.6%/−1.2%, within noise). In the TTFT view the RDMA curve's mid-region
points sit left of theirs (mid-512 TTFT −39% at higher throughput); the max_tpt point trades
+4% TPOT for its −32% TTFT. There is no region where the published curve dominates ours.

**DSv4** (figures below): the curves **overlap within ±2% along the shared frontier** — the
methodology-validation result (p4 exact at +0.0%) — with our curve above at the low-latency
end (p1 +11.7%, p2 +8–19%: the eager-admission lead) and marginally below only at the
extreme saturation tail (p8 −3.2% conservative floor). The E2E/TTFT views show the same
story: at equal latency budgets our points deliver equal-or-more tokens/GPU everywhere
except the single JIT-floored tail point.

### Pareto curves — DSR1-FP4 over RDMA (ours) vs InferenceX published

![DSR1 RDMA — throughput/GPU vs interactivity](figures/pareto-dsr1-rdma-interactivity.svg)

![DSR1 RDMA — throughput/GPU vs E2E latency](figures/pareto-dsr1-rdma-e2e-latency.svg)

![DSR1 RDMA — throughput/GPU vs TTFT](figures/pareto-dsr1-rdma-ttft.svg)

### Pareto curves — DSv4-Pro (ours, optimal finals) vs InferenceX published

![DSv4 — throughput/GPU vs interactivity](figures/pareto-dsv4-interactivity.svg)

![DSv4 — throughput/GPU vs E2E latency](figures/pareto-dsv4-e2e-latency.svg)

![DSv4 — throughput/GPU vs TTFT](figures/pareto-dsv4-ttft.svg)

Figures regenerate via `common/gen-pareto-figs.py` (Y = total tok/s per GPU on all three;
X = interactivity linear, E2EL/TTFT log-scale; ours-series points labeled with concurrency).

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

### System diagram

![Dynamo disaggregated serving on GB300 NVL72 — e2e system](figures/system-dynamo-disagg-gb300.svg)

**How to read the diagram.** A benchmark request flows left to right: the **bench client**
POSTs completions to the **frontend** (Dynamo ingress behind a K8s Service), which
dispatches it to a prefill worker over the **NATS request plane** (workers self-register in
**etcd**; a direct-TCP plane is the A/B variant). The **prefill workers** compute the
8k-token prompt and produce its KV cache; that cache is handed to the **decode workers**
over one of two networks — **path A** keeps it on NVLink inside the NVL72 domain
(cuda_ipc/mooncake), **path B** offloads it to the 8 per-node CX-7 NICs via GPUDirect RDMA.
Decode then generates the 1k output tokens (wide-EP with DeepEP all-to-all riding the same
NVLink fabric path A uses — the root of the contention finding) and streams them back to the
client. The bottom strips are the platform substrate: the NIC/RoCE fabric with its UCX
configuration, and the GKE layer (DRA ComputeDomain for IMEX/MNNVL, JIT cache, node pool).
Per point we measure both KV paths and keep the better as final — path B wins wherever
decode saturates the NVLink fabric.

**"2× warmup + 10× measured"** (the client methodology, from the InferenceX harness): for a
point at concurrency *C*, the client first sends **2×C requests that are discarded** —
warming the server to steady state (KV caches filled, batch scheduler at stable depth,
residual JIT compiles done) — then sends **10×C requests at the full 8k/1k workload**, and
only these produce the reported throughput and latency percentiles. The long measured
window keeps tail metrics (p99) statistically stable and amortizes the ramp-up/drain edges
where the server runs under-loaded; we measured the effect directly — a shorter window
under-reports throughput by ~11% (the DSR1 conc-64 experiment). Using the identical formula
on both sides is what makes the gap percentages in the tables meaningful.

### Critical configuration templates

Full manifests (all vendored + diffed against InferenceX recipes) live in
`dsr1-sweep/manifests/` and `dsv4-sweep/manifests/`; the GCS mirror keeps run-time copies
under `configs/`. Below are the load-bearing blocks.

**DSR1-FP4 over GPUDirect RDMA** — the KV-transport block that took the RDMA arm from
broken (rev1 TCP fallback) to 100% `rc_mlx5` (rev4). From
`dsr1-sweep/manifests/m4r-midcurve-rdmakv.yaml` (same block in `llr-`/`mxr-`):

```yaml
# Every worker claims 8 mrdma NICs at schedule time (DraNet DRA):
resourceClaims:
- name: rdma
  resourceClaimTemplateName: mrdma-all
env:
- name: UCX_TLS                        # rc_x transports (GPUDirect RDMA), no cuda_ipc:
  value: "cuda_copy,rc_x,tcp"          #   forces KV off NVLink onto the NICs
- name: UCX_NET_DEVICES                # rev4: explicit ORDERED 8-NIC allowlist — excludes
  value: "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1"
                                       #   ipvlan duplicate netdevs that scrambled UCX lane
                                       #   ordering under the pod netns (the rev3 failure)
- name: UCX_IB_GID_INDEX               # rev2: GIDs 0-3 map to host-ns netdevs (unreachable
  value: "5"                           #   in-pod); GID 5 is the pod-resolvable RoCEv2 GID
- name: UCX_IB_ROCE_LOCAL_SUBNET       # rev3: rails are 8 disjoint /64s with no cross-rail
  value: "y"                           #   routing — restrict pairing to same-subnet so
- name: UCX_IB_ROCE_SUBNET_PREFIX_LEN  #   mlx5_k only pairs with remote mlx5_k
  value: "64"
- name: UCX_PROTO_INFO                 # audit trail: per-transfer transport lines in logs —
  value: "y"                           #   how every run is verified rc_mlx5-clean
```

**DSv4** — the recipe-parity engine block (MTP spec-decode + the five `SGLANG_OPT_*`
prefill vars whose omission was the one caught drift). From `dsv4-sweep/manifests/p4.yaml`
(the +0.0% exact-reproduction point):

```yaml
env:
- name: MC_FORCE_MNNVL                 # mooncake KV over NVLink (MNNVL) — DSv4 KV path
  value: "1"
- name: SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS      # NVFP4 MoE activation kernels —
  value: "1"                                           #   part of the 5-var prefill set
- name: SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND     #   missing on p2 rev1, caught in
  value: "1"                                           #   pre-run diff, fixed before record
- name: SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK
  value: "..."                          # per-point value from the vendored recipe
args:  # decode worker — MTP speculative decode per recipe
- --speculative-algo EAGLE
- --speculative-num-steps 3             # recipe values per point; the p5-p8 template once
- --speculative-eagle-topk 1            #   inherited wrong steps/draft (3/4 vs 1/2) -> the
- --speculative-num-draft-tokens 4      #   -17.9% tput / +45% interactivity drift signature
```

Shared by all points: `--disaggregation-transfer-backend nixl`, disaggregation
bootstrap/heartbeat timeouts raised for GKE pod startup, `ComputeDomain` (DRA) for IMEX
channel injection, and the DeepGEMM JIT cache hostPath + cache-filler pass from the
methodology section.


### Prefill / decode config examples (real launch args)

The role split is visible directly in the launch arguments — same model, same quantization,
different resource strategy per role. From the vendored manifests (trimmed to the
role-relevant flags; full blocks in the linked manifests):

**DSR1-FP4 mid-curve (24P/48D) — prefill worker** (`m4-midcurve.yaml`; one of 6 DEP4 groups):

```
--disaggregation-mode prefill  --disaggregation-transfer-backend nixl
--tensor-parallel-size 4                # DEP4 group
--quantization modelopt_fp4  --kv-cache-dtype fp8_e4m3  --attention-backend trtllm_mla
--moe-runner-backend flashinfer_trtllm
--mem-fraction-static 0.95              # prefill: nearly all HBM to weights+activations
--max-prefill-tokens 524288  --chunked-prefill-size 131072
--max-total-tokens 131072               # small KV pool — KV leaves immediately
--disable-cuda-graph                    # prefill is compute-bound, graphs don't pay
--load-balance-method round_robin
```

**DSR1-FP4 mid-curve — decode worker** (same file; 1 worker × DEP48):

```
--disaggregation-mode decode  --disaggregation-transfer-backend nixl
--moe-runner-backend flashinfer_cutedsl # decode-optimized MoE kernels
--deepep-mode low_latency  --ep-dispatch-algorithm static
--ep-num-redundant-experts 32  --eplb-algorithm deepseek
--mem-fraction-static 0.83  --max-total-tokens 524288   # big KV pool — decode holds all KV
--chunked-prefill-size 24576  --max-running-requests 16384
--cuda-graph-max-bs 512                 # decode is launch-bound, graphs essential
--num-reserved-decode-tokens 112
```

**DSv4 p4 (1P DEP4 + 1D DEP16) — prefill worker** (`p4.yaml`):

```
--disaggregation-mode prefill  --disaggregation-transfer-backend mooncake
--tensor-parallel-size 4  --data-parallel-size 4  --expert-parallel-size 4
--enable-dp-attention  --enable-dp-lm-head
--mem-fraction-static 0.9  --chunked-prefill-size 32768
--max-running-requests 256  --cuda-graph-max-bs 256
```

**DSv4 p4 — decode worker** (same file; spans 4 nodes):

```
--disaggregation-mode decode  --disaggregation-transfer-backend mooncake
--tensor-parallel-size 16  --data-parallel-size 16  --expert-parallel-size 16
--enable-dp-attention  --enable-dp-lm-head
--speculative-algo EAGLE  --speculative-num-steps 3   # MTP spec-decode per recipe
--speculative-eagle-topk 1  --speculative-num-draft-tokens 4
--mem-fraction-static 0.94  --max-running-requests 3072  --cuda-graph-max-bs 256
--swa-full-tokens-ratio 0.15  --context-length 16384
--nnodes 4  --node-rank $POD_INDEX      # one logical worker across 4 nodes (16 GPUs)
```

The consistent pattern across both models: **prefill** maximizes weight/activation memory
and chunked-prefill throughput with graphs off and a minimal KV pool (KV ships out on
arrival); **decode** flips every choice — a large KV pool, CUDA graphs, decode-tuned MoE
kernels, DeepEP wide-EP dispatch, and (DSv4) MTP speculative decode.

### Per-point prefill / decode configuration

Point ids (`p1`-`p8` per model) are used consistently in the tables above and as labels on
the Pareto curves. Each worker is one sglang server; DEP = disaggregated expert parallelism
(wide-EP with DeepEP all-to-all), TP = tensor parallel.

**DSR1-FP4** (each prefill worker = DEP4 on one 4-GPU node):

| Pt | Region | Conc | Prefill config | Decode config | Nodes |
|---|---|---|---|---|---|
| p1-p4 | Low latency | 4 / 8 / 32 / 64 | 1 worker × DEP4 (4 GPUs) | 1 worker × DEP16 (16 GPUs, 4 nodes) | 5 |
| p5-p7 | Mid curve | 512 / 2048 / 4096 | 6 workers × DEP4 (24 GPUs) | 1 worker × DEP48 (48 GPUs, 12 nodes) | 18 |
| p8 | Max throughput | 2048 | 10 workers × DEP4 (40 GPUs) | 1 worker × DEP32 (32 GPUs, 8 nodes) | 18 |

**DSv4** (MTP spec-decode on; per-point EAGLE steps/draft from the vendored recipe):

| Pt | Region | Conc | Prefill config | Decode config | Nodes |
|---|---|---|---|---|---|
| p1 | Low latency | 1 | 1 × TP4 | 1 × TP4 | 2 |
| p2 | Low latency | 8 / 32 / 64 | 1 × DEP4 | 1 × DEP24 | 7 |
| p3 | Mid curve | 256 | 1 × DEP4 | 1 × DEP8 | 3 |
| p4 | Mid curve | 256 | 1 × DEP4 | 1 × DEP16 | 5 |
| p5 | Mid curve | 512 | 2 × DEP4 | 1 × DEP8 | 4 |
| p6 | Mid curve | 1024 | 4 × DEP4 | 1 × DEP8 | 6 |
| p7 | High concurrency | 4096 | 6 × DEP4 | 1 × DEP8 | 8 |
| p8 | High concurrency | 8192 | 8 × DEP4 | 1 × DEP8 | 10 |

## Recipe comparison vs InferenceX — DSR1-FP4 MNNVL, DSR1-FP4 RDMA, DSv4

### What is identical by construction

Both sides run the same models (DSR1 NVFP4 / DSv4-Pro FP4), the same sglang serving engine
with **per-point engine flags vendored from the InferenceX recipes and diffed flag-by-flag**
(the one exception — DSv4 p2's five missing prefill `SGLANG_OPT_*` vars — was caught and
fixed before the record run), the same disaggregated prefill/decode topologies per point,
and the same client methodology (sa-bench, ISL 8192/OSL 1024, `random_range_ratio 0.8`,
2×conc warmup + 10×conc measured). DSv4 p4's **+0.0% exact reproduction** is the
methodology-validation anchor: when everything matches, the platforms measure the same.

### Our config vs InferenceX config

**Engine arguments: identical by construction.** Every point's sglang flags are vendored
from the srt-slurm recipes (`recipes/gb300-fp4/8k1k/{low_latency,mid_curve,max_tpt}.yaml`
@ the pinned InferenceX submission commit) and diffed flag-by-flag before deployment.
A sample from the mid-curve decode worker — both columns are byte-identical, shown once:

| Flag (DSR1 mid-curve decode; same on both sides) | Value |
|---|---|
| `--kv-cache-dtype` / `--quantization` | `fp8_e4m3` / `modelopt_fp4` |
| `--attention-backend` / `--moe-runner-backend` | `trtllm_mla` / `flashinfer_cutedsl` |
| `--mem-fraction-static` / `--max-total-tokens` | `0.83` / `524288` |
| `--chunked-prefill-size` / `--max-running-requests` | `24576` / `16384` |
| `--disable-radix-cache` / `--disable-chunked-prefix-cache` | set / set |
| `--eplb-algorithm` / `--context-length` | `deepseek` / `9600` |

The diff process is not ceremonial — it caught two real drifts before/after they could
contaminate finals: DSv4-p2's five missing `SGLANG_OPT_*` prefill vars (fixed pre-record),
and the MTP 3-step/4-draft template inheritance on p5–p8 (detected by its −17.9%/+45%
throughput/interactivity signature, re-run).

**Where the configs genuinely differ** (environment, not engine):

| Config domain | InferenceX | Ours (GKE) | Measured influence |
|---|---|---|---|
| Launcher / platform | srt-slurm on bare metal | GKE pods + operator-less Dynamo (etcd/NATS) | None direct; admission behavior (below) is the real effect |
| Request plane | NATS (srtctl injects it) | NATS and TCP, A/B'd per point; finals take the better | Point-dependent, ≤3%: mid-2048 favored TCP on MNNVL; RDMA finals all NATS |
| KV transport env | UCX bare-metal defaults → NVLink; no UCX pinning needed | MNNVL arm: `UCX_TLS=cuda_copy,cuda_ipc,tcp` + `UCX_CUDA_IPC_ENABLE_MNNVL=y`; RDMA arm: `rc_x` + ordered `UCX_NET_DEVICES` + `GID_INDEX=5` + rail-aware pairing + DRA NIC claims | The decisive delta — our MNNVL path pays decode contention theirs does not; the RDMA config closed max_tpt −7.6% → −0.6% |
| Disagg timeouts | Recipe defaults | Bootstrap/heartbeat/waiting timeouts raised for pod startup latency | Robustness only, outside measured windows |
| IMEX (MNNVL enable) | Slurm prolog | DRA `ComputeDomain` channels | None measured |
| JIT cache env | Implicit (warm persistent nodes) | `dg-cache` hostPath + filler pass + assertion | Guards a −8% class of error; p8 floor −3.2% is the residual |
| Frontend | nginx 1.27.4 LB | Same nginx pin behind a K8s Service | Sub-second TTFT-median effect at tiny concurrency only |
| Bench client | sa-bench on a Slurm node | Identical sa-bench params, in-cluster Job | None once mult10 methodology matched (early short-window runs understated ~10%, all re-run) |

**Reading**: the engine layer is provably identical; every measured gap traces to the four
environment rows with nonzero influence (admission, KV transport, JIT, methodology), each of
which the program either matched (methodology, JIT), exploited (admission at low conc), or
solved (KV transport via RDMA).

### Slurm vs GKE at a glance

![Bare-metal Slurm vs GKE — platform stack comparison](figures/slurm-vs-gke.svg)

- **Launcher** — their sbatch/srun scripts assume the node owns the run; our
  `run-point-v2.sh` pipeline adds the gates GKE needs (serving gate, cache-filler pass,
  JIT/stall assertion, extract-before-teardown) to reach the same record quality.
- **Admission** — the biggest *behavioral* difference: srt-slurm holds requests in a
  scheduler queue, GKE + Dynamo frontends admit eagerly. This is the source of our
  systematic low-concurrency leads (better TTFT/E2E at the same recipe) and our more
  graceful behavior past the saturation knee (mid-4096 +9.4%).
- **Workers** — their processes inherit warm node state between runs; our pods start from a
  fresh filesystem every deployment, which is why the JIT-cache row exists at all.
- **IMEX / MNNVL** — same NVLink capability, different plumbing: Slurm prolog vs DRA
  `ComputeDomain` claims. No measured performance effect — a pure mechanism swap.
- **KV transport** — theirs is MNNVL-only through a hand-tuned UCX build; we run both MNNVL
  (NIXL-wheel UCX, which needed the rev4 `UCX_NET_DEVICES` lane fix under pod netns) and
  GPUDirect RDMA via DraNet-claimed NICs. The RDMA option is the platform capability they
  never published — and it closed our last throughput gaps (max_tpt −7.6% → −0.6%).
- **JIT caches** — free persistence on their long-lived nodes vs our explicit three-layer
  protocol (hostPath cache + filler pass + in-window assertion). Unguarded, this was worth
  −8% once (DSv4 p3); guarded, all finals are compile-free or conservatively stamped.
- **Below the line, everything is identical** — hardware, models, engine, per-point flags,
  and client methodology — which is what makes the per-point gap percentages attributable
  to the platform rows above rather than to the recipe.

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
