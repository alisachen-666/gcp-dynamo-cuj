# RDMA-KV Network Report — GPUDirect RDMA transport A/B on GKE A4X Max

Every point re-run with the KV-transfer path on **GPUDirect RDMA** over the 8-rail
RoCE fabric (DraNet ipvlan, DRA `mrdma.google.com` claims) instead of MNNVL/NVLink.
NCCL/DeepEP collectives stay on NVLink in both arms — only KV transfer differs.
MNNVL results remain the recipe-parity record; this report isolates the network
transport variable.

**DSR1 (NIXL/UCX)**: `UCX_TLS=cuda_copy,rc_x,tcp`, `UCX_IB_GID_INDEX=5` (in-pod
ipvlan RoCEv2 GID), rail-aware pairing `UCX_IB_ROCE_LOCAL_SUBNET=y` +
`UCX_IB_ROCE_SUBNET_PREFIX_LEN=64` (rails are 8 disjoint /64s, no cross-rail routes).
Transport audited per run: bulk ops must be 100% `rc_mlx5`, zero `cuda_ipc`.

**DSv4 (mooncake)**: `MC_FORCE_MNNVL=0` + mrdma claims (mooncake pairs NICs by GPU
affinity — rail-aligned by construction on A4X).

Methodology: identical sa-bench parameters as the MNNVL record runs (mult10 for
DSR1, 3x warm + 10x measured for DSv4). DSv4 RDMA points p1r–p7r ran PASSES=1
(diagnostic) with the JIT/stall assertion stamping VALID/SUSPECT; m4r/mxr/p8r/llr
per their runner defaults. `pending` = not yet collected.

## DSR1-FP4 8k1k

| Topology | Conc | InferenceMax out/GPU | MNNVL final out/GPU | RDMA out/GPU | RDMA vs MNNVL | RDMA vs InferenceMax | RDMA TPOT vs MNNVL |
|---|---|---|---|---|---|---|---|
| 4P/16D | 4 | 34.4 | 36.5 | 37.2 | +1.9% | +8.1% | -0.4% (lower better) |
| 4P/16D | 8 | 64.3 | 69.3 | 69.6 | +0.4% | +8.2% | -0.7% (lower better) |
| 4P/16D | 32 | 192.1 | 197.8 | 195.1 | -1.4% | +1.5% | -0.7% (lower better) |
| 4P/16D | 64 | 290.3 | 303.1 | 300.2 | -1.0% | +3.4% | -0.9% (lower better) |
| 24P/48D | 512 | 442.2 | 469.0 | 491.1 | +4.7% | +11.1% | -1.9% (lower better) |
| 24P/48D | 2048 | 838.8 | 816.5 | 828.7 | +1.5% | -1.2% | -26.5% (lower better) |
| 24P/48D | 4096 | 765.5 | 837.4 | 830.8 | -0.8% | +8.5% | -22.5% (lower better) |
| 40P/32D | 2048 | 1,781.6 | 1,646.7 | 1,771.7 | +7.6% | -0.6% | +3.9% (lower better) |

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

## DSv4 8k1k

**Descoped (user decision 2026-08-07): no DSv4 RDMA sweep.** DSv4 record results
are MNNVL (recipe parity); the rows below show the mooncake-RDMA arms as designed
but not executed.

| Topology | Conc | InferenceMax out/GPU | MNNVL final out/GPU | RDMA out/GPU | RDMA vs MNNVL | RDMA vs InferenceMax | RDMA TPOT vs MNNVL |
|---|---|---|---|---|---|---|---|
| 4P/4D | 1 | 43.2 | 48.3 | *pending* | — | — | — |
| 4P/8D | 256 | 875.1 | 877.7 | *pending* | — | — | — |
| 4P/16D | 256 | 424.5 | 424.5 | *pending* | — | — | — |
| 4P/24D | 8 | 43.0 | 51.0 | *pending* | — | — | — |
| 4P/24D | 32 | 124.5 | 136.3 | *pending* | — | — | — |
| 4P/24D | 64 | 207.0 | 223.5 | *pending* | — | — | — |
| 8P/8D | 512 | 1,620.9 | 1,650.2 | *pending* | — | — | — |
| 16P/8D | 1024 | 3,249.0 | 3,263.6 | *pending* | — | — | — |
| 24P/8D | 4096 | 5,169.3 | 5,151.2 | *pending* | — | — | — |
| 32P/8D | 8192 | 6,795.2 | — | *pending* | — | — | — |

## Reading the numbers

- **RDMA vs MNNVL** is the controlled A/B (same recipe, same methodology, same
  cluster) — it isolates the KV-transport contribution.
- **RDMA vs InferenceMax** contextualizes against the published bare-metal numbers.
- Transport evidence per run (rc_mlx5/cuda_ipc/NIXL-error totals) is in the run
  logs under `server-logs/dsr1-{llr,m4r-rev3-3pts,mxr}` and `server-logs/dsv4-p*r`
  in GCS; configs under `configs/` (see the main reports for the per-cell links).

