# DeepSeek-R1 FP8 GB300 8k1k Benchmark Report — GKE A4X Max

**Target**: InferenceX `dsr1-fp8-gb300-dynamo-sglang` (configs/nvidia-master.yaml L3077), recipes
srt-slurm `recipes/gb300-fp8/8k1k/stp/` @ branch `sa-submission-q2-2026` (vendored in
`../dsr1-fp8-sweep/reference/`; see PLAN.md there for the full parity table).

**Setup**: cluster `REDACTED-GKE-CLUSTER` (REDACTED-GCP-PROJECT), `a4x-maxgpu-4g-metal` (4x GB300/node);
image `lmsysorg/sglang:v0.5.8.post1-cu130` + ai-dynamo 0.8.0; model `deepseek-ai/DeepSeek-R1-0528`
(native FP8); operator-less Dynamo (etcd+NATS), NIXL KV transfer; sa-bench ISL 8192 / OSL 1024,
range-ratio 0.8, ignore-eos, req_rate inf; warmup 3x conc @250 + measured 10x conc.

**Run status (2026-08-01)**: mid (5P-DEP8 / 1D-DEP32, 72 GPUs) conc 128/256/512/1024 complete;
low-latency (1P1D TP4) conc 4 complete (result reconstructed from the bench client log — the
bench job died at conc 8 before extracting its JSON); low-latency conc 8 FAILED
("Never received a valid chunk"); max (6P-DEP8 / 1D-DEP24) conc 2048/4096 died mid-warmup.
Remaining: ll conc-8, max conc-2048/4096.

**Known caveat**: mid conc-128 completed 1250/1280 requests (30 failures) — treat as
provisional pending a rerun.

## Comparison vs InferenceX published results (per metric category)

Theirs converted from the CSV's mislabeled seconds to true ms. `(base)` = our run pre-dates the 10x-measured methodology (understates our throughput ~10%); `(log-extracted)` = result reconstructed from the bench log.

### Throughput (out ÷ decode GPUs, in ÷ prefill GPUs, total ÷ all)

| Config | Conc | Out/decode-GPU ours | theirs | gap | In/prefill-GPU ours | theirs | gap | Total/GPU ours | theirs | gap |
|---|---|---|---|---|---|---|---|---|---|---|
| 4P/4D (log-extracted) | 4 | 91.4 | 90.1 | +1.4% | 730 | 720 | +1.4% | 410.6 | 405.1 | +1.4% |
| 4P/4D (not run) | 8 | — | 145.6 | — | — | 1150 | — | — | 647.8 | — |
| 40P/32D  | 128 | 233.0 | 227.8 | +2.3% | 1496 | 1463 | +2.2% | 934.8 | 914.3 | +2.2% |
| 40P/32D  | 256 | 442.9 | 428.5 | +3.4% | 2834 | 2741 | +3.4% | 1771.1 | 1713.5 | +3.4% |
| 40P/32D  | 512 | 789.4 | 773.3 | +2.1% | 5052 | 4949 | +2.1% | 3157.6 | 3093.3 | +2.1% |
| 40P/32D  | 1024 | 1352.7 | 1354.7 | -0.2% | 8655 | 8669 | -0.2% | 5409.5 | 5418.3 | -0.2% |
| 48P/24D (not run) | 2048 | — | 2505.0 | — | — | 10011 | — | — | 7508.8 | — |
| 48P/24D (not run) | 4096 | — | 2858.9 | — | — | 11439 | — | — | 8579.1 | — |

### Latency (true ms; lower is better)

| Config | Conc | TTFT med ours | theirs | gap | TTFT p99 ours | theirs | gap | TPOT mean ours | theirs | gap | E2E mean ours | theirs | gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4P/4D (log-extracted) | 4 | 450 | 429 | +4.9% | 1465 | 1999 | -26.7% ✅ | 10.03 | 10.10 | -0.7% | 9803 | 9930 | -1.3% |
| 40P/32D  | 128 | 1064 | 1299 | -18.1% ✅ | 3209 | 4118 | -22.1% ✅ | 15.05 | 15.29 | -1.6% | 15069 | 15557 | -3.1% |
| 40P/32D  | 256 | 1383 | 1646 | -16.0% ✅ | 5093 | 5954 | -14.5% ✅ | 15.53 | 15.78 | -1.6% | 15963 | 16449 | -3.0% |
| 40P/32D  | 512 | 2030 | 2454 | -17.3% ✅ | 9241 | 9149 | +1.0% | 16.77 | 16.76 | +0.1% | 17776 | 18125 | -1.9% |
| 40P/32D  | 1024 | 2645 | 3126 | -15.4% ✅ | 16304 | 16247 | +0.4% | 18.97 | 18.47 | +2.7% | 20891 | 20839 | +0.2% |

### Interactivity (tok/s/user = 1/TPOT; like-for-like)

| Config | Conc | Mean ours | theirs | gap | Median ours | theirs | gap |
|---|---|---|---|---|---|---|---|
| 4P/4D (log-extracted) | 4 | 99.7 | 99.0 | +0.7% | 99.2 | 98.5 | +0.7% |
| 40P/32D  | 128 | 66.5 | 65.4 | +1.6% | 66.5 | 65.6 | +1.5% |
| 40P/32D  | 256 | 64.4 | 63.4 | +1.6% | 64.4 | 63.8 | +0.9% |
| 40P/32D  | 512 | 59.6 | 59.7 | -0.1% | 59.4 | 59.9 | -0.8% |
| 40P/32D  | 1024 | 52.7 | 54.1 | -2.6% | 52.6 | 54.0 | -2.7% |
