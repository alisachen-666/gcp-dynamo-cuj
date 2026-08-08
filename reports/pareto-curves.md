# Pareto Curves — DSR1 & DSV4 on GB300 NVL72 (GKE A4X Max)

Throughput-per-GPU Pareto curves for the three sweeps, in the style of the
[InferenceX inference dashboard](https://inferencex.semianalysis.com/inference):
**Ours** (GKE A4X Max measured) overlaid on **InferenceX published** GB300 numbers, with
three x-axis views per model — interactivity, E2E latency, and TTFT.

- Workload: 8k/1k (ISL 8192 / OSL 1024), random dataset, ignore-eos, disaggregated prefill/decode.
- **Y axis** is total (input + output) throughput per GPU: `(input tok/s + output tok/s) / total GPUs`.
- **Metrics are mean-derived** (interactivity = 1000 / mean TPOT; TTFT and E2E are means),
  matching the mean-to-mean convention in `compare_inferencex*.py`.
- Point selection follows the repo rules: `mult10` runs preferred where present, untuned
  primaries, NATS/TCP/kv-headroom A/B variants excluded.
- Figures regenerate from `results-summary/*.json` + `reference/*.csv`
  (see `reports/figures/`).

## DeepSeek-R1 FP4

Points: `ll` low-latency 4P/16D (conc 4–64), `m4` 24P/48D (conc 512–4096), `mx` 40P/32D (conc 2048).

![DSR1 FP4 — throughput/GPU vs interactivity](figures/pareto-dsr1-fp4-interactivity.svg)

![DSR1 FP4 — throughput/GPU vs E2E latency](figures/pareto-dsr1-fp4-e2e-latency.svg)

![DSR1 FP4 — throughput/GPU vs TTFT](figures/pareto-dsr1-fp4-ttft.svg)

### Point-by-point vs InferenceMax — DSR1 FP4

| Source | Concurrency | Config | Curve | Total Tput (tok/s/GPU) | Input Tput (tok/s/GPU) | Output Tput (tok/s/GPU) | Median E2EL (s) | Median ITL (s) |
|---|---|---|---|---|---|---|---|---|
| InferenceMax | 4 | 4P/16D · 20 GPU | Low-latency | 247 | 1,100 | 34 | 6.40 | 0.33 |
| GKE (`ll`) | | | | 253 (+2.08%) | 1,123 (+2.16%) | 35 (+1.42%) | 6.25 (-2.41%) | 0.31 (-6.26%) |
| InferenceMax | 8 | 4P/16D · 20 GPU | Low-latency | 458 | 2,032 | 64 | 6.99 | 0.37 |
| GKE (`ll`) | | | | 475 (+3.75%) | 2,108 (+3.74%) | 67 (+3.87%) | 6.69 (-4.27%) | 0.33 (-11.34%) |
| InferenceMax | 32 | 4P/16D · 20 GPU | Low-latency | 1,375 | 6,105 | 192 | 9.09 | 0.46 |
| GKE (`ll/mult10`) | | | | 1,415 (+2.94%) | 6,284 (+2.93%) | 198 (+2.95%) | 8.87 (-2.45%) | 0.45 (-2.13%) |
| InferenceMax | 64 | 4P/16D · 20 GPU | Low-latency | 2,094 | 9,307 | 290 | 11.98 | 0.61 |
| GKE (`ll/mult10`) | | | | 2,186 (+4.39%) | 9,716 (+4.39%) | 303 (+4.41%) | 11.43 (-4.61%) | 0.58 (-3.70%) |
| InferenceMax | 512 | 24P/48D · 72 GPU | Mid-curve | 2,653 | 7,076 | 442 | 15.14 | 0.64 |
| GKE (`m4`) | | | | 2,618 (-1.32%) | 6,982 (-1.32%) | 436 (-1.32%) | 16.67 (+10.12%) | 0.81 (+27.09%) |
| InferenceMax | 2048 | 24P/48D · 72 GPU | Mid-curve | 5,029 | 13,409 | 839 | 44.28 | 0.66 |
| GKE (`m4`) | | | | 4,894 (-2.67%) | 13,050 (-2.67%) | 816 (-2.66%) | 41.47 (-6.33%) | 0.90 (+36.99%) |
| InferenceMax | 4096 | 24P/48D · 72 GPU | Mid-curve | 4,594 | 12,252 | 766 | 92.94 | 0.68 |
| GKE (`m4`) | | | | 5,026 (+9.38%) | 13,402 (+9.38%) | 837 (+9.39%) | 84.29 (-9.31%) | 0.90 (+32.51%) |
| InferenceMax | 2048 | 40P/32D · 72 GPU | High-throughput | 7,121 | 11,392 | 1,782 | 26.39 | 0.86 |
| GKE (`mx`) | | | | 6,477 (-9.04%) | 10,362 (-9.04%) | 1,621 (-9.03%) | 32.67 (+23.79%) | 0.87 (+0.77%) |

Percentages on GKE rows are the gap vs the InferenceMax published value for the same point: `(GKE − InferenceMax) / InferenceMax`. Positive means GKE is higher — better for throughput columns, worse for latency columns. Input/output tput are per prefill/decode GPU respectively; E2EL and ITL are medians in seconds.

## DeepSeek-R1 FP8

Points: `ll` 4P/4D (conc 4), `mid` 40P/32D (conc 128–1024). InferenceX additionally
publishes 48P/24D at conc 2048/4096, which we have not run.

![DSR1 FP8 — throughput/GPU vs interactivity](figures/pareto-dsr1-fp8-interactivity.svg)

![DSR1 FP8 — throughput/GPU vs E2E latency](figures/pareto-dsr1-fp8-e2e-latency.svg)

![DSR1 FP8 — throughput/GPU vs TTFT](figures/pareto-dsr1-fp8-ttft.svg)

## DeepSeek-V4-Pro FP4

Points: `p1`–`p8` replicating the published ladder (1 → 8192 concurrency). The dashed
hollow series is the parity-drift reruns of p5–p8 (labelled, not clean comparisons —
see `dsv4-sweep/compare_inferencex.py`). The clean p8 rerun is latency-only (no
throughput) and therefore appears only in the drift series on the chart.

![DSV4 — throughput/GPU vs interactivity](figures/pareto-dsv4-interactivity.svg)

![DSV4 — throughput/GPU vs E2E latency](figures/pareto-dsv4-e2e-latency.svg)

![DSV4 — throughput/GPU vs TTFT](figures/pareto-dsv4-ttft.svg)

### Point-by-point vs InferenceMax — DSV4-Pro

| Source | Concurrency | Config | Curve | Total Tput (tok/s/GPU) | Input Tput (tok/s/GPU) | Output Tput (tok/s/GPU) | Median E2EL (s) | Median ITL (s) |
|---|---|---|---|---|---|---|---|---|
| InferenceMax | 1 | 4P/4D · 8 GPU | Low-latency | 193 | 344 | 43 | 5.06 | 0.02 |
| GKE (`p1`) | | | | 216 (+11.70%) | 384 (+11.70%) | 48 (+11.70%) | 4.33 (-14.47%) | 0.02 (-0.10%) |
| InferenceMax | 8 | 4P/24D · 28 GPU | Low-latency | 328 | 2,035 | 43 | 6.19 | 0.02 |
| GKE | | | | — | — | — | — | — |
| InferenceMax | 32 | 4P/24D · 28 GPU | Low-latency | 954 | 5,933 | 124 | 8.88 | 0.02 |
| GKE | | | | — | — | — | — | — |
| InferenceMax | 64 | 4P/24D · 28 GPU | Low-latency | 1,599 | 9,954 | 207 | 10.56 | 0.02 |
| GKE | | | | — | — | — | — | — |
| InferenceMax | 256 | 4P/16D · 20 GPU | Mid-curve | 3,056 | 13,580 | 424 | 32.36 | 0.80 |
| GKE (`p4`) | | | | 3,056 (+0.01%) | 13,581 (+0.01%) | 425 (+0.01%) | 31.59 (-2.37%) | 0.79 (-0.62%) |
| InferenceMax | 256 | 4P/8D · 12 GPU | Mid-curve | 5,250 | 13,999 | 875 | 31.79 | 1.06 |
| GKE (`p3`) | | | | 5,265 (+0.30%) | 14,040 (+0.30%) | 878 (+0.30%) | 32.31 (+1.66%) | 1.08 (+1.73%) |
| InferenceMax | 512 | 8P/8D · 16 GPU | Mid-curve | 7,295 | 12,968 | 1,621 | 32.08 | 1.32 |
| GKE (`p5`) | | | | 7,426 (+1.81%) | 13,202 (+1.81%) | 1,650 (+1.81%) | 32.13 (+0.16%) | 1.36 (+3.33%) |
| InferenceMax | 1024 | 16P/8D · 24 GPU | Mid-curve | 9,746 | 12,995 | 3,249 | 32.02 | 1.68 |
| GKE (`p6`) | | | | 9,790 (+0.45%) | 13,053 (+0.45%) | 3,264 (+0.45%) | 32.42 (+1.27%) | 1.69 (+0.39%) |
| InferenceMax | 4096 | 24P/8D · 32 GPU | High-throughput | 11,634 | 13,789 | 5,169 | 84.82 | 1.71 |
| GKE (`p7`) | | | | 11,593 (-0.35%) | 13,741 (-0.35%) | 5,151 (-0.35%) | 84.29 (-0.63%) | 1.65 (-3.29%) |
| InferenceMax | 8192 | 32P/8D · 40 GPU | High-throughput | 12,233 | 13,593 | 6,795 | 132.31 | 2.21 |
| GKE (`p8`, latency-only rerun) | | | | — | — | — | 135.03 (+2.06%) | 2.14 (-2.91%) |

Percentages on GKE rows are the gap vs the InferenceMax published value for the same point: `(GKE − InferenceMax) / InferenceMax`. Positive means GKE is higher — better for throughput columns, worse for latency columns. Input/output tput are per prefill/decode GPU respectively; E2EL and ITL are medians in seconds.
GKE p8 (conc 8192) is the latency-only clean rerun — no throughput numbers; concurrencies 8–64 (4P/24D) have no GKE run yet.

## Reading the curves

- On the interactivity view, up-and-right is better; on the latency views (log-x),
  up-and-left is better.
- The curves are connected scatters across topologies (as on the InferenceX dashboard),
  so segments between different P/D splits are visual guides, not operating points.
- Full per-point tables and deltas vs published live in the per-model reports:
  `benchmark-report-dsr1-8k1k.md`, `benchmark-report-dsr1-fp8-8k1k.md`,
  `benchmark-report-dsv4-8k1k.md`.
