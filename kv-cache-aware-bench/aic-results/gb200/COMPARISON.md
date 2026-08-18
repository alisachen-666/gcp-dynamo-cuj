# AIC gb200/trtllm/24-GPU agg solve vs NVIDIA's published Kimi-K2.5 recipe numbers

Solve: aiconfigurator 0.10.0, SILICON DB (trtllm 1.3.0rc10), workload encoded from
NVIDIA's dataset description: ISL 200,000 (their "~200k context window"), OSL 1,000
(estimate; theirs unpublished), goodput gates TTFT 5 s / TPOT 10 ms (their recipe's),
warm bracket prefix 190k (their "KV-reuse-heavy"), cold bracket no reuse.

## AIC predictions (aggregated, 24 GPUs)

| Bracket | SLA-best config | TTFT | TPOT | tok/s/user | output tok/s/GPU |
|---|---|---|---|---|---|
| warm | dp4/etp4 4-GPU workers, conc 4 | 3.6 s | 19.5 ms | 51.4 | 46.9 |
| warm (max-throughput pareto) | dp8/ep8, conc 80 | — | 48.5 ms | 20.6 | 184.5 |
| cold | tp4/etp4, conc 3 | 25.0 s | 35.9 ms | 27.8 | 20.6 |

## vs NVIDIA's published "Agg + Round-robin, no Eagle": ~105 tok/s/user, ~1,700 "tok/s/GPU"

1. **AIC cannot reproduce 105 tok/s/user at ISL 200k**: with a 200k KV context,
   per-token decode attention alone pushes TPOT to ~19–36 ms (52–28 tok/s/user).
   105 tok/s/user (≈9.5 ms TPOT) is only coherent at a much smaller context —
   AIC-implied true mean ISL for their trace ≈ 100–140k. Conclusion: their
   "~200k context window" is a cap, not a mean; published per-user numbers imply
   roughly half that as the working ISL. (Matches our earlier independent
   inference, now backed by NVIDIA's own solver.)
2. **"~1,700 tok/s/GPU" is not output throughput under any AIC scenario**
   (predicted output range: 21–185 tok/s/GPU). Total-token (sent) accounting at
   their conc-8 pin lands in the right order of magnitude (≈0.2 req/s × ~140k
   ISL ≈ 1,200–1,700/GPU) — strongest evidence yet for the total-token reading.
3. **Reuse is worth 2× even in solver space**: warm vs cold = 51 vs 28 tok/s/user,
   TTFT 3.6 s vs 25 s. Any published number on this workload is meaningless
   without stating realized cache hit rate.
4. **Cross-check vs our GB300 silicon**: at SLA-comparable interactivity our
   measured sglang agg KV cell (conc 8: 84 tok/s/user, 47.5 output tok/s/GPU,
   real mean ISL 96k) sits within ~1% per-GPU of AIC's gb200 warm SLA point
   (51 user / 46.9 per-GPU at ISL 200k) — GB300-vs-GB200 and the ISL difference
   roughly offsetting. Eagle rows excluded (AIC 0.10 does not model speculative
   decoding).
