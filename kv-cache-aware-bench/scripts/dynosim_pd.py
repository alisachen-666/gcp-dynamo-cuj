"""DynoSim v0: trace-driven P:D + routing-policy simulator for GB300 24-GPU disagg.

Simulates closed-loop replay of the Weka trace through a disaggregated stack:
  - Router: EXACT worker-scoring formula from dynamo v1.3.1
    (lib/kv-router/src/scheduling/selector.rs::worker_logit):
      score = prefill_load_scale * max(0, prefill_blocks - credit*overlap) + decode_blocks
    plus round-robin as baseline. Deterministic argmin (temperature 0).
  - Prefill workers: FCFS queue, service time = new_tokens / PREFILL_RATE
    (aic SILICON warm solve: 5.202 seq/s/worker @137k ISL cached -> rate from
    saturated seq rate x mean new tokens; calibrated as token rate).
  - Per-prefill-worker radix prefix cache over trace hash_ids (64-token blocks),
    LRU-evicted at KV capacity.
  - Decode workers: TPOT model fit to aic pareto points
    (tpot_ms ~= TPOT_BASE + TPOT_SLOPE * worker_load), decode time = out * tpot.
  - Closed loop at fixed concurrency; requests dispatched in trace order.

Outputs per (P:D, policy): throughput, TTFT p50/p95, TPOT mean, cache hit rate.
Usage: dynosim_pd.py <trace.jsonl> [--conc 32] [--requests 4000]
"""
import argparse
import heapq
import json
from collections import OrderedDict

BLOCK_TOKENS = 64
# --- aic SILICON warm-solve derived constants (gb300, trtllm, 137k ISL) ---
PREFILL_TOKRATE = 713_000    # tokens/s per 4-GPU prefill worker (5.202 seq/s x 137k)
TPOT_BASE_MS = 14.5          # fit to aic pareto: (bs/worker=2 -> 15.9), (4 -> 17.6), (12 -> 23.4)
TPOT_SLOPE_MS = 0.75
KV_CAPACITY_TOKENS = 11_900_000  # per prefill worker: 4 GPU x ~104GB free x 0.8 / 35KB/token
ROUTER = {"prefill_load_scale": 1.0, "overlap_credit": 1.0}  # KV-NVDA defaults; tuned arm overrides


class PrefillWorker:
    def __init__(self):
        self.cache = OrderedDict()   # block_id -> None (LRU order)
        self.free_at = 0.0           # queue tail time
        self.active_blocks = 0.0     # proxy for load (blocks queued, decays as served)
        self.queued = []             # (finish_time, blocks) for load accounting

    def overlap_blocks(self, hash_ids):
        n = 0
        for h in hash_ids:
            if h in self.cache:
                n += 1
            else:
                break  # prefix property: stop at first miss
        return n

    def insert(self, hash_ids):
        for h in hash_ids:
            if h in self.cache:
                self.cache.move_to_end(h)
            else:
                self.cache[h] = None
        cap = KV_CAPACITY_TOKENS // BLOCK_TOKENS
        while len(self.cache) > cap:
            self.cache.popitem(last=False)

    def load_blocks(self, now):
        self.queued = [(t, b) for (t, b) in self.queued if t > now]
        return sum(b for _, b in self.queued)


def simulate(trace, n_prefill, n_decode, policy, conc, router=ROUTER):
    P = [PrefillWorker() for _ in range(n_prefill)]
    D_load = [0] * n_decode              # in-flight sequences per decode worker
    D_free = []                          # heap of (finish_time, worker, ) decode completions
    rr_i = 0
    results = []
    hits = tot_blocks = 0

    # closed loop: maintain `conc` in flight; event clock via heap of completions
    pending = list(trace)
    inflight = []                        # heap of (decode_done_time, d_worker)
    now = 0.0
    idx = 0
    while idx < len(pending) or inflight:
        while idx < len(pending) and len(inflight) < conc:
            r = pending[idx]; idx += 1
            hid = r["hash_ids"]
            total_blocks = len(hid)
            # --- route ---
            if policy == "rr":
                w = rr_i % n_prefill; rr_i += 1
                ov = P[w].overlap_blocks(hid)
            else:  # kv-aware: verified worker_logit, argmin
                best, w, ov = None, 0, 0
                for i, pw in enumerate(P):
                    o = pw.overlap_blocks(hid)
                    adj = max(0.0, total_blocks - router["overlap_credit"] * o)
                    score = router["prefill_load_scale"] * adj + pw.load_blocks(now)
                    if best is None or score < best:
                        best, w, ov = score, i, o
            hits += ov; tot_blocks += total_blocks
            new_tokens = (total_blocks - ov) * BLOCK_TOKENS
            svc = max(0.005, new_tokens / PREFILL_TOKRATE)
            start = max(now, P[w].free_at)
            pf_done = start + svc
            P[w].free_at = pf_done
            P[w].queued.append((pf_done, total_blocks - ov))
            P[w].insert(hid)
            ttft = pf_done - now
            # --- decode: least-loaded worker ---
            d = min(range(n_decode), key=lambda i: D_load[i])
            D_load[d] += 1
            tpot = (TPOT_BASE_MS + TPOT_SLOPE_MS * D_load[d]) / 1000.0
            dec_done = pf_done + r["output_length"] * tpot
            heapq.heappush(inflight, (dec_done, d))
            results.append((ttft, tpot * 1000, r["output_length"], dec_done))
        if inflight:
            done_t, d = heapq.heappop(inflight)
            now = max(now, done_t)
            D_load[d] -= 1
    dur = max(x[3] for x in results)
    ttfts = sorted(x[0] for x in results)
    out_tokens = sum(x[2] for x in results)
    return {
        "throughput_tok_s": out_tokens / dur,
        "ttft_p50_s": ttfts[len(ttfts) // 2],
        "ttft_p95_s": ttfts[int(len(ttfts) * 0.95)],
        "tpot_mean_ms": sum(x[1] for x in results) / len(results),
        "hit_rate": hits / max(1, tot_blocks),
        "req_per_s": len(results) / dur,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--conc", type=int, default=32)
    ap.add_argument("--requests", type=int, default=4000)
    args = ap.parse_args()
    trace = []
    with open(args.trace) as f:
        for line in f:
            trace.append(json.loads(line))
            if len(trace) >= args.requests:
                break
    print(f"DynoSim v0: {len(trace)} requests, conc {args.conc}, 24 GPUs (4/worker)")
    print(f"{'P:D':>6} {'policy':>6} {'req/s':>7} {'tok/s':>8} {'ttft_p50':>9} "
          f"{'ttft_p95':>9} {'tpot_ms':>8} {'hit%':>6}")
    for np_, nd in [(1, 5), (2, 4), (3, 3)]:
        for pol in ["rr", "kv"]:
            m = simulate(trace, np_, nd, pol, args.conc)
            print(f"{np_}:{nd:>4} {pol:>6} {m['req_per_s']:7.2f} {m['throughput_tok_s']:8.0f} "
                  f"{m['ttft_p50_s']:8.2f}s {m['ttft_p95_s']:8.2f}s "
                  f"{m['tpot_mean_ms']:8.1f} {100*m['hit_rate']:5.1f}")
