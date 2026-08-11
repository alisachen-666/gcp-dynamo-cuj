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
PREFILL_TOKRATE = 65_000    # UNCACHED tok/s per 4-GPU worker (aic cold solve: 137k tok / 8.4s per DP slot x4)
TPOT_BASE_MS = 14.3          # recalibrated vs live S4: ITL p50 15.4ms @ ~11 seq/worker
TPOT_SLOPE_MS = 0.1
KV_CAPACITY_TOKENS = 11_900_000  # per prefill worker: 4 GPU x ~104GB free x 0.8 / 35KB/token
ROUTER = {"prefill_load_scale": 1.0, "overlap_credit": 1.0}  # KV-NVDA defaults; tuned arm overrides
# aggregated single-node TP4 worker model (AIC silicon agg rows: ttft 442-497ms
# warm, tpot 13.6/16.4/18.3 ms at bs 1/2/3); calibrate vs live agg1n when it lands
AGG_PREFILL_RATE = 45_000   # uncached tok/s per TP4 agg worker (plain TP, no attention-DP)
AGG_TPOT_BASE_MS = 11.5
AGG_TPOT_SLOPE_MS = 2.3


def agg_tpot_ms(batch):
    return AGG_TPOT_BASE_MS + AGG_TPOT_SLOPE_MS * batch


# sglang seed constants (sgl-sim v1): AIC-ratio transfer onto the live-calibrated
# trtllm constants above, from the matched SILICON solves (same isl/osl/prefix).
#   prefill: AIC cold TTFT floor per GPU x4 (lens validated: trtllm AIC floor
#            16.3k/GPU x4 = 65.1k == live-calibrated 65k)
#   decode:  min-TPOT-per-batch ratio sgl/trt at operating bs (0.72 base, 0.70 slope)
#   agg:     sglang shows a batch cliff at bs>=8 for 137k ISL (AIC ratio jumps
#            0.70 -> 2.47) -> piecewise model instead of one line
def apply_sglang_constants():
    global PREFILL_TOKRATE, TPOT_BASE_MS, TPOT_SLOPE_MS, AGG_PREFILL_RATE, agg_tpot_ms
    PREFILL_TOKRATE = 78_000       # 19.4k/GPU AIC cold floor x4
    TPOT_BASE_MS = 10.3            # 14.3 x 0.72
    TPOT_SLOPE_MS = 0.07           # 0.1 x 0.70
    AGG_PREFILL_RATE = 23_700      # 45k x cold-agg TTFT-floor ratio 0.527
    def agg_tpot_ms(batch):        # noqa: F811 - piecewise, rebinds module fn
        if batch <= 7:
            return 7.0 + 1.6 * batch           # 0.70 x trt live segment
        return 28.4 + 5.68 * batch             # 2.47 x trt live segment
    globals()["agg_tpot_ms"] = agg_tpot_ms


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


def simulate(trace, n_prefill, n_decode, policy, conc, router=None):
    router = router or ROUTER
    P = [PrefillWorker() for _ in range(n_prefill)]
    D_load = [0] * n_decode              # in-flight sequences per decode worker
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
            elif policy == "ll":  # least-loaded (queued blocks), cache-blind
                w = min(range(n_prefill), key=lambda i: P[i].load_blocks(now))
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
    # steady-state window: second half of requests (first half warms caches)
    half = len(results) // 2
    ss = results[half:]
    t0 = min(x[3] for x in ss) - max(x[1] for x in ss) / 1000 * max(x[2] for x in ss)
    dur = max(x[3] for x in ss) - min(x[3] for x in ss) or 1e-9
    ttfts = sorted(x[0] for x in ss)
    out_tokens = sum(x[2] for x in ss)
    return {
        "throughput_tok_s": out_tokens / dur,
        "ttft_p50_s": ttfts[len(ttfts) // 2],
        "ttft_p95_s": ttfts[int(len(ttfts) * 0.95)],
        "ttft_p99_s": ttfts[min(len(ttfts) - 1, int(len(ttfts) * 0.99))],
        "tpot_mean_ms": sum(x[1] for x in ss) / len(ss),
        "hit_rate": hits / max(1, tot_blocks),
        "req_per_s": len(ss) / dur,
    }


def simulate_agg(trace, n_workers, policy, conc, router=None):
    """Aggregated: each worker prefills AND decodes; router picks one worker."""
    router = router or ROUTER
    P = [PrefillWorker() for _ in range(n_workers)]   # reuse cache+queue model
    D_load = [0] * n_workers
    rr_i = 0
    results = []
    hits = tot_blocks = 0
    pending = list(trace)
    inflight = []
    now = 0.0
    idx = 0
    while idx < len(pending) or inflight:
        while idx < len(pending) and len(inflight) < conc:
            r = pending[idx]; idx += 1
            hid = r["hash_ids"]; total_blocks = len(hid)
            if policy == "rr":
                w = rr_i % n_workers; rr_i += 1
                ov = P[w].overlap_blocks(hid)
            elif policy == "ll":
                w = min(range(n_workers), key=lambda i: P[i].load_blocks(now) + D_load[i]*100)
                ov = P[w].overlap_blocks(hid)
            else:
                best, w, ov = None, 0, 0
                for i, pw in enumerate(P):
                    o = pw.overlap_blocks(hid)
                    adj = max(0.0, total_blocks - router["overlap_credit"] * o)
                    score = router["prefill_load_scale"] * adj + pw.load_blocks(now) + D_load[i]*100
                    if best is None or score < best:
                        best, w, ov = score, i, o
            hits += ov; tot_blocks += total_blocks
            new_tokens = (total_blocks - ov) * BLOCK_TOKENS
            svc = max(0.005, new_tokens / AGG_PREFILL_RATE)
            start = max(now, P[w].free_at)
            pf_done = start + svc
            P[w].free_at = pf_done
            P[w].queued.append((pf_done, total_blocks - ov))
            P[w].insert(hid)
            ttft = pf_done - now
            D_load[w] += 1
            tpot = agg_tpot_ms(D_load[w]) / 1000.0
            dec_done = pf_done + r["output_length"] * tpot
            heapq.heappush(inflight, (dec_done, w))
            results.append((ttft, tpot * 1000, r["output_length"], dec_done))
        if inflight:
            done_t, w = heapq.heappop(inflight)
            now = max(now, done_t)
            D_load[w] -= 1
    half = len(results) // 2
    ss = results[half:]
    dur = max(x[3] for x in ss) - min(x[3] for x in ss) or 1e-9
    ttfts = sorted(x[0] for x in ss)
    return {
        "throughput_tok_s": sum(x[2] for x in ss) / dur,
        "ttft_p50_s": ttfts[len(ttfts)//2],
        "ttft_p95_s": ttfts[int(len(ttfts)*0.95)],
        "ttft_p99_s": ttfts[min(len(ttfts)-1, int(len(ttfts)*0.99))],
        "tpot_mean_ms": sum(x[1] for x in ss) / len(ss),
        "hit_rate": hits / max(1, tot_blocks),
        "req_per_s": len(ss) / dur,
    }


def sweep_agg(trace, conc_list, out_csv):
    import csv as _csv
    POLICIES = [("rr", None), ("ll", None),
                ("kv-nvda", {"prefill_load_scale": 1.0, "overlap_credit": 1.0}),
                ("kv-s1.5-c0.8", {"prefill_load_scale": 1.5, "overlap_credit": 0.8}),
                ("kv-s3.0-c0.8", {"prefill_load_scale": 3.0, "overlap_credit": 0.8}),
                ] + [(f"kv-t-c{c}", {"prefill_load_scale": 2.0, "overlap_credit": c})
                     for c in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]
    rows = []
    for conc in conc_list:
        cell = {}
        for name, rt in POLICIES:
            pol = "kv" if name.startswith("kv") else name
            m = simulate_agg(trace, 6, pol, conc, rt)
            m.update(pd="agg6", conc=conc, policy=name)
            m["sla_pass"] = m["ttft_p95_s"] <= 5.0 and m["tpot_mean_ms"] <= 20.0
            cell[name] = m; rows.append(m)
        bk = max((m for n, m in cell.items() if n.startswith("kv")), key=lambda m: m["throughput_tok_s"])
        rr = cell["rr"]
        print(f"  agg6 conc={conc:>3} kv={bk['throughput_tok_s']:6.0f} tok/s (ttft95 {bk['ttft_p95_s']:6.2f}s) "
              f"rr={rr['throughput_tok_s']:6.0f} (ttft95 {rr['ttft_p95_s']:7.2f}s) ratio={bk['throughput_tok_s']/max(1,rr['throughput_tok_s']):.2f}")
    with open(out_csv, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} agg cells -> {out_csv}")


def sweep(trace, conc_list, out_csv, splits=None):
    """Full sweep: P:D x conc x policy; no SLA gate — max-throughput hunt.
    sla_pass column retained as an informational annotation (TTFT p95 <= 5s, TPOT <= 20ms)."""
    import csv as _csv
    POLICIES = [
        ("rr", None),
        ("ll", None),
        ("kv-nvda", {"prefill_load_scale": 1.0, "overlap_credit": 1.0}),
        *[(f"kv-t-c{c}", {"prefill_load_scale": 2.0, "overlap_credit": c})
          for c in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)],
        *[(f"kv-s{s}-c0.8", {"prefill_load_scale": s, "overlap_credit": 0.8})
          for s in (1.5, 3.0)],
    ]
    rows = []
    for np_, nd in (splits or [(1, 5), (2, 4), (3, 3), (4, 2), (5, 1)]):
        for conc in conc_list:
            cell = {}
            for name, rt in POLICIES:
                pol = "kv" if name.startswith("kv") else name
                m = simulate(trace, np_, nd, pol, conc, rt)
                m.update(pd=f"{np_}:{nd}", conc=conc, policy=name)
                m["sla_pass"] = m["ttft_p95_s"] <= 5.0 and m["tpot_mean_ms"] <= 20.0
                cell[name] = m
                rows.append(m)
            # no SLA gate (2026-08-10): pure max-throughput hunt; latency reported as info
            best_kv = max((m for n, m in cell.items() if n.startswith("kv")),
                          key=lambda m: m["throughput_tok_s"], default=None)
            if best_kv:
                rr = cell["rr"]
                ratio = best_kv["throughput_tok_s"] / max(1e-9, rr["throughput_tok_s"])
                impact = best_kv["throughput_tok_s"] * min(ratio, 10.0)
                print(f"  {np_}:{nd} conc={conc:>3} best-kv={best_kv['policy']:<10} "
                      f"kv_tok/s={best_kv['throughput_tok_s']:8.0f} (ttft95 {best_kv['ttft_p95_s']:6.2f}s) "
                      f"rr_tok/s={rr['throughput_tok_s']:8.0f} (ttft95 {rr['ttft_p95_s']:7.2f}s"
                      f"{'' if rr['sla_pass'] else ' FAIL'}) ratio={ratio:5.2f} impact={impact:9.0f}")
    with open(out_csv, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} cells -> {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--conc", type=int, default=32)
    ap.add_argument("--requests", type=int, default=4000)
    ap.add_argument("--sweep", action="store_true", help="full highlight-point sweep")
    ap.add_argument("--agg", action="store_true", help="aggregated 6xTP4 sweep")
    ap.add_argument("--disagg72", action="store_true", help="72-GPU disagg: 18 workers, P:D splits")
    ap.add_argument("--out", default="dynosim_sweep.csv")
    ap.add_argument("--backend", choices=["trtllm", "sglang"], default="trtllm")
    args = ap.parse_args()
    if args.backend == "sglang":
        apply_sglang_constants()
        print("backend=sglang (sgl-sim v1 seed: prefill 78k, tpot 10.3+0.07bs, "
              "agg 23.7k + piecewise tpot with bs>=8 cliff)")
    trace = []
    with open(args.trace) as f:
        for line in f:
            trace.append(json.loads(line))
            if len(trace) >= args.requests:
                break
    if args.sweep:
        print(f"DynoSim sweep: {len(trace)} requests, 24 GPUs, SLA: ttft95<=5s tpot<=20ms")
        sweep(trace, [8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384], args.out)
    elif getattr(args, "disagg72", False):
        print(f"DynoSim 72-GPU disagg sweep: {len(trace)} requests, 18x TP4 workers")
        sweep(trace, [48, 96, 144, 192, 288, 384, 512, 768],
              args.out, splits=[(3, 15), (6, 12), (9, 9), (12, 6), (15, 3)])
    elif getattr(args, "agg", False):
        print(f"DynoSim AGG sweep: {len(trace)} requests, 6x TP4 workers (24 GPUs)")
        sweep_agg(trace, [8, 16, 24, 32, 48, 64, 96, 128, 192, 256], args.out)
    else:
        print(f"DynoSim: {len(trace)} requests, conc {args.conc}, 24 GPUs (4/worker)")
        for np_, nd in (splits or [(1, 5), (2, 4), (3, 3), (4, 2), (5, 1)]):
            for pol in ["rr", "kv"]:
                m = simulate(trace, np_, nd, pol, args.conc)
                print(f"{np_}:{nd} {pol:>3} req/s={m['req_per_s']:.2f} tok/s={m['throughput_tok_s']:.0f} "
                      f"ttft50={m['ttft_p50_s']:.2f}s ttft95={m['ttft_p95_s']:.2f}s "
                      f"tpot={m['tpot_mean_ms']:.1f}ms hit={100*m['hit_rate']:.1f}%")
