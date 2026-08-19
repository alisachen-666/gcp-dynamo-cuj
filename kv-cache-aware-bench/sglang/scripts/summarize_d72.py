#!/usr/bin/env python3
"""Summarize 72-GPU disagg silicon runs (aiperf profile_export_aiperf.json) into one CSV.

Usage: python3 scripts/summarize_d72.py [results/silicon/disagg72] > results/silicon/disagg72/summary.csv
Each run dir is <epoch>_<job>/<model>_trace_c<conc>_<ts>/profile_export_aiperf.json
(mirrors gs://alisachen-models/perf/). Labels/validity come from RUN_LABELS below.
"""
import csv, glob, json, os, sys, datetime

GPUS = 72
# run-id prefix -> (arm, topology, router flags, protocol, validity note)
RUN_LABELS = {
  "1786437783": ("KV", "6P:12D", "kv defaults", "pre-native-loader (mooncake conv.), 1800s", "INVALID (0% wire-level reuse; pre-9c1284f)"),
  "1786437882": ("RR", "6P:12D", "round-robin", "pre-native-loader (mooncake conv.), 1800s", "INVALID (pre-9c1284f)"),
  "1786465823": ("RR", "6P:12D", "round-robin", "native loader, 900s warmup, 1800s; pre-rebuild cluster", "pre-rebuild (Aug 11); GDR path unknown"),
  "1786466333": ("KV", "6P:12D", "kv defaults", "native loader, 900s warmup, 1800s; pre-rebuild cluster", "pre-rebuild (Aug 11); GDR path unknown"),
  "1786944977": ("RR", "6P:12D", "round-robin", "ladder v1 (single frontend), 1800s/pt", "post-rebuild, GDR bypass ON"),
  "1786955823": ("KV", "6P:12D", "kv defaults", "ladder v1 (single frontend), 1800s/pt; c384 missing (job Error: all reqs failed)", "post-rebuild, GDR bypass ON; router-state stall suspected"),
  "1786970675": ("KV", "6P:12D", "kv + --router-prefill-load-scale 2.0", "flag sweep, warm workers, 900s", "frontend RS rev 4 @12:37Z"),
  "1786972787": ("KV", "6P:12D", "kv + --router-prefill-load-scale 3.0", "flag sweep, warm workers, 900s", "frontend RS rev 5 @13:07Z"),
  "1786974821": ("KV", "6P:12D", "kv + --router-kv-overlap-score-credit 0.8", "flag sweep, warm workers, 900s", "frontend RS rev 6 @13:44Z"),
  "1786976717": ("KV", "6P:12D", "kv + credit 1.0 + --router-kv-overlap-score-credit-decay 0.8", "flag sweep, warm workers, 900s", "frontend RS rev 7 @14:17Z"),
  "1786978690": ("KV", "6P:12D", "kv + --router-temperature 0.5", "flag sweep, warm workers, 900s", "frontend RS rev 8 @14:50Z"),
  "1786980665": ("KV", "6P:12D", "kv defaults (temp 0 / fcfs)", "flag sweep, warm workers, 900s", "defaults re-run; RS d54987997 re-adopted as rev 9 (inferred)"),
  "1786992066": ("KV", "6P:12D", "kv defaults", "ladder v2: fresh frontend per point, 300s settle, 1800s", "CANONICAL"),
  "1786995068": ("KV", "6P:12D", "kv defaults", "ladder v2: fresh frontend per point, 300s settle, 1800s", "CANONICAL"),
  "1786997955": ("KV", "6P:12D", "kv defaults", "ladder v2: fresh frontend per point, 300s settle, 1800s", "CANONICAL"),
  "1787002681": ("RR", "6P:12D", "round-robin", "down-sweep: fresh frontend per point, 240s settle, 1800s", "CANONICAL"),
  "1787006166": ("RR", "6P:12D", "round-robin", "down-sweep: fresh frontend per point, 240s settle, 1800s", "CANONICAL"),
  "1787008942": ("RR", "6P:12D", "round-robin", "down-sweep: fresh frontend per point, 240s settle, 1800s", "CANONICAL"),
  "1787011705": ("RR", "6P:12D", "round-robin", "down-sweep: fresh frontend per point, 240s settle, 1800s", "CANONICAL"),
  "1787043438": ("KV", "9P:8D", "kv defaults", "9x9 topology probe (decode at 8 replicas), 900s", "exploratory"),
  "1787072770": ("KV", "9P:8D", "kv defaults", "9x9 topology probe (decode at 8 replicas), 1800s", "exploratory"),
}

def g(d, k, s="avg"):
    v = (d.get(k) or {}).get(s)
    return v

def main(root):
    w = csv.writer(sys.stdout)
    w.writerow(["run", "start_utc", "arm", "topology", "router", "conc", "dur_s", "requests", "good_requests",
                "out_tok_s", "out_tok_s_per_gpu", "req_s", "ttft_p50_s", "ttft_p90_s", "ttft_p95_s", "ttft_p99_s",
                "itl_avg_ms", "isl_avg", "osl_avg", "errors", "protocol", "validity"])
    for f in sorted(glob.glob(os.path.join(root, "*", "*", "profile_export_aiperf.json"))):
        run = f.split(os.sep)[-3]; sub = f.split(os.sep)[-2]
        d = json.load(open(f))
        epoch = int(run.split("_")[0]); conc = int(sub.split("_c")[1].split("_")[0])
        arm, topo, router, proto, valid = RUN_LABELS.get(run.split("_")[0], ("?", "?", "?", "?", "unlabeled"))
        errs = d.get("error_summary") or []
        errn = sum(x.get("count", 0) for x in errs) if isinstance(errs, list) else errs
        tput = g(d, "output_token_throughput") or 0
        w.writerow([run, datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"),
                    arm, topo, router, conc, round(g(d, "benchmark_duration") or 0), g(d, "request_count"), g(d, "good_request_count"),
                    round(tput), round(tput / GPUS, 1), round(g(d, "request_throughput") or 0, 2),
                    *[round((g(d, "time_to_first_token", p) or 0) / 1000, 2) for p in ("p50", "p90", "p95", "p99")],
                    round(g(d, "inter_token_latency") or 0, 1), round(g(d, "input_sequence_length") or 0), round(g(d, "output_sequence_length") or 0),
                    errn, proto, valid])

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/silicon/disagg72")
