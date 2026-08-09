import json
import os

CONFIGS = [
    ("24-GPU Aggregated", "metrics_agg_24gpu.json"),
    ("24-GPU Disaggregated", "metrics_disagg_24gpu.json"),
    ("72-GPU Disaggregated (Sim)", "metrics_disagg_72gpu_sim.json"),
    ("72-GPU Disaggregated (Live)", "metrics_disagg_72gpu_live.json"),
]

ARMS = [
    ("nvda_recipe", "NVDA Recipe Baseline"),
    ("weka_round_robin", "Weka Trace / Round-Robin"),
    ("weka_kv_nvda", "Weka Trace / KV-Aware (NVDA Recipe Defaults)"),
    ("weka_kv_tuned", "Weka Trace / KV-Aware (Self-Tuned Strategy C)"),
    ("weka_kv", "Weka Trace / KV-Aware"),  # legacy single-KV key (72-GPU runs)
]


def load_metrics():
    results = {}
    for label, path in CONFIGS:
        if os.path.exists(path):
            results[label] = json.load(open(path))
        else:
            print(f"WARNING: {path} not found, skipping {label}")
    return results


def print_summary_table(results):
    print("| Configuration | Benchmark Type | KV Hit Rate (%) | TTFT p50 (ms) | TTFT p99 (ms) | Tok/s/GPU |")
    print("|---|---|---|---|---|---|")
    for label, _ in CONFIGS:
        metrics = results.get(label)
        if not metrics:
            continue
        for arm_key, arm_label in ARMS:
            m = metrics.get(arm_key)
            if not m:
                continue
            print(f"| {label} | {arm_label} | {m.get('kv_hit_rate')}% | "
                  f"{m.get('ttft_p50')} | {m.get('ttft_p99')} | {m.get('tps_gpu')} |")


def pct_gain(kv, rr, lower_is_better=False):
    if kv is None or rr is None or rr == 0:
        return "n/a"
    delta = (rr - kv) / rr * 100 if lower_is_better else (kv - rr) / rr * 100
    return f"{delta:+.1f}%"


GAIN_PAIRS = [
    ("weka_kv_nvda", "weka_round_robin", "KV-NVDA vs. Round-Robin"),
    ("weka_kv_tuned", "weka_round_robin", "KV-Tuned vs. Round-Robin"),
    ("weka_kv_tuned", "weka_kv_nvda", "KV-Tuned vs. KV-NVDA"),
    ("weka_kv", "weka_round_robin", "KV-Aware vs. Round-Robin"),
]


def print_perf_gain_matrix(results):
    print()
    print("## Routing Perf Gain (same topology, same trace)")
    print("| Configuration | Comparison | Δ TTFT p50 | Δ TTFT p99 | Δ Tok/s/GPU | Δ KV Hit Rate (pp) |")
    print("|---|---|---|---|---|---|")
    for label, _ in CONFIGS:
        metrics = results.get(label)
        if not metrics:
            continue
        for test_key, base_key, comparison in GAIN_PAIRS:
            test, base = metrics.get(test_key), metrics.get(base_key)
            if not test or not base:
                continue
            hit_delta = "n/a"
            if test.get("kv_hit_rate") is not None and base.get("kv_hit_rate") is not None:
                hit_delta = f"{test['kv_hit_rate'] - base['kv_hit_rate']:+.1f}"
            print(f"| {label} | {comparison} "
                  f"| {pct_gain(test.get('ttft_p50'), base.get('ttft_p50'), lower_is_better=True)} "
                  f"| {pct_gain(test.get('ttft_p99'), base.get('ttft_p99'), lower_is_better=True)} "
                  f"| {pct_gain(test.get('tps_gpu'), base.get('tps_gpu'))} "
                  f"| {hit_delta} |")


def print_sim_vs_live_delta(results):
    sim = results.get("72-GPU Disaggregated (Sim)")
    live = results.get("72-GPU Disaggregated (Live)")
    if not sim or not live:
        return
    print()
    print("## DynoSim vs. Live Delta at 72 GPUs (simulator fidelity)")
    print("| Routing | Metric | Sim | Live | Delta |")
    print("|---|---|---|---|---|")
    for arm_key, arm_label in [("weka_round_robin", "Round-Robin"), ("weka_kv", "KV-Aware")]:
        s, l = sim.get(arm_key), live.get(arm_key)
        if not s or not l:
            continue
        for metric in ("ttft_p50", "ttft_p99", "tps_gpu", "kv_hit_rate"):
            sv, lv = s.get(metric), l.get(metric)
            delta = f"{(sv - lv) / lv * 100:+.1f}%" if sv is not None and lv else "n/a"
            print(f"| {arm_label} | {metric} | {sv} | {lv} | {delta} |")


def generate_summary():
    results = load_metrics()
    print_summary_table(results)
    print_perf_gain_matrix(results)
    print_sim_vs_live_delta(results)


if __name__ == "__main__":
    generate_summary()
