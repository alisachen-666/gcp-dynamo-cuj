"""Generate per-arm AIPerf benchmark Jobs (GKE) from the recipe's perf.yaml.

The recipe perf job is already Kubernetes-native; this parameterizes it per arm:
  - job name + frontend endpoint
  - trace file (full run vs smoke slices) via TRACE_FILE
  - concurrency via CONCURRENCIES
All jobs replay the AIPerf-mooncake Weka trace from the model-cache PVC.
"""
from pathlib import Path

RECIPE_PERF = (Path.home() /
               "kv-cache-aware-bench/recipes/recipes/kimi-k2.5/trtllm/agg-round-robin/perf.yaml")
OUT = Path.home() / "kv-cache-aware-bench/manifests/perf"

FULL_TRACE = "/model-cache/traces/weka_256k_aiperf.jsonl"

# arm -> (dgd name, concurrencies)
ARMS = {
    "arm1b-agg-rr": ("kimi-k25-agg-rr", "8"),
    "arm1c-agg-kv-nvda": ("kimi-k25-agg-kv-nvda", "8"),
    "arm1d-agg-kv-tuned": ("kimi-k25-agg-kv-tuned", "8"),
    "arm2b-disagg-rr": ("kimi-k25-disagg-rr", "32"),
    "arm2c-disagg-kv-nvda": ("kimi-k25-disagg-kv-nvda", "32"),
    "arm2d-disagg-kv-tuned": ("kimi-k25-disagg-kv-tuned", "32"),
}

# smoke jobs: (suffix, base arm, trace file, concurrency, duration_s)
SMOKES = [
    ("smoke2-mini-e2e", "arm2b-disagg-rr", "/model-cache/traces/smoke_tiny.jsonl", "2", "600"),
    ("smoke3-short-replay", "arm2b-disagg-rr", "/model-cache/traces/smoke_short.jsonl", "32", "900"),
]

base = RECIPE_PERF.read_text()


def make_job(job_name, dgd, trace, conc, duration):
    t = base
    # tokenizer from the PVC-staged checkpoint (gated repo would 401 without a token)
    t = t.replace('--tokenizer "${TARGET_MODEL}"',
                  '--tokenizer /model-cache/alisachen/Kimi-K2.5-NVFP4')
    t = t.replace("value: nvidia/Kimi-K2.5-NVFP4", "value: alisachen/Kimi-K2.5-NVFP4")
    # model-cache is gcsfuse: sidecar injection annotation required
    t = t.replace("      labels:\n        app: kimi-k25-agg-rr-bench",
                  "      labels:\n        app: kimi-k25-agg-rr-bench\n"
                  "      annotations:\n        gke-gcsfuse/volumes: \"true\"")
    t = t.replace("name: kimi-k25-agg-rr-bench", f"name: {job_name}")
    t = t.replace("app: kimi-k25-agg-rr-bench", f"app: {job_name}")
    t = t.replace("- kimi-k25-agg-rr", f"- {dgd}")  # antiaffinity DGD label value
    t = t.replace("value: kimi-k25-agg-rr-frontend:8000", f"value: {dgd}-frontend:8000")
    t = t.replace("value: /model-cache/traces/agent_trace_data/dataset.jsonl",
                  f"value: {trace}")
    t = t.replace('value: "8"\n        - name: BENCHMARK_DURATION',
                  f'value: "{conc}"\n        - name: BENCHMARK_DURATION')
    t = t.replace('- name: BENCHMARK_DURATION\n          value: "3600"',
                  f'- name: BENCHMARK_DURATION\n          value: "{duration}"')
    assert f"name: {job_name}" in t and f"{dgd}-frontend:8000" in t
    return t


OUT.mkdir(exist_ok=True)
for arm, (dgd, conc) in ARMS.items():
    p = OUT / f"{arm}-bench.yaml"
    p.write_text(make_job(f"{dgd}-bench", dgd, FULL_TRACE, conc, "3600"))
    print(f"wrote {p.name} (endpoint {dgd}-frontend:8000, conc {conc})")

for suffix, arm, trace, conc, duration in SMOKES:
    dgd, _ = ARMS[arm]
    p = OUT / f"{suffix}.yaml"
    p.write_text(make_job(f"{dgd}-{suffix}", dgd, trace, conc, duration))
    print(f"wrote {p.name} (trace {trace}, conc {conc}, {duration}s)")
