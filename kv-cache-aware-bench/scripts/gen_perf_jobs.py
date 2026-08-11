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

FULL_TRACE = "/model-cache/traces/weka_256k_bench4k.jsonl"  # 4k-request interleaved slice: full trace costs ~4h of aiperf prompt synthesis per job

# arm -> (dgd name, concurrencies)
ARMS = {
    "aggtp8-rr": ("kimi-k25-aggtp8-rr", "8,16,24,32,48"),
    "aggtp8-kvt": ("kimi-k25-aggtp8-kvt", "8,16,24,32,48"),
    "agg1n-rr": ("kimi-k25-agg1n-rr", "32,48,64,96"),
    "agg1n-kvt": ("kimi-k25-agg1n-kvt", "32,48,64,96"),
    "arm1a-agg-eagle-rr": ("kimi-k25-agg-rr-eagle", "24"),
    "arm1a-agg-eagle-kv": ("kimi-k25-agg-kv-eagle", "24"),
    "arm1b-agg-rr": ("kimi-k25-agg-rr", "8"),
    "arm1c-agg-kv-nvda": ("kimi-k25-agg-kv-nvda", "8"),
    "arm1d-agg-kv-tuned": ("kimi-k25-agg-kv-tuned", "8"),
    "arm2b-disagg-rr": ("kimi-k25-disagg-rr", "32,48,64,96,128,192,256"),
    "arm2c-disagg-kv-nvda": ("kimi-k25-disagg-kv-nvda", "32,48,64,96,128,192,256"),
    "arm2d-disagg-kv-tuned": ("kimi-k25-disagg-kv-tuned", "32,48,64,96,128,192,256"),
    # 72-GPU disagg (18x TP4, full pool) — selected cells 6:12/384 + 12:6/192 ladder
    "disagg72-rr": ("kimi-k25-disagg-rr", "96,192,288,384"),
    "disagg72-kvt": ("kimi-k25-disagg-kv-tuned", "96,192,288,384"),
    # sglang arms (selected points): disagg72 headline ladder; agg peak bracket
    "sgl-disagg72-rr": ("sgl-disagg72-rr", "96,192,288,384"),
    "sgl-disagg72-kv": ("sgl-disagg72-kv", "96,192,288,384"),
    "sgl-agg-rr": ("sgl-agg-rr", "8,16,24,32"),
    "sgl-agg-kv": ("sgl-agg-kv", "8,16,24,32"),
}

# smoke jobs: (suffix, base arm, trace file, concurrency, duration_s)
SMOKES = [
    ("smoke2-mini-e2e", "arm2b-disagg-rr", "/model-cache/traces/smoke_tiny.jsonl", "2", "600"),
    ("smoke3-short-replay", "arm2b-disagg-rr", "/model-cache/traces/smoke_short.jsonl", "32", "900"),
]

base = RECIPE_PERF.read_text()


def make_job(job_name, dgd, trace, conc, duration):
    t = base
    # tokenizer handling (aiperf 0.12.0): its dataset-decode workers force
    # HF_HUB_OFFLINE, and the offline branch (_resolve_local_snapshot) only
    # accepts HF-cache repo ids — a filesystem --tokenizer path crashes with
    # HFValidationError. Workaround: stage the tokenizer files from the PVC
    # into a local HF cache under the served-model id and run fully offline.
    t = t.replace('--tokenizer "${TARGET_MODEL}"', '--tokenizer "${TARGET_MODEL}"'
                  )  # keep repo-id form; resolution comes from the offline cache
    t = t.replace(
        "wait_for_model_ready\n",
        "wait_for_model_ready\n\n"
        "          export HF_HOME=/tmp/hf\n"
        "          CACHE=/tmp/hf/hub/models--alisachen--Kimi-K2.5-NVFP4\n"
        "          mkdir -p $CACHE/snapshots/local $CACHE/refs\n"
        "          cp /model-cache/alisachen/Kimi-K2.5-NVFP4/*.json "
        "/model-cache/alisachen/Kimi-K2.5-NVFP4/*.py "
        "/model-cache/alisachen/Kimi-K2.5-NVFP4/*.jinja "
        "/model-cache/alisachen/Kimi-K2.5-NVFP4/*.model $CACHE/snapshots/local/ 2>/dev/null || true\n"
        "          printf local > $CACHE/refs/main\n"
        "          export HF_HUB_OFFLINE=1\n"
        "          ls $CACHE/snapshots/local/ | head -20\n",
        1)
    t = t.replace("value: nvidia/Kimi-K2.5-NVFP4", "value: alisachen/Kimi-K2.5-NVFP4")
    # our trace uses the dataset's native 64-token hash blocks; aiperf's
    # mooncake loader defaults to 512 and rejects the trace without this.
    # needs aiperf >= 0.12.0 — in 0.10.0 (recipe pin) --isl-block-size is
    # synthetic-only and aborts when combined with --input-file
    t = t.replace('pip install "aiperf==0.10.0"', 'pip install "aiperf==0.12.0" tiktoken blobfile')
    t = t.replace("--custom-dataset-type mooncake_trace \\",
                  "--custom-dataset-type mooncake_trace \\\n              --isl-block-size 64 \\\n              --no-fixed-schedule \\\n              --ignore-trace-delays \\")
    # model-cache is gcsfuse: sidecar injection annotation required
    t = t.replace("      labels:\n        app: kimi-k25-agg-rr-bench",
                  "      labels:\n        app: kimi-k25-agg-rr-bench\n"
                  "      annotations:\n        gke-gcsfuse/volumes: \"true\"\n"
                  "        gke-gcsfuse/memory-limit: 4Gi\n"
                  "        gke-gcsfuse/cpu-limit: \"2\"")
    t = t.replace("activeDeadlineSeconds: 7200", "activeDeadlineSeconds: 28800")  # 3-conc sweep ~3.5h + warmup
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
    p.write_text(make_job(f"alisachen-{dgd}-bench", dgd, FULL_TRACE, conc, "1800"))
    print(f"wrote {p.name} (endpoint {dgd}-frontend:8000, conc {conc})")

for suffix, arm, trace, conc, duration in SMOKES:
    dgd, _ = ARMS[arm]
    p = OUT / f"{suffix}.yaml"
    p.write_text(make_job(f"alisachen-{dgd}-{suffix}", dgd, trace, conc, duration))
    print(f"wrote {p.name} (trace {trace}, conc {conc}, {duration}s)")
