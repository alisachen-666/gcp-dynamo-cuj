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
    # cache warmup: the recipe's warmup is 5 synthetic requests (CUDA graphs/JIT
    # only) — it never touches the trace, so point 1 would measure a cold prefix
    # cache and an empty router index. Replay the real trace for 900s at the
    # ladder's first concurrency first (artifacts discarded) so every measured
    # point sees steady-state caches, matching the sim's steady-state window.
    first_conc = conc.split(",")[0]
    # neutralize the recipe's synthetic warmup: with ignore_eos and no output cap
    # its 5 requests generate unbounded (measured 33 min/request). The trace
    # cache-warmup below is the real warmup.
    t = t.replace("# Warmup", "# Warmup (synthetic phase disabled; see cache warmup)\n          if false; then")
    t = t.replace('echo "Warmup complete"', 'fi\n          echo "Warmup complete"', 1)
    t = t.replace(
        'echo "Warmup complete"',
        'echo "Warmup complete"\n\n'
        '          CWARM_DIR="${ROOT_DIR}/cache-warmup"\n'
        '          mkdir -p "$CWARM_DIR"\n'
        '          aiperf profile \\\n'
        '            -m "${TARGET_MODEL}" \\\n'
        '            --tokenizer "${TARGET_MODEL}" \\\n'
        '            --tokenizer-trust-remote-code \\\n'
        '            --input-file "${TRACE_FILE}" \\\n'
        '            --custom-dataset-type mooncake_trace \\\n'
        '            --isl-block-size 64 \\\n'
        '            --no-fixed-schedule \\\n'
        '            --ignore-trace-delays \\\n'
        '            --url "http://${ENDPOINT}" \\\n'
        '            --streaming \\\n'
        '            --ui dashboard \\\n'
        '            --extra-inputs ignore_eos:true \\\n'
        f'            --concurrency {first_conc} \\\n'
        '            --benchmark-duration 900 \\\n'
        '            --benchmark-grace-period 60 \\\n'
        '            --request-timeout-seconds 1200 \\\n'
        '            --artifact-dir "$CWARM_DIR" || echo "cache warmup non-fatal failure"\n'
        '          echo "Cache warmup complete (trace replay, 900s)"', 1)
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
        "          cp -r /model-cache/hf-cache/hub/datasets--semianalysisai--cc-traces-weka-062126-256k /tmp/hf/hub/ 2>/dev/null "
        "&& echo 'weka hub snapshot staged' || echo 'WARN: hub snapshot missing'\n"
        "          mkdir -p /tmp/hf/datasets && cp -r /model-cache/hf-cache/datasets/* /tmp/hf/datasets/ 2>/dev/null "
        "&& echo 'weka processed cache staged' || echo 'WARN: processed cache missing'\n"
        "          export HF_HUB_OFFLINE=1\n"
        "          ls $CACHE/snapshots/local/ | head -20\n",
        1)
    # the trace-file existence gate is obsolete under --public-dataset
    t = t.replace('if [ ! -f "${TRACE_FILE}" ]; then', 'if false; then')
    t = t.replace("value: nvidia/Kimi-K2.5-NVFP4", "value: alisachen/Kimi-K2.5-NVFP4")
    t = t.replace('pip install "aiperf==0.10.0"', 'pip install "aiperf==0.12.0" tiktoken blobfile')
    # NATIVE weka loader (2026-08-11 pivot): our mooncake conversion hit aiperf's
    # non-deterministic block synthesis — with --workers-max N each worker process
    # synthesized DIFFERENT text for the same hash_id, so no two requests shared
    # a prefix on the wire (59 sampled requests -> 58 unique prefixes despite the
    # trace offering 84.6% block reuse). Silicon showed 0 cached tokens + router
    # predicted-hit 0 for 98% of requests while a manual repeated prompt hit
    # 1216/1234 cached. The SemiAnalysisCCTracesWeka loader keys block content by
    # (trace_id, hash_id) via HashIdRandomGenerator — deterministic cross-process.
    # Dataset pre-staged to /model-cache/hf-cache (pods run HF_HUB_OFFLINE=1).
    t = t.replace("--input-file '${TRACE_FILE}' \\", "--public-dataset semianalysis_cc_traces_weka_062126_256k \\")
    t = t.replace('--input-file "${TRACE_FILE}" \\', "--public-dataset semianalysis_cc_traces_weka_062126_256k \\")
    t = t.replace("--custom-dataset-type mooncake_trace \\",
                  "--no-fixed-schedule \\\n              --ignore-trace-delays \\\n"
                  "              --slice-duration 1.0 \\\n"
                  "              --max-context-length 262144 \\")
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
