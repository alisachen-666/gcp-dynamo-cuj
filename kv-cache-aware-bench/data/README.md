# Trace data

- `weka_256k_trace.jsonl.gz` — canonical converted trace (source of truth, md5 in
  environment_lock.json).
- `smoke_tiny.jsonl.gz` / `smoke_short.jsonl.gz` — smoke-ladder slices.
- The AIPerf replay trace is NOT stored (interleaving breaks gzip locality → >100 MB);
  regenerate deterministically:
  ```
  python3 scripts/convert_to_aiperf_mooncake.py weka_256k_trace.jsonl aiperf.jsonl
  python3 scripts/interleave_trace.py weka_256k_trace.jsonl aiperf.jsonl weka_256k_aiperf.jsonl
  ```
  The live copy lives at gs://alisachen-models/traces/weka_256k_aiperf.jsonl.
