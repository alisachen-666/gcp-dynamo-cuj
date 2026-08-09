"""Convert semianalysisai/cc-traces-weka-062126-256k traces.jsonl (streamed on
stdin, one session per line) into a flat per-request replay trace.

Usage: curl -sL <traces.jsonl url> | python3 ingest_weka_trace.py <out.jsonl>
Stdlib only — no `datasets` dependency, nothing buffered beyond one session.
"""
import json
import sys

out_path = sys.argv[1] if len(sys.argv) > 1 else "weka_256k_trace.jsonl"

n_requests = 0
n_sessions = 0
with open(out_path, "w") as out:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        sid = str(row["id"])
        block_size = row.get("block_size", 64)
        reqs = row["requests"]
        if isinstance(reqs, str):
            reqs = json.loads(reqs)
        # hash_id_scope is "local" (ids only unique within a session), so
        # keep requests grouped by session and ordered by offset time.
        reqs = sorted(
            (json.loads(r) if isinstance(r, str) else r for r in reqs),
            key=lambda r: r.get("t", 0.0),
        )
        for req in reqs:
            out.write(json.dumps({
                "session_id": sid,
                "timestamp_delay": req.get("t", 0.0),
                "model": req.get("model"),
                "input_length": req.get("in"),
                "output_length": req.get("out"),
                "hash_ids": req.get("hash_ids", []),
                "block_size": block_size,
            }) + "\n")
            n_requests += 1
        n_sessions += 1

print(f"Ingested {n_requests} requests across {n_sessions} sessions into {out_path}")
