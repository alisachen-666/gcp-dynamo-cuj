"""Cut smoke-test slices from the AIPerf-mooncake Weka trace.

  smoke_tiny.jsonl  — 50 small requests (input <= 16k tokens), for the first
                      end-to-end request-path check at concurrency 2.
  smoke_short.jsonl — 3 complete sessions (preserves real prefix-reuse chains
                      and 100k+ contexts), for a ~10-15 min replay at target
                      concurrency before any multi-hour run.

Usage: python3 make_smoke_slices.py <weka_256k_aiperf.jsonl> <outdir>
"""
import json
import sys
from collections import OrderedDict

src, outdir = sys.argv[1], sys.argv[2]

tiny, sessions = [], OrderedDict()
with open(src) as f:
    for line in f:
        r = json.loads(line)
        sessions.setdefault(r["session_id"], []).append(r)
        if len(tiny) < 50 and r["input_length"] <= 16384:
            tiny.append(r)

# pick the 3 sessions closest to the median request count (representative, not degenerate)
by_len = sorted(sessions.values(), key=len)
mid = len(by_len) // 2
short = [r for s in by_len[mid - 1: mid + 2] for r in s]

for name, rows in [("smoke_tiny.jsonl", tiny), ("smoke_short.jsonl", short)]:
    with open(f"{outdir}/{name}", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ins = [r["input_length"] for r in rows]
    print(f"{name}: {len(rows)} requests, input p50={sorted(ins)[len(ins)//2]}, "
          f"max={max(ins)}, total_in={sum(ins)/1e6:.1f}M tok")
