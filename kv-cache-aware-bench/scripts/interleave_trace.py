"""Interleave the AIPerf-mooncake trace round-robin across sessions.

Why: the converted trace is session-grouped (a session's turns are consecutive
rows). Closed-loop replay at concurrency C then keeps only ~C consecutive rows
in flight — a handful of sessions — so prefix caches never see realistic
pressure, same-session turns race in flight, and routing policies are
indistinguishable. The source dataset has no absolute arrival times, so we
model the standard agentic-serving arrival process: many independent sessions
advancing turn-by-turn — i.e., round-robin interleaving of session streams.
Turn order WITHIN each session is preserved exactly.

Usage: interleave_trace.py <weka_256k_trace.jsonl> <weka_256k_aiperf.jsonl> <out.jsonl>
"""
import json
import sys
from collections import OrderedDict

src, conv, out = sys.argv[1], sys.argv[2], sys.argv[3]
conv_lines = open(conv).read().splitlines()

sessions = OrderedDict()
ci = 0
with open(src) as f:
    for line in f:
        r = json.loads(line)
        if r["input_length"] is None or r["output_length"] is None:
            continue
        sessions.setdefault(r["session_id"], []).append(conv_lines[ci])
        ci += 1
assert ci == len(conv_lines), f"alignment broke: {ci} != {len(conv_lines)}"

queues = [list(v) for v in sessions.values()]
out_lines = []
i = 0
while queues:
    q = queues[i % len(queues)]
    out_lines.append(q.pop(0))
    if not q:
        queues.pop(i % len(queues))
        # keep position stable relative to remaining queues
        i -= 1
    i += 1

assert len(out_lines) == len(conv_lines)
with open(out, "w") as f:
    f.write("\n".join(out_lines) + "\n")
print(f"interleaved {len(out_lines)} rows across {len(sessions)} sessions -> {out}")
