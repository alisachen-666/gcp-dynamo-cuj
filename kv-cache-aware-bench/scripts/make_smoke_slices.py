"""Cut smoke-test slices from the AIPerf-mooncake Weka trace.

Session membership comes from the SOURCE trace (the aiperf file deliberately
drops session_id — aiperf would otherwise concatenate turn history); rows are
paired 1:1 with converted lines by replaying the converter's skip logic
(null-token rows are absent from the converted file).

  smoke_tiny.jsonl  — 50 small requests (input <= 16k tokens), conc-2 e2e check.
  smoke_short.jsonl — 3 complete sessions (real prefix-reuse chains, 100k+ ctx).

Usage: python3 make_smoke_slices.py <weka_256k_trace.jsonl> <weka_256k_aiperf.jsonl> <outdir>
"""
import json
import sys
from collections import OrderedDict

src, conv, outdir = sys.argv[1], sys.argv[2], sys.argv[3]

conv_lines = open(conv).read().splitlines()

tiny, sessions = [], OrderedDict()
ci = 0
with open(src) as f:
    for line in f:
        r = json.loads(line)
        if r["input_length"] is None or r["output_length"] is None:
            continue  # converter skipped this row: no converted line consumed
        cline = conv_lines[ci]
        ci += 1
        sessions.setdefault(r["session_id"], []).append(cline)
        if len(tiny) < 50 and r["input_length"] <= 16384:
            tiny.append(cline)

assert ci == len(conv_lines), f"row alignment broke: {ci} != {len(conv_lines)}"

# 3 sessions closest to the median request count (representative, not degenerate)
by_len = sorted(sessions.values(), key=len)
mid = len(by_len) // 2
short = [l for s in by_len[mid - 1: mid + 2] for l in s]

for name, rows in [("smoke_tiny.jsonl", tiny), ("smoke_short.jsonl", short)]:
    with open(f"{outdir}/{name}", "w") as f:
        f.write("\n".join(rows) + "\n")
    ins = [json.loads(l)["input_length"] for l in rows]
    print(f"{name}: {len(rows)} requests, input p50={sorted(ins)[len(ins)//2]}, "
          f"max={max(ins)}, total_in={sum(ins)/1e6:.1f}M tok")
