"""Convert weka_256k_trace.jsonl into an AIPerf mooncake_trace input file.

Critical transform: the source dataset has hash_id_scope=local (hash ids are
only unique within a session). AIPerf materializes equal hash_ids as identical
token blocks, so local ids replayed verbatim would fabricate cross-session
prefix sharing and inflate KV hit rates. We renumber (session_id, local_id) to
globally-unique ids, preserving within-session overlap exactly.

Usage: python3 convert_to_aiperf_mooncake.py <in.jsonl> <out.jsonl>
"""
import json
import sys

src, dst = sys.argv[1], sys.argv[2]

global_ids = {}  # (session_id, local_id) -> global id
n = skipped = 0
with open(src) as fin, open(dst, "w") as fout:
    for line in fin:
        r = json.loads(line)
        if r["input_length"] is None or r["output_length"] is None:
            skipped += 1  # non-replayable rows (no token counts) — AIPerf can't build them
            continue
        # engine rejects input+output > max_seq_len (262144); clamp replayed
        # output so near-max-context rows don't 400 (affects ~2% of requests)
        out_len = max(1, min(r["output_length"], 262144 - r["input_length"]))
        sid = r["session_id"]
        gids = []
        for h in r["hash_ids"]:
            key = (sid, h)
            if key not in global_ids:
                global_ids[key] = len(global_ids)
            gids.append(global_ids[key])
        # NO session_id in output: aiperf groups rows with session_id into
        # multi-turn conversations and CONCATENATES turn history — but our
        # input_length is already cumulative (full context per request), so
        # grouping double-counts history (254k rows ballooned to 290-374k
        # tokens + multi-MB payloads). Standalone rows replay correctly;
        # prefix reuse is still realized via identical hash_ids.
        fout.write(json.dumps({
            "timestamp": int(float(r["timestamp_delay"]) * 1000),
            "input_length": r["input_length"],
            "output_length": out_len,
            "hash_ids": gids,
        }) + "\n")
        n += 1

print(f"wrote {n} requests ({skipped} skipped: null token counts), "
      f"{len(global_ids)} globally-unique hash blocks -> {dst}")
