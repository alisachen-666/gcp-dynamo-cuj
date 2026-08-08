#!/usr/bin/env python3
"""Write git-friendly summary copies of sa-bench result JSONs.

Raw results contain per-request/per-token arrays (ttfts, itls, generated_texts...)
which reach hundreds of MB. This strips them but PRE-COMPUTES the p90/p95
percentiles the report generators derive from those arrays, so reports remain
fully reproducible from the slim copies. Raw JSONs stay out of git (archived in
gs://REDACTED-GCS-BUCKET/mlperf_results/.../results/).
"""
import json, glob, os

BULK = {'ttfts', 'itls', 'input_lens', 'output_lens', 'generated_texts', 'errors',
        'tpots', 'e2els', 'reasoning_lens', 'ttft_per_request', 'request_ids'}
PAIRS = [('ttfts', 'ttft'), ('itls', 'itl')]

def pctile(arr, p):
    flat = []
    for x in arr:
        flat.extend(x) if isinstance(x, list) else flat.append(x)
    s = sorted(v * 1000.0 for v in flat)
    if not s:
        return None
    k = (len(s) - 1) * p / 100.0
    f = int(k); c = min(f + 1, len(s) - 1)
    return round(s[f] + (s[c] - s[f]) * (k - f), 3)

def slim(src, dst):
    d = json.load(open(src))
    for key, name in PAIRS:
        if isinstance(d.get(key), list) and d[key]:
            for p in (90, 95):
                v = pctile(d[key], p)
                if v is not None:
                    d[f'p{p}_{name}_ms'] = v
    out = {k: v for k, v in d.items() if k not in BULK}
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(out, open(dst, 'w'), indent=1, sort_keys=True)
    return os.path.getsize(src), os.path.getsize(dst)

if __name__ == '__main__':
    root = os.path.expanduser('~/dsr1-pareto')
    jobs = [(f'{root}/dsr1-sweep/results', f'{root}/dsr1-sweep/results-summary'),
            (f'{root}/dsv4-sweep/results', f'{root}/dsv4-sweep/results-summary'),
            (f'{root}/dsr1-fp8-sweep/results', f'{root}/dsr1-fp8-sweep/results-summary')]
    tot_in = tot_out = n = 0
    for srcdir, dstdir in jobs:
        for src in glob.glob(f'{srcdir}/**/*.json', recursive=True):
            rel = os.path.relpath(src, srcdir)
            a, b = slim(src, os.path.join(dstdir, rel))
            tot_in += a; tot_out += b; n += 1
    print(f'{n} files: {tot_in/1048576:.0f} MB raw -> {tot_out/1024:.0f} KB summary')
