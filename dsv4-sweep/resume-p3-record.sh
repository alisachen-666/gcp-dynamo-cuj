#!/usr/bin/env bash
# Resume p3 after the killed runner: the deployment is standing and the cache-filler bench
# pass completed, so this runs ONLY the record pass + extract + assert + upload + teardown.
set -uo pipefail
export PATH=$HOME/google-cloud-sdk/bin:$PATH
D=$HOME/dsr1-pareto/dsv4-sweep
DEST="gs://REDACTED-GCS-BUCKET/mlperf_results/nvfp4_20260519_run5/logs/logs"
S=$(mktemp -d)
PID=p3

echo "=== record pass (server standing, filler pass already completed)"
kubectl delete job "dsv4-bench-$PID" -n dynamo-cloud --ignore-not-found >/dev/null 2>&1
for i in $(seq 1 30); do kubectl get job "dsv4-bench-$PID" -n dynamo-cloud >/dev/null 2>&1 || break; sleep 5; done
kubectl apply -f "$D/manifests/bench-$PID.yaml" >/dev/null
for i in $(seq 1 240); do
  st=$(kubectl get job "dsv4-bench-$PID" -n dynamo-cloud -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null)
  [ -n "$st" ] && break
  sleep 60
done
echo "=== record bench terminal: ${st:-TIMEOUT}"
kubectl logs -n dynamo-cloud "job/dsv4-bench-$PID" > "$S/bench-$PID.log" 2>/dev/null

python3 - "$PID" "$S/bench-$PID.log" <<'EOF'
import json, re, sys, os
pid, logf = sys.argv[1], sys.argv[2]
log = open(logf, errors='replace').read()
out = os.path.expanduser('~/dsr1-pareto/dsv4-sweep/results')
os.makedirs(out, exist_ok=True)
found = 0
# primary: full JSON blocks; fallback: rotation-proof SLIM-RESULT tail lines
for m in re.finditer(r'--- /tmp/results/(results_concurrency_\d+_gpus_\d+_ctx_\d+_gen_\d+)\.json\n(\{.*?\})\n', log, re.S):
    d = json.loads(m.group(2)); found += 1
    json.dump(d, open(f'{out}/{pid}-{m.group(1)}.json', 'w'))
if not found:
    for m in re.finditer(r'=== SLIM-RESULT /tmp/results/(results_concurrency_\d+_gpus_\d+_ctx_\d+_gen_\d+)\.json (\{.*\})', log):
        d = json.loads(m.group(2)); found += 1
        json.dump(d, open(f'{out}/{pid}-{m.group(1)}.slim.json', 'w'))
        print('(recovered via SLIM-RESULT tail line)')
for f in os.listdir(out):
    if f.startswith(f'{pid}-results_concurrency'):
        d = json.load(open(os.path.join(out, f)))
        g = re.search(r'gpus_(\d+)_ctx_(\d+)_gen_(\d+)', f); tot, pg, dg = map(int, g.groups())
        print(f"{pid} conc {d['max_concurrency']}: out/decode-gpu {d['output_throughput']/dg:.1f} | "
              f"in/prefill-gpu {d['total_input_tokens']/d['duration']/pg:.0f} | total/gpu {d['total_token_throughput']/tot:.1f} | "
              f"TPOT {d['mean_tpot_ms']:.2f}ms | TTFT med {d['median_ttft_ms']/1000:.1f}s | TTFT p99 {d['p99_ttft_ms']/1000:.1f}s")
print(f"[{pid}] extracted {found} result(s)")
EOF

mkdir -p "$S/srv"; for p in $(kubectl get pods -n dynamo-cloud --no-headers | grep "dsv4-$PID" | awk '{print $1}'); do kubectl logs -n dynamo-cloud "$p" > "$S/srv/$p.log" 2>/dev/null; done

python3 - "$PID" "$S/bench-$PID.log" "$S/srv" <<'EOF'
import datetime, glob, re, sys
pid, benchlog, srvdir = sys.argv[1], sys.argv[2], sys.argv[3]
log = open(benchlog, errors='replace').read()
windows = []
for m in re.finditer(r'"date": "(\d{8}-\d{6})".*?"duration": ([0-9.]+)', log):
    end = datetime.datetime.strptime(m.group(1), '%Y%m%d-%H%M%S')
    windows.append((end - datetime.timedelta(seconds=float(m.group(2))), end))
ansi = re.compile(r'\x1b\[[0-9;]*m')
verdict = 'VALID'
for f in glob.glob(f'{srvdir}/*prefill*.log'):
    jits, batches = [], []
    for line in open(f, errors='replace'):
        line = ansi.sub('', line)
        m = re.match(r'(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+)Z', line)
        if not m: continue
        t = datetime.datetime.fromisoformat(m.group(1))
        if 'Entering DeepGEMM JIT' in line: jits.append(t)
        elif 'report_prefill_stats' in line: batches.append(t)
    for (a, b) in windows:
        for t in jits:
            if a <= t <= b: print(f"!!! ASSERT: JIT at {t.time()} inside {a.time()}-{b.time()}"); verdict = 'SUSPECT'
        w = [t for t in batches if a <= t <= b]
        for x, y in zip(w, w[1:]):
            if (y - x).total_seconds() > 5: print(f"!!! ASSERT: prefill silent {(y-x).total_seconds():.1f}s"); verdict = 'SUSPECT'
print(f"[{pid}] windows checked: {len(windows)}; RECORD VERDICT: {verdict}")
EOF

export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token 2>/dev/null)
gcloud storage rsync -r "$S/srv" "$DEST/server-logs/dsv4-$PID-rerecord" 2>&1 | tail -1
gcloud storage cp "$S/bench-$PID.log" "$DEST/client-logs/dsv4-bench-$PID-rerecord.log" 2>&1 | tail -1
gcloud storage rsync -r "$D/results" "$DEST/results/dsv4" 2>&1 | tail -1
echo "=== tearing down $PID"
kubectl delete -n dynamo-cloud "deployment/dsv4-$PID-frontend" "deployment/dsv4-$PID-prefill" >/dev/null 2>&1
kubectl delete -n dynamo-cloud "statefulset/dsv4-$PID-decode" >/dev/null 2>&1
kubectl delete -n dynamo-cloud "computedomain/dsv4-$PID-cd" "service/dsv4-$PID-frontend" "service/dsv4-$PID-decode" "job/dsv4-bench-$PID" >/dev/null 2>&1
rm -rf "$S"
echo "=== $PID RECORD DONE"
