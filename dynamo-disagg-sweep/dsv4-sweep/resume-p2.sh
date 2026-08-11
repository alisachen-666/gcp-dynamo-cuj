#!/usr/bin/env bash
# Resume p2: server standing + registered (runner was killed by the shell-less etcd gate bug).
# Runs: frontend restart -> hard gate -> filler pass -> record pass -> extract/assert/upload/teardown.
set -uo pipefail
export PATH=$HOME/google-cloud-sdk/bin:$PATH
D=$HOME/dsr1-pareto/dsv4-sweep
DEST="gs://gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs"
S=$(mktemp -d); PID=p2
kubectl rollout restart "deployment/dsv4-$PID-frontend" -n dynamo-cloud >/dev/null
kubectl rollout status "deployment/dsv4-$PID-frontend" -n dynamo-cloud --timeout=420s >/dev/null
ok=0
for i in $(seq 1 60); do
  r=$(kubectl run "probe-$PID-r$i" -n dynamo-cloud --image=curlimages/curl --restart=Never --rm -q -i --timeout=180s -- \
      -s -m 120 -X POST "http://dsv4-$PID-frontend.dynamo-cloud.svc.cluster.local:8000/v1/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"deepseek-ai/DeepSeek-V4-Pro","prompt":"hi","max_tokens":4,"stream":false}' 2>/dev/null)
  if echo "$r" | grep -q '"text"'; then ok=$((ok+1)); else ok=0; fi
  [ "$ok" -ge 3 ] && { echo "=== serving verified"; break; }
  sleep 20
done
[ "$ok" -ge 3 ] || { echo "!!! not serving — aborting"; exit 1; }
for pass in 1 2; do
  [ "$pass" = 1 ] && TAG=FILLER || TAG=RECORD
  echo "=== bench pass $pass ($TAG)"
  kubectl delete job "dsv4-bench-$PID" -n dynamo-cloud --ignore-not-found >/dev/null 2>&1
  for i in $(seq 1 30); do kubectl get job "dsv4-bench-$PID" -n dynamo-cloud >/dev/null 2>&1 || break; sleep 5; done
  kubectl apply -f "$D/manifests/bench-$PID.yaml" >/dev/null
  for i in $(seq 1 240); do
    st=$(kubectl get job "dsv4-bench-$PID" -n dynamo-cloud -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null)
    [ -n "$st" ] && break; sleep 60
  done
  echo "=== pass $pass terminal: ${st:-TIMEOUT}"
  kubectl logs -n dynamo-cloud "job/dsv4-bench-$PID" > "$S/bench-$PID-$TAG.log" 2>/dev/null
done
python3 - "$PID" "$S/bench-$PID-RECORD.log" <<'PYEOF'
import json, re, sys, os
pid, logf = sys.argv[1], sys.argv[2]
log = open(logf, errors='replace').read()
out = os.path.expanduser('~/dsr1-pareto/dsv4-sweep/results'); os.makedirs(out, exist_ok=True)
found = 0
pats = [r'--- /tmp/results/(results_concurrency_\d+_gpus_\d+_ctx_\d+_gen_\d+)\.json\n(\{.*?\})\n',
        r'=== SLIM-RESULT /tmp/results/(results_concurrency_\d+_gpus_\d+_ctx_\d+_gen_\d+)\.json (\{.*\})']
seen = set()
for pat in pats:
    for m in re.finditer(pat, log, re.S if '---' in pat else 0):
        if m.group(1) in seen: continue
        seen.add(m.group(1)); found += 1
        d = json.loads(m.group(2))
        json.dump(d, open(f'{out}/{pid}-{m.group(1)}.json', 'w'))
        g = re.search(r'gpus_(\d+)_ctx_(\d+)_gen_(\d+)', m.group(1)); tot, pg, dg = map(int, g.groups())
        print(f"{pid} conc {d['max_concurrency']}: out/decode-gpu {d['output_throughput']/dg:.1f} | in/prefill-gpu {d['total_input_tokens']/d['duration']/pg:.0f} | TPOT {d['mean_tpot_ms']:.2f}ms | TTFT med {d['median_ttft_ms']/1000:.1f}s")
print(f"[{pid}] extracted {found} result(s)")
PYEOF
mkdir -p "$S/srv"; for p in $(kubectl get pods -n dynamo-cloud --no-headers | grep "dsv4-$PID" | awk '{print $1}'); do kubectl logs -n dynamo-cloud "$p" > "$S/srv/$p.log" 2>/dev/null; done
export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token 2>/dev/null)
gcloud storage rsync -r "$S/srv" "$DEST/server-logs/dsv4-$PID" 2>&1 | tail -1
for fl in "$S"/bench-$PID*.log; do gcloud storage cp "$fl" "$DEST/client-logs/dsv4-$(basename "$fl")" 2>&1 | tail -1; done
gcloud storage rsync -r "$D/results" "$DEST/results/dsv4" 2>&1 | tail -1
echo "=== tearing down $PID"
kubectl delete -n dynamo-cloud "deployment/dsv4-$PID-frontend" "deployment/dsv4-$PID-prefill" "deployment/dsv4-$PID-decode" >/dev/null 2>&1
kubectl delete -n dynamo-cloud "statefulset/dsv4-$PID-decode" "computedomain/dsv4-$PID-cd" "service/dsv4-$PID-frontend" "service/dsv4-$PID-decode" "job/dsv4-bench-$PID" >/dev/null 2>&1
rm -rf "$S"
echo "=== $PID DONE"
