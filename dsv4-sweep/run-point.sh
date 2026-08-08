#!/usr/bin/env bash
# Run one DSv4 point end-to-end: deploy -> registrations -> frontend restart -> bench
# -> extract result -> upload artifacts -> teardown.
# Usage: run-point.sh <pid> <expected_workers>
set -uo pipefail
PID=$1
EXP=${2:-2}
D=$HOME/dsr1-pareto/dsv4-sweep
DEST="gs://REDACTED-GCS-BUCKET/mlperf_results/nvfp4_20260519_run5/logs/logs"
S=$(mktemp -d)

echo "=== deploying $PID"
kubectl apply -f "$D/manifests/$PID.yaml" >/dev/null
echo "=== waiting for $EXP worker registrations"
for i in $(seq 1 120); do
  n=$(kubectl exec -n dynamo-cloud dynamo-platform-etcd-0 -- sh -c 'ETCDCTL_API=3 etcdctl --endpoints=localhost:2379 get v1/instances/ --prefix --keys-only' 2>/dev/null | grep -c generate)
  [ "${n:-0}" -ge "$EXP" ] && break
  bad=$(kubectl get pods -n dynamo-cloud --no-headers 2>/dev/null | grep -E "dsv4-$PID-(prefill|decode)" | grep -cE "CrashLoop|Error" || true)
  if [ "$bad" != "0" ]; then
    echo "!!! $PID WORKER FAILURE"; kubectl get pods -n dynamo-cloud | grep "dsv4-$PID" | grep -E "CrashLoop|Error"
    kubectl logs -n dynamo-cloud "deploy/dsv4-$PID-prefill" --tail=20 2>/dev/null | grep -E "Error|Traceback" | head -5
    exit 1
  fi
  sleep 60
done
echo "=== $n registered; restarting frontends"
kubectl rollout restart "deployment/dsv4-$PID-frontend" -n dynamo-cloud >/dev/null
kubectl rollout status "deployment/dsv4-$PID-frontend" -n dynamo-cloud --timeout=420s >/dev/null
# HARD GATE: a real completion must succeed through the Service (not just /v1/models),
# repeated 3x consecutively so every frontend replica + worker is truly serving.
ok=0
for i in $(seq 1 60); do
  r=$(kubectl run "probe-$PID-$i" -n dynamo-cloud --image=curlimages/curl --restart=Never --rm -q -i --timeout=180s -- \
      -s -m 120 -X POST "http://dsv4-$PID-frontend.dynamo-cloud.svc.cluster.local:8000/v1/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"deepseek-ai/DeepSeek-V4-Pro","prompt":"hi","max_tokens":4,"stream":false}' 2>/dev/null)
  if echo "$r" | grep -q '"text"'; then ok=$((ok+1)); else ok=0; fi
  [ "$ok" -ge 3 ] && { echo "=== serving verified (3 consecutive completions)"; break; }
  sleep 20
done
[ "$ok" -ge 3 ] || { echo "!!! $PID never served completions — aborting point"; exit 1; }
echo "=== launching bench"
kubectl apply -f "$D/manifests/bench-$PID.yaml" >/dev/null
for i in $(seq 1 240); do
  st=$(kubectl get job "dsv4-bench-$PID" -n dynamo-cloud -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null)
  [ -n "$st" ] && break
  sleep 60
done
echo "=== bench terminal: ${st:-TIMEOUT}"
kubectl logs -n dynamo-cloud "job/dsv4-bench-$PID" > "$S/bench-$PID.log" 2>/dev/null
python3 - "$PID" "$S/bench-$PID.log" <<'EOF'
import json, re, sys, os
pid, logf = sys.argv[1], sys.argv[2]
log = open(logf).read()
out = os.path.expanduser('~/dsr1-pareto/dsv4-sweep/results')
found = 0
for m in re.finditer(r'--- /tmp/results/(results_concurrency_\d+_gpus_\d+_ctx_\d+_gen_\d+)\.json\n(\{.*?\})\n', log, re.S):
    d = json.loads(m.group(2)); found += 1
    json.dump(d, open(f'{out}/{pid}-{m.group(1)}.json', 'w'))
    g = re.search(r'gpus_(\d+)_ctx_(\d+)_gen_(\d+)', m.group(1)); tot, pg, dg = map(int, g.groups())
    intps = d['total_input_tokens']/d['duration']
    print(f"{pid} conc {d['max_concurrency']}: out/decode-gpu {d['output_throughput']/dg:.1f} | in/prefill-gpu {intps/pg:.0f} | total/gpu {d['total_token_throughput']/tot:.1f} | TPOT {d['mean_tpot_ms']:.2f}ms | TTFT med {d['median_ttft_ms']/1000:.1f}s")
print(f"[{pid}] extracted {found} result(s)")
EOF
mkdir -p "$S/srv"; for p in $(kubectl get pods -n dynamo-cloud --no-headers | grep "dsv4-$PID" | awk '{print $1}'); do kubectl logs -n dynamo-cloud "$p" > "$S/srv/$p.log" 2>/dev/null; done
gcloud storage rsync -r "$S/srv" "$DEST/server-logs/dsv4-$PID" 2>&1 | tail -1
gcloud storage cp "$S/bench-$PID.log" "$DEST/client-logs/dsv4-bench-$PID.log" 2>&1 | tail -1
gcloud storage rsync -r "$D/results" "$DEST/results/dsv4" 2>&1 | tail -1
echo "=== tearing down $PID"
kubectl delete -n dynamo-cloud "deployment/dsv4-$PID-frontend" "deployment/dsv4-$PID-prefill" >/dev/null 2>&1
kubectl delete -n dynamo-cloud "statefulset/dsv4-$PID-decode" >/dev/null 2>&1
kubectl delete -n dynamo-cloud "computedomain/dsv4-$PID-cd" "service/dsv4-$PID-frontend" "service/dsv4-$PID-decode" "job/dsv4-bench-$PID" >/dev/null 2>&1
rm -rf "$S"
echo "=== $PID DONE"
