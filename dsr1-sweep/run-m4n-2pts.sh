#!/usr/bin/env bash
# Deploy the m4n (mid_curve NATS) server once and bench conc 512 then 4096 against it,
# completing the NATS-parity sweep of mid_curve. Extraction handles both full-JSON and
# SLIM-RESULT tail lines (rotation-proof).
set -uo pipefail
D=$HOME/dsr1-pareto/dsr1-sweep
DEST="gs://REDACTED-GCS-BUCKET/mlperf_results/nvfp4_20260519_run5/logs/logs"
S=$(mktemp -d)
PFX="dsr1-m4n"

echo "=== deploying m4n (2-point pass: conc 512 + 4096)"
kubectl apply -f "$D/manifests/m4n-midcurve-nats.yaml" >/dev/null
for i in $(seq 1 120); do
  n=$(kubectl exec -n dynamo-cloud dynamo-platform-etcd-0 -- etcdctl --endpoints=localhost:2379 get v1/instances/ --prefix --keys-only 2>/dev/null | grep -c generate)
  [ "${n:-0}" -ge 7 ] && break
  bad=$(kubectl get pods -n dynamo-cloud --no-headers 2>/dev/null | grep -E "$PFX-(prefill|decode)" | grep -cE "CrashLoop|Error" || true)
  [ "$bad" != "0" ] && { echo "!!! WORKER FAILURE"; exit 1; }
  sleep 60
done
echo "=== $n registered; restarting frontends"
kubectl rollout restart "deployment/$PFX-frontend" -n dynamo-cloud >/dev/null
kubectl rollout status "deployment/$PFX-frontend" -n dynamo-cloud --timeout=420s >/dev/null
ok=0
for i in $(seq 1 60); do
  r=$(kubectl run "probe-m4n2-$i" -n dynamo-cloud --image=curlimages/curl --restart=Never --rm -q -i --timeout=180s -- \
      -s -m 120 -X POST "http://$PFX-frontend.dynamo-cloud.svc.cluster.local:8000/v1/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"deepseek-ai/DeepSeek-R1","prompt":"hi","max_tokens":4,"stream":false}' 2>/dev/null)
  if echo "$r" | grep -q '"text"'; then ok=$((ok+1)); else ok=0; fi
  [ "$ok" -ge 3 ] && { echo "=== serving verified"; break; }
  sleep 20
done
[ "$ok" -ge 3 ] || { echo "!!! never served — aborting"; exit 1; }

mkdir -p "$D/results"
for CONC in 512 4096; do
  JOB="dsr1-m4n-bench-c$CONC"
  echo "=== bench conc $CONC"
  kubectl delete job "$JOB" -n dynamo-cloud --ignore-not-found >/dev/null 2>&1
  for i in $(seq 1 30); do kubectl get job "$JOB" -n dynamo-cloud >/dev/null 2>&1 || break; sleep 5; done
  kubectl apply -f "$D/manifests/m4n-bench-conc$CONC.yaml" >/dev/null
  for i in $(seq 1 240); do
    st=$(kubectl get job "$JOB" -n dynamo-cloud -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null)
    [ -n "$st" ] && break
    sleep 60
  done
  echo "=== conc $CONC terminal: ${st:-TIMEOUT}"
  kubectl logs -n dynamo-cloud "job/$JOB" > "$S/bench-m4n-c$CONC.log" 2>/dev/null
  python3 - "$S/bench-m4n-c$CONC.log" "$D/results" <<'EOF'
import json, re, sys
logf, out = sys.argv[1], sys.argv[2]
log = open(logf, errors='replace').read()
found = 0
for m in re.finditer(r'--- /tmp/results/(results_concurrency_\d+_gpus_\d+_ctx_\d+_gen_\d+_nats)\.json\n(\{.*?\})\n', log, re.S):
    d = json.loads(m.group(2)); found += 1
    json.dump(d, open(f'{out}/m4n-{m.group(1)}.json', 'w'))
if not found:
    for m in re.finditer(r'=== SLIM-RESULT /tmp/results/(results_concurrency_\d+_gpus_\d+_ctx_\d+_gen_\d+_nats)\.json (\{.*\})', log):
        d = json.loads(m.group(2)); found += 1
        json.dump(d, open(f'{out}/m4n-{m.group(1)}.slim.json', 'w'))
        print('(recovered via SLIM tail)')
for m in re.finditer(r'--- /tmp/results/|=== SLIM-RESULT', log[:0]): pass
if found:
    import glob, os
    f = sorted(glob.glob(f'{out}/m4n-results_concurrency_*'), key=os.path.getmtime)[-1]
    d = json.load(open(f))
    print(f"m4n conc {d['max_concurrency']}: out/decode-gpu {d['output_throughput']/48:.1f} | TPOT {d['mean_tpot_ms']:.2f}ms | TTFT med {d['median_ttft_ms']/1000:.1f}s")
print(f'extracted {found}')
EOF
done

mkdir -p "$S/srv"; for p in $(kubectl get pods -n dynamo-cloud --no-headers | grep "$PFX" | awk '{print $1}'); do kubectl logs -n dynamo-cloud "$p" > "$S/srv/$p.log" 2>/dev/null; done
export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token 2>/dev/null)
gcloud storage rsync -r "$S/srv" "$DEST/server-logs/dsr1-m4n-2pts" 2>&1 | tail -1
for fl in "$S"/bench-m4n-c*.log; do gcloud storage cp "$fl" "$DEST/client-logs/$(basename "$fl")" 2>&1 | tail -1; done
gcloud storage rsync -r "$D/results" "$DEST/results" 2>&1 | tail -1
echo "=== tearing down m4n"
kubectl delete -n dynamo-cloud "deployment/$PFX-frontend" "deployment/$PFX-prefill" >/dev/null 2>&1
kubectl delete -n dynamo-cloud "statefulset/$PFX-decode" >/dev/null 2>&1
kubectl delete -n dynamo-cloud "computedomain/$PFX-cd" "service/$PFX-frontend" "service/$PFX-decode" "job/dsr1-m4n-bench-c512" "job/dsr1-m4n-bench-c4096" >/dev/null 2>&1
rm -rf "$S"
echo "=== M4N 2PTS DONE"
