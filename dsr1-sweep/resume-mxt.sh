#!/usr/bin/env bash
# Resume mxt after the stale-registration abort: lln leftovers now deleted; wait for the 4
# freed nodes to schedule the pending decode pods, then gate + bench + harvest + teardown.
set -uo pipefail
export PATH=$HOME/google-cloud-sdk/bin:$PATH
D=$HOME/dsr1-pareto/dsr1-sweep
DEST="gs://gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs"
S=$(mktemp -d); PFX="dsr1-mxt"

echo "=== waiting for decode 8/8 Running + 11 fresh registrations"
for i in $(seq 1 90); do
  run=$(kubectl get pods -n dynamo-cloud --no-headers 2>/dev/null | grep "$PFX-decode" | grep -c Running || true)
  n=$(kubectl exec -n dynamo-cloud dynamo-platform-etcd-0 -- sh -c 'ETCDCTL_API=3 etcdctl --endpoints=localhost:2379 get v1/instances/ --prefix --keys-only' 2>/dev/null | grep -c generate)
  echo "  decode running: $run/8, registrations: $n"
  [ "$run" = "8" ] && [ "${n:-0}" -ge 11 ] && break
  sleep 60
done
kubectl rollout restart "deployment/$PFX-frontend" -n dynamo-cloud >/dev/null
kubectl rollout status "deployment/$PFX-frontend" -n dynamo-cloud --timeout=420s >/dev/null
ok=0
for i in $(seq 1 90); do
  r=$(kubectl run "probe-mxtr-$i" -n dynamo-cloud --image=curlimages/curl --restart=Never --rm -q -i --timeout=180s -- \
      -s -m 120 -X POST "http://$PFX-frontend.dynamo-cloud.svc.cluster.local:8000/v1/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"deepseek-ai/DeepSeek-R1","prompt":"hi","max_tokens":4,"stream":false}' 2>/dev/null)
  if echo "$r" | grep -q '"text"'; then ok=$((ok+1)); else ok=0; fi
  [ "$ok" -ge 3 ] && { echo "=== serving verified"; break; }
  sleep 20
done
[ "$ok" -ge 3 ] || { echo "!!! still not serving — aborting"; exit 1; }
echo "=== launching bench"
kubectl delete job dsr1-mxt-bench-c2048 -n dynamo-cloud --ignore-not-found >/dev/null 2>&1
sleep 10
kubectl apply -f "$D/manifests/mxt-bench-conc2048.yaml" >/dev/null
for i in $(seq 1 240); do
  st=$(kubectl get job dsr1-mxt-bench-c2048 -n dynamo-cloud -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null)
  [ -n "$st" ] && break
  sleep 60
done
echo "=== bench terminal: ${st:-TIMEOUT}"
mkdir -p "$D/results"
kubectl logs -n dynamo-cloud job/dsr1-mxt-bench-c2048 > "$S/bench-mxt.log" 2>/dev/null
python3 - "$S/bench-mxt.log" "$D/results" <<'PYEOF'
import json, re, sys
logf, out = sys.argv[1], sys.argv[2]
log = open(logf, errors='replace').read()
found = 0
for m in re.finditer(r'=== SLIM-RESULT /tmp/results/(results_concurrency_\d+_gpus_\d+_ctx_\d+_gen_\d+_tcp)\.json (\{.*\})', log):
    d = json.loads(m.group(2)); found += 1
    json.dump(d, open(f'{out}/mxt-{m.group(1)}.slim.json', 'w'))
    print(f"mxt conc {d['max_concurrency']}: out/decode-gpu {d['output_throughput']/32:.1f} | TPOT {d['mean_tpot_ms']:.2f}ms | TTFT med {d['median_ttft_ms']/1000:.1f}s | E2E {d['mean_e2el_ms']/1000:.1f}s")
print(f'extracted {found}')
PYEOF
mkdir -p "$S/srv"; for p in $(kubectl get pods -n dynamo-cloud --no-headers | grep "$PFX" | awk '{print $1}'); do kubectl logs -n dynamo-cloud "$p" > "$S/srv/$p.log" 2>/dev/null; done
grep -h -m2 -iE "request plane|NetworkManager" "$S/srv/$PFX-decode-0.log" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | head -2
export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token 2>/dev/null)
gcloud storage rsync -r "$S/srv" "$DEST/server-logs/dsr1-mxt" 2>&1 | tail -1
gcloud storage cp "$S/bench-mxt.log" "$DEST/client-logs/dsr1-mxt-bench.log" 2>&1 | tail -1
gcloud storage rsync -r "$D/results" "$DEST/results" 2>&1 | tail -1
echo "=== tearing down mxt"
kubectl delete -n dynamo-cloud "deployment/$PFX-frontend" "deployment/$PFX-prefill" "deployment/$PFX-decode" >/dev/null 2>&1
kubectl delete -n dynamo-cloud "statefulset/$PFX-decode" "computedomain/$PFX-cd" "service/$PFX-frontend" "service/$PFX-decode" "job/dsr1-mxt-bench-c2048" >/dev/null 2>&1
rm -rf "$S"
echo "=== MXT DONE"
