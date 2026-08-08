#!/usr/bin/env bash
# M4R MESH TEST: deploy m4r rev3 (unchanged env) on np-3 -> registrations -> exec-based
# serving gate -> KV-transport gate -> MESH-WARM BURST under guard watch -> verdict ->
# evidence -> teardown. No full bench. Answers: does the multi-rank wireup storm still
# exist on the rebuilt pool? Healthy end-to-end ~40 min; guard bounds any failure to minutes.
set -uo pipefail
export KUBECONFIG=$HOME/.kube/REDACTED-GKE-CLUSTER.config
D=$HOME/dsr1-pareto/dsr1-sweep
GUARD=$HOME/dsr1-pareto/common/kv-transport-guard.sh
DEST="gs://gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs"
S=$(mktemp -d)
PFX="dsr1-m4r"
EXTRA_ENV="${1:-}"   # optional: "NAME=VAL NAME2=VAL2" applied to both workers (candidate arms)

echo "=== deploying m4r rev3 on np-3 (mesh test)${EXTRA_ENV:+ + candidate env: $EXTRA_ENV}"
kubectl apply -f "$D/manifests/mrdma-claim-template.yaml" >/dev/null
kubectl apply -f "$D/manifests/m4r-midcurve-rdmakv.yaml" >/dev/null
if [ -n "$EXTRA_ENV" ]; then
  kubectl set env deployment/$PFX-prefill -n dynamo-cloud $EXTRA_ENV >/dev/null
  kubectl set env statefulset/$PFX-decode -n dynamo-cloud $EXTRA_ENV >/dev/null
fi
n=0
for i in $(seq 1 90); do
  n=$(kubectl exec -n dynamo-cloud dynamo-platform-etcd-0 -- etcdctl --endpoints=localhost:2379 get v1/instances/ --prefix --keys-only 2>/dev/null | grep -c generate)
  [ "${n:-0}" -ge 7 ] && break
  bad=$(kubectl get pods -n dynamo-cloud --no-headers 2>/dev/null | grep -E "$PFX-(prefill|decode)" | grep -cE "CrashLoop|Error" || true)
  if [ "$bad" != "0" ]; then
    echo "!!! WORKER FAILURE during bring-up"
    kubectl get pods -n dynamo-cloud | grep "$PFX" | grep -E "CrashLoop|Error"
    exit 1
  fi
  sleep 60
done
[ "${n:-0}" -ge 7 ] || { echo "!!! registration timeout ($n/7)"; exit 1; }
echo "=== $n registered; restarting frontends"
kubectl rollout restart "deployment/$PFX-frontend" -n dynamo-cloud >/dev/null
kubectl rollout status "deployment/$PFX-frontend" -n dynamo-cloud --timeout=420s >/dev/null

# exec-based serving gate (immune to the pod-network probe blackhole seen on rebuilt pools)
ok=0
for i in $(seq 1 40); do
  r=$(kubectl exec -n dynamo-cloud "deploy/$PFX-frontend" -- python3 -c "
import json, urllib.request
req = urllib.request.Request('http://localhost:8000/v1/completions',
    json.dumps({'model': 'deepseek-ai/DeepSeek-R1', 'prompt': 'hi', 'max_tokens': 4,
                'stream': False}).encode(), {'Content-Type': 'application/json'})
print(urllib.request.urlopen(req, timeout=90).read().decode()[:120])
" 2>/dev/null)
  if echo "$r" | grep -q '"text"'; then ok=$((ok+1)); else ok=0; fi
  [ "$ok" -ge 3 ] && { echo "=== serving verified"; break; }
  sleep 15
done
[ "$ok" -ge 3 ] || { echo "!!! never served — aborting"; exit 1; }

echo "=== KV-transport pre-burst gate"
"$GUARD" gate "$PFX" || { echo "!!! MESH TEST ABORT: transport dirty before burst"; exit 1; }

echo "=== MESH-WARM BURST (conc 512 x 1024 8k-prompts, rate inf) with guard watch"
kubectl delete job dsr1-m4r-meshwarm -n dynamo-cloud --ignore-not-found >/dev/null 2>&1
for i in $(seq 1 30); do kubectl get job dsr1-m4r-meshwarm -n dynamo-cloud >/dev/null 2>&1 || break; sleep 5; done
kubectl apply -f "$D/manifests/m4r-meshwarm.yaml" >/dev/null
export GUARD_MARKER="$S/kv-guard-tripped"
"$GUARD" watch "$PFX" dsr1-m4r-meshwarm 60 & GPID=$!
st=""
for i in $(seq 1 35); do
  st=$(kubectl get job dsr1-m4r-meshwarm -n dynamo-cloud -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null)
  [ -n "$st" ] && break
  [ -f "$GUARD_MARKER" ] && break
  sleep 60
done
kill "$GPID" >/dev/null 2>&1
kubectl logs -n dynamo-cloud job/dsr1-m4r-meshwarm > "$S/meshwarm.log" 2>/dev/null

echo "=== post-burst evidence"
mkdir -p "$S/srv"
for p in $(kubectl get pods -n dynamo-cloud --no-headers | grep "$PFX-prefill" | awk '{print $1}'); do
  kubectl logs -n dynamo-cloud "$p" > "$S/srv/$p.log" 2>/dev/null
done
kubectl logs -n dynamo-cloud dsr1-m4r-decode-0 > "$S/srv/decode-0.log" 2>/dev/null
we=$(cat "$S"/srv/*.log | grep -c "no remote ep address")
nerr=$(cat "$S"/srv/*.log | grep -cE "NIXL_ERR|REMOTE_DISCONNECT")
rc=$(cat "$S"/srv/*.log | grep -c "rc_mlx5")
done_req=$(grep -oE "Successful requests:\s+[0-9]+" "$S/meshwarm.log" | tail -1 || true)
dur=$(grep -oE "Benchmark duration \(s\):\s+[0-9.]+" "$S/meshwarm.log" | tail -1 || true)
echo "MESH EVIDENCE: wireup_err=$we nixl_err=$nerr rc_mlx5=$rc | $done_req | $dur | job=${st:-TIMEOUT/KILLED}"
if [ -f "$GUARD_MARKER" ]; then
  echo "MESH VERDICT: STORM REPRODUCED (guard tripped; see violations above)"
elif [ "$we" = "0" ] && [ "$nerr" = "0" ] && [[ "${st:-}" == *Complete* ]]; then
  echo "MESH VERDICT: MESH-CLEAN on np-3 — full RDMA sweep unblocked"
else
  echo "MESH VERDICT: DEGRADED (job=$st wireup=$we nixl=$nerr) — inspect logs"
fi

export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token 2>/dev/null)
gcloud storage rsync -r "$S/srv" "$DEST/server-logs/dsr1-m4r-meshtest" 2>&1 | tail -1
gcloud storage cp "$S/meshwarm.log" "$DEST/client-logs/dsr1-m4r-meshwarm.log" 2>&1 | tail -1
echo "=== tearing down m4r"
kubectl delete -n dynamo-cloud "deployment/$PFX-frontend" "deployment/$PFX-prefill" >/dev/null 2>&1
kubectl delete -n dynamo-cloud "statefulset/$PFX-decode" "deployment/$PFX-decode" >/dev/null 2>&1
kubectl delete -n dynamo-cloud "computedomain/$PFX-cd" "service/$PFX-frontend" "service/$PFX-decode" "job/dsr1-m4r-meshwarm" >/dev/null 2>&1
rm -rf "$S"
echo "=== M4R MESH TEST DONE"
