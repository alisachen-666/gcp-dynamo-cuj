#!/usr/bin/env bash
# LLR: RDMA-KV sweep arm of DSR1 low_latency (1P TP4 + 4D TP4, 5 nodes, NATS, mult10).
# One deployment, one bench-all job (conc 4/8/32/64), transport audit, teardown.
set -uo pipefail
export KUBECONFIG=$HOME/.kube/REDACTED-GKE-CLUSTER.config
D=$HOME/dsr1-pareto/dsr1-sweep
DEST="gs://REDACTED-GCS-BUCKET/mlperf_results/nvfp4_20260519_run5/logs/logs"
S=$(mktemp -d)
PFX="dsr1-llr"

echo "=== deploying llr (RDMA-KV low_latency)"
kubectl apply -f "$D/manifests/mrdma-claim-template.yaml" >/dev/null
kubectl apply -f "$D/manifests/llr-lowlatency-rdmakv.yaml" >/dev/null
n=0
for i in $(seq 1 90); do
  n=$(kubectl exec -n dynamo-cloud dynamo-platform-etcd-0 -- etcdctl --endpoints=localhost:2379 get v1/instances/ --prefix --keys-only 2>/dev/null | grep -c generate)
  [ "${n:-0}" -ge 5 ] && break
  bad=$(kubectl get pods -n dynamo-cloud --no-headers 2>/dev/null | grep -E "$PFX-(prefill|decode)" | grep -cE "CrashLoop|Error" || true)
  if [ "$bad" != "0" ]; then
    echo "!!! LLR WORKER FAILURE"; kubectl get pods -n dynamo-cloud | grep "$PFX"
    for p in $(kubectl get pods -n dynamo-cloud --no-headers | grep -E "$PFX-(prefill|decode)" | grep -E "CrashLoop|Error" | awk '{print $1}' | head -2); do
      kubectl logs -n dynamo-cloud "$p" --tail=30 --previous 2>/dev/null | grep -iE "error|nixl|ucx|disconnect" | head -6
    done
    exit 1
  fi
  sleep 60
done
[ "${n:-0}" -ge 5 ] || { echo "!!! llr registration timeout ($n/5)"; exit 1; }
echo "=== $n registered; restarting frontend"
kubectl rollout restart "deployment/$PFX-frontend" -n dynamo-cloud >/dev/null
kubectl rollout status "deployment/$PFX-frontend" -n dynamo-cloud --timeout=420s >/dev/null
ok=0
for i in $(seq 1 60); do
  r=$(kubectl run "probe-llr-$i" -n dynamo-cloud --image=curlimages/curl --restart=Never --rm -q -i --timeout=180s -- \
      -s -m 120 -X POST "http://$PFX-frontend.dynamo-cloud.svc.cluster.local:8000/v1/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"deepseek-ai/DeepSeek-R1","prompt":"hi","max_tokens":4,"stream":false}' 2>/dev/null)
  if echo "$r" | grep -q '"text"'; then ok=$((ok+1)); else ok=0; fi
  [ "$ok" -ge 3 ] && { echo "=== serving verified"; break; }
  sleep 20
done
[ "$ok" -ge 3 ] || { echo "!!! llr never served — aborting"; exit 1; }

echo "=== KV-transport gate (probes above exercised the KV path)"
GUARD=$HOME/dsr1-pareto/common/kv-transport-guard.sh
"$GUARD" gate "$PFX" || { echo "!!! LLR ABORTED: KV path not clean RDMA"; exit 1; }

echo "=== bench-all (conc 4/8/32/64, mult10) with transport watchdog"
kubectl delete job dsr1-llr-bench -n dynamo-cloud --ignore-not-found >/dev/null 2>&1
for i in $(seq 1 30); do kubectl get job dsr1-llr-bench -n dynamo-cloud >/dev/null 2>&1 || break; sleep 5; done
kubectl apply -f "$D/manifests/llr-bench-all.yaml" >/dev/null
export GUARD_MARKER="$S/kv-guard-tripped"
"$GUARD" watch "$PFX" dsr1-llr-bench 180 & GPID=$!
st=""
for i in $(seq 1 300); do
  st=$(kubectl get job dsr1-llr-bench -n dynamo-cloud -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null)
  [ -n "$st" ] && break
  [ -f "$GUARD_MARKER" ] && break
  sleep 60
done
kill "$GPID" >/dev/null 2>&1
if [ -f "$GUARD_MARKER" ]; then
  echo "!!! LLR FAILED: KV transfer degraded to TCP mid-bench (guard killed the job)"
  kubectl delete -n dynamo-cloud "deployment/$PFX-frontend" "deployment/$PFX-prefill" "deployment/$PFX-decode" "computedomain/$PFX-cd" "service/$PFX-frontend" >/dev/null 2>&1
  exit 1
fi
echo "=== bench terminal: ${st:-TIMEOUT}"
mkdir -p "$D/results"
kubectl logs -n dynamo-cloud job/dsr1-llr-bench > "$S/bench-llr.log" 2>/dev/null
python3 - "$S/bench-llr.log" "$D/results" <<'EOF'
import json, re, sys
logf, out = sys.argv[1], sys.argv[2]
log = open(logf, errors='replace').read()
found = 0
for m in re.finditer(r'--- /tmp/results/(results_concurrency_\d+_gpus_\d+_ctx_\d+_gen_\d+_mult10_rdmakv)\.json\n(\{.*?\})\n', log, re.S):
    d = json.loads(m.group(2)); found += 1
    json.dump(d, open(f'{out}/llr-{m.group(1)}.json', 'w'))
    print(f"llr conc {d['max_concurrency']}: out/decode-gpu {d['output_throughput']/16:.1f} | TPOT {d['mean_tpot_ms']:.2f}ms")
for m in re.finditer(r'=== SLIM-RESULT /tmp/results/(results_concurrency_\d+_gpus_\d+_ctx_\d+_gen_\d+_mult10_rdmakv)\.json (\{.*\})', log):
    name = m.group(1)
    import os
    if not os.path.exists(f'{out}/llr-{name}.json'):
        d = json.loads(m.group(2)); found += 1
        json.dump(d, open(f'{out}/llr-{name}.slim.json', 'w'))
        print(f'(recovered {name} via SLIM tail)')
print(f'llr extracted {found}')
EOF

echo "=== transport-evidence audit"
mkdir -p "$S/srv"
for p in $(kubectl get pods -n dynamo-cloud --no-headers | grep "$PFX" | awk '{print $1}'); do
  kubectl logs -n dynamo-cloud "$p" > "$S/srv/$p.log" 2>/dev/null
done
tot_rc=0; tot_ipc=0; tot_err=0
for f in "$S"/srv/*prefill*.log "$S"/srv/*decode*.log; do
  [ -f "$f" ] || continue
  tot_rc=$((tot_rc+$(grep -c 'rc_mlx5' "$f"))); tot_ipc=$((tot_ipc+$(grep -c 'cuda_ipc' "$f"))); tot_err=$((tot_err+$(grep -cE 'NIXL_ERR|REMOTE_DISCONNECT' "$f")))
done
echo "TRANSPORT TOTALS: rc_mlx5=$tot_rc cuda_ipc=$tot_ipc nixl_errors=$tot_err"
[ "$tot_ipc" = "0" ] && [ "$tot_err" = "0" ] && [ "$tot_rc" != "0" ] && echo "TRANSPORT VERDICT: RDMA-KV CLEAN" || echo "TRANSPORT VERDICT: CHECK LOGS"

export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token 2>/dev/null)
gcloud storage rsync -r "$S/srv" "$DEST/server-logs/dsr1-llr" 2>&1 | tail -1
gcloud storage cp "$S/bench-llr.log" "$DEST/client-logs/dsr1-llr-bench.log" 2>&1 | tail -1
gcloud storage rsync -r "$D/results" "$DEST/results" 2>&1 | tail -1
echo "=== tearing down llr"
kubectl delete -n dynamo-cloud "deployment/$PFX-frontend" "deployment/$PFX-prefill" "deployment/$PFX-decode" >/dev/null 2>&1
kubectl delete -n dynamo-cloud "computedomain/$PFX-cd" "service/$PFX-frontend" "job/dsr1-llr-bench" >/dev/null 2>&1
rm -rf "$S"
echo "=== LLR DONE"
