#!/usr/bin/env bash
# RXR rail-crossing smoke: reproduce the m4r rev3 UCX failure in minutes (no e2e bench).
# Arm A repro (prefill GPU0 vs decode GPU2, disjoint rails) -> expect wireup errors/tcp fallback
# Arm B control (decode GPU0, same rail)                    -> expect clean rc_mlx5
# Arm C fix candidate (GPU0 vs GPU2 + UCX_IB_PREFER_NEAREST_DEVICE=n)
set -uo pipefail
export KUBECONFIG=$HOME/.kube/REDACTED-GKE-CLUSTER.config
D=$HOME/dsr1-pareto/dsr1-sweep
S=$(mktemp -d)
PFX="dsr1-rxr"
LONGP=$(python3 -c "print(' '.join(['rail crossing kv transfer probe token']*80))")

wait_serving() {  # $1 = arm label; returns 0 if served, 1 if not
  # exec-based probe: runs INSIDE the frontend pod against localhost — immune to the
  # pod-network hang observed on the rebuilt pools (probe pods' SYNs to the frontend
  # never arrived while port-forward served fine; harness fault, 2026-08-06).
  local ok=0 r
  for i in $(seq 1 25); do
    r=$(kubectl exec -n dynamo-cloud "deploy/$PFX-frontend" -- python3 -c "
import json, urllib.request
req = urllib.request.Request('http://localhost:8000/v1/completions',
    json.dumps({'model': 'Qwen/Qwen3-4B', 'prompt': '$LONGP', 'max_tokens': 16,
                'stream': False}).encode(), {'Content-Type': 'application/json'})
print(urllib.request.urlopen(req, timeout=90).read().decode()[:200])
" 2>/dev/null)
    if echo "$r" | grep -q '"text"'; then ok=$((ok+1)); else ok=0; fi
    [ "$ok" -ge 2 ] && return 0
    sleep 15
  done
  return 1
}

audit() {  # $1 = arm label
  mkdir -p "$S/$1"
  local pp dp
  pp=$(kubectl get pods -n dynamo-cloud --no-headers | grep "$PFX-prefill" | grep Running | awk '{print $1}' | head -1)
  dp=$(kubectl get pods -n dynamo-cloud --no-headers | grep "$PFX-decode"  | grep Running | awk '{print $1}' | head -1)
  kubectl logs -n dynamo-cloud "$pp" > "$S/$1/prefill.log" 2>/dev/null
  kubectl logs -n dynamo-cloud "$dp" > "$S/$1/decode.log" 2>/dev/null
  kubectl logs -n dynamo-cloud "deploy/$PFX-frontend" > "$S/$1/frontend.log" 2>/dev/null
  local we rc tcp
  we=$(cat "$S/$1"/*.log | grep -c "no remote ep address")
  rc=$(cat "$S/$1"/*.log | grep -c "rc_mlx5")
  tcp=$(cat "$S/$1"/*.log | grep -cE "tcp/gpu[0-9]+ipvlan")
  echo "ARM $1 EVIDENCE: wireup_err=$we rc_mlx5=$rc tcp_ipvlan_bulk=$tcp served=$2"
}

echo "=== ARM A (repro): deploying rxr (prefill GPU0, decode GPU2)"
kubectl apply -f "$D/manifests/mrdma-claim-template.yaml" >/dev/null
kubectl apply -f "$D/manifests/rxr-railcross-smoke.yaml" >/dev/null
for i in $(seq 1 40); do
  n=$(kubectl exec -n dynamo-cloud dynamo-platform-etcd-0 -- etcdctl --endpoints=localhost:2379 get v1/instances/ --prefix --keys-only 2>/dev/null | grep -c generate)
  [ "${n:-0}" -ge 2 ] && break
  sleep 30
done
echo "=== $n registered; restarting frontend"
kubectl rollout restart deployment/$PFX-frontend -n dynamo-cloud >/dev/null
kubectl rollout status deployment/$PFX-frontend -n dynamo-cloud --timeout=300s >/dev/null
if wait_serving A; then sa=YES; else sa=NO; fi
audit A "$sa"

echo "=== ARM B (control): decode -> GPU0 (same rail as prefill)"
kubectl set env deployment/$PFX-decode -n dynamo-cloud CUDA_VISIBLE_DEVICES=0 >/dev/null
kubectl rollout status deployment/$PFX-decode -n dynamo-cloud --timeout=600s >/dev/null
sleep 90   # re-registration
if wait_serving B; then sb=YES; else sb=NO; fi
audit B "$sb"

echo "=== ARM C (fix candidate): GPU0 vs GPU2 + UCX_IB_PREFER_NEAREST_DEVICE=n"
kubectl set env deployment/$PFX-decode  -n dynamo-cloud CUDA_VISIBLE_DEVICES=2 UCX_IB_PREFER_NEAREST_DEVICE=n >/dev/null
kubectl set env deployment/$PFX-prefill -n dynamo-cloud UCX_IB_PREFER_NEAREST_DEVICE=n >/dev/null
kubectl rollout status deployment/$PFX-decode  -n dynamo-cloud --timeout=600s >/dev/null
kubectl rollout status deployment/$PFX-prefill -n dynamo-cloud --timeout=600s >/dev/null
sleep 90
if wait_serving C; then sc=YES; else sc=NO; fi
audit C "$sc"

echo "=== VERDICT SUMMARY"
echo "A(repro GPU0-GPU2):    served=$sa  (REPRODUCED if served=NO or tcp fallback/wireup errors present)"
echo "B(control GPU0-GPU0):  served=$sb  (expect YES, clean rc_mlx5)"
echo "C(fix candidate):      served=$sc  (fix validated if YES with rc_mlx5 and no wireup errors)"

export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token 2>/dev/null)
gcloud storage rsync -r "$S" "gs://REDACTED-GCS-BUCKET/mlperf_results/nvfp4_20260519_run5/logs/logs/server-logs/dsr1-rxr-smoke" 2>&1 | tail -1
cp -r "$S" "$D/results-summary/rxr-smoke-logs" 2>/dev/null
echo "=== tearing down rxr"
kubectl delete -n dynamo-cloud deployment/$PFX-frontend deployment/$PFX-prefill deployment/$PFX-decode service/$PFX-frontend >/dev/null 2>&1
rm -rf "$S"
echo "=== RXR SMOKE DONE"
