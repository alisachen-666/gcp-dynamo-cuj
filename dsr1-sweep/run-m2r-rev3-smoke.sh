#!/usr/bin/env bash
# M2R rev3 smoke: rail-aware RDMA-KV (UCX_IB_ROCE_LOCAL_SUBNET=y, PREFIX_LEN=64) on the
# 2-node 1P-TP4 + 1D-TP4 handoff. PASS = completions flow AND bulk KV rides rc_mlx5
# (cuda_ipc 0, tcp bulk 0) — i.e. the subnet filter pairs same-rail devices without
# excluding the in-pod ipvlan GIDs. Gates the 18-node m4r rev3.
set -uo pipefail
export KUBECONFIG=$HOME/.kube/REDACTED-GKE-CLUSTER.config
D=$HOME/DynamoBench/dsr1-sweep
DEST="gs://REDACTED-GCS-BUCKET/mlperf_results/nvfp4_20260519_run5/logs/logs"
S=$(mktemp -d)
PFX="dsr1-m2r"

echo "=== deploying $PFX rev3 (rail-aware smoke)"
kubectl apply -f "$D/manifests/mrdma-claim-template.yaml" >/dev/null
kubectl apply -f "$D/manifests/m2r-rdma-smoke.yaml" >/dev/null

echo "=== waiting for 2 worker registrations"
n=0
for i in $(seq 1 90); do
  n=$(kubectl exec -n dynamo-cloud dynamo-platform-etcd-0 -- etcdctl --endpoints=localhost:2379 get v1/instances/ --prefix --keys-only 2>/dev/null | grep -c generate)
  [ "${n:-0}" -ge 2 ] && break
  bad=$(kubectl get pods -n dynamo-cloud --no-headers 2>/dev/null | grep -E "$PFX-(prefill|decode)" | grep -cE "CrashLoop|Error" || true)
  if [ "$bad" != "0" ]; then
    echo "!!! WORKER FAILURE"
    kubectl get pods -n dynamo-cloud | grep "$PFX"
    for p in $(kubectl get pods -n dynamo-cloud --no-headers | grep -E "$PFX-(prefill|decode)" | awk '{print $1}'); do
      kubectl logs -n dynamo-cloud "$p" --tail=30 2>/dev/null | grep -iE "error|nixl|ucx|traceback" | head -10
    done
    exit 1
  fi
  sleep 60
done
[ "${n:-0}" -ge 2 ] || { echo "!!! registration timeout"; exit 1; }
echo "=== $n registered; restarting frontend"
kubectl rollout restart "deployment/$PFX-frontend" -n dynamo-cloud >/dev/null
kubectl rollout status "deployment/$PFX-frontend" -n dynamo-cloud --timeout=420s >/dev/null

ok=0
for i in $(seq 1 40); do
  r=$(kubectl run "probe-m2r3-$i" -n dynamo-cloud --image=curlimages/curl --restart=Never --rm -q -i --timeout=180s -- \
      -s -m 120 -X POST "http://$PFX-frontend.dynamo-cloud.svc.cluster.local:8000/v1/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"deepseek-ai/DeepSeek-R1","prompt":"hi","max_tokens":4,"stream":false}' 2>/dev/null)
  if echo "$r" | grep -q '"text"'; then ok=$((ok+1)); else ok=0; fi
  [ "$ok" -ge 3 ] && { echo "=== serving verified"; break; }
  sleep 20
done
[ "$ok" -ge 3 ] || { echo "!!! never served — SMOKE FAIL (collect logs below)"; }

if [ "$ok" -ge 3 ]; then
  echo "=== driving 12 KV-handoff completions (longer prompts)"
  LONGP=$(python3 -c "print(' '.join(['benchmark rail aware kv transfer smoke']*120))")
  for i in $(seq 1 12); do
    kubectl run "probe-m2r3-kv-$i" -n dynamo-cloud --image=curlimages/curl --restart=Never --rm -q -i --timeout=240s -- \
      -s -m 180 -X POST "http://$PFX-frontend.dynamo-cloud.svc.cluster.local:8000/v1/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"deepseek-ai/DeepSeek-R1\",\"prompt\":\"$LONGP $i\",\"max_tokens\":32,\"stream\":false}" >/dev/null 2>&1
  done
  echo "=== completions sent"
fi

echo "=== collecting worker logs + transport evidence"
mkdir -p "$S/srv"
for p in $(kubectl get pods -n dynamo-cloud --no-headers | grep "$PFX" | awk '{print $1}'); do
  kubectl logs -n dynamo-cloud "$p" > "$S/srv/$p.log" 2>/dev/null
done
for f in "$S"/srv/*prefill*.log "$S"/srv/*decode*.log; do
  [ -f "$f" ] || continue
  echo "--- $(basename "$f")"
  echo "  rc_mlx5 proto lines : $(grep -c 'rc_mlx5' "$f")"
  echo "  cuda_ipc proto lines: $(grep -c 'cuda_ipc' "$f")"
  echo "  tcp bulk lines      : $(grep -cE 'tcp/(gpu|eth)[^ ]*.*(rndv|bulk|zcopy)' "$f")"
  echo "  NIXL errors         : $(grep -cE 'NIXL_ERR|REMOTE_DISCONNECT' "$f")"
  echo "  unreachable/ifindex : $(grep -ciE 'unreachable|no route|ifindex' "$f")"
done

export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token 2>/dev/null)
gcloud storage rsync -r "$S/srv" "$DEST/server-logs/dsr1-m2r-rev3-smoke" 2>&1 | tail -1

echo "=== tearing down $PFX"
kubectl delete -n dynamo-cloud "deployment/$PFX-frontend" "deployment/$PFX-prefill" "deployment/$PFX-decode" >/dev/null 2>&1
kubectl delete -n dynamo-cloud "computedomain/$PFX-cd" "service/$PFX-frontend" >/dev/null 2>&1
cp -r "$S/srv" "$D/results-summary/m2r-rev3-smoke-logs" 2>/dev/null
rm -rf "$S"
echo "=== M2R REV3 SMOKE DONE (serving ok: $ok)"
