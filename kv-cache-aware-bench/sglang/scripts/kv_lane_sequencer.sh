#!/bin/bash
# Full KV lane: wait RR ladder -> swap arms on np-3 -> gate -> KV main ladder
# (defaults) under guard -> live router-flag sweep.
set -u
NS=dynamo-cloud
LOG=/tmp/kv_lane.log
say() { echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }
say "waiting for RR ladder to finish"
while true; do
  st=$(kubectl get jobs -n $NS alisachen-sgl-disagg72-rr-bench --no-headers 2>/dev/null | awk '{print $2}')
  [ "$st" = "Complete" ] || [ "$st" = "Failed" ] || [ -z "$st" ] && break
  sleep 300
done
say "RR bench state: ${st:-deleted-by-guard}; swapping arms on np-3"
kubectl scale deployment -n $NS sgl-disagg72-rr-prefill sgl-disagg72-rr-decode sgl-disagg72-rr-frontend --replicas=0 >> "$LOG" 2>&1
sleep 120
kubectl scale deployment -n $NS sgl-disagg72-kv-frontend --replicas=1 >> "$LOG" 2>&1
kubectl scale deployment -n $NS sgl-disagg72-kv-prefill --replicas=6 >> "$LOG" 2>&1
kubectl scale deployment -n $NS sgl-disagg72-kv-decode --replicas=12 >> "$LOG" 2>&1
for i in $(seq 1 60); do
  ready=$(kubectl get pods -n $NS -l "nvidia.com/dynamo-graph-deployment-name=sgl-disagg72-kv" --no-headers 2>/dev/null | awk '$2=="2/2"' | wc -l)
  say "[kv-up $i] $ready/19"; [ "$ready" -ge 19 ] && break; sleep 120
done
[ "$ready" -ge 19 ] || { say "KV STACK TIMEOUT"; exit 1; }
FE=$(kubectl get pods -n $NS -l app=sgl-disagg72-kv-frontend -o name | head -1)
kubectl exec -n $NS "$FE" -- curl -s -m 330 -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"alisachen/Kimi-K2.5-NVFP4","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"stream":false}' \
  2>/dev/null | grep -q '"completion_tokens":[1-9]' || { say "KV COMPLETION GATE FAILED"; exit 1; }
ev=$(kubectl logs -n $NS -l app=sgl-disagg72-kv-prefill -c prefill --tail -1 2>/dev/null | grep -c "use_kv_events=True")
say "completion OK; use_kv_events lines=$ev"
sleep 30
bash "$HOME/DynamoBench/common/kv-transport-guard.sh" gate "sgl-disagg72-kv" || { say "KV TRANSPORT GATE FAILED - RDMA POLICY STOP"; exit 1; }
say "gates passed - launching KV MAIN ladder under guard"
kubectl apply -n $NS -f "$HOME/kv-cache-aware-bench/manifests/perf/sgl-disagg72-kv-bench.yaml" >> "$LOG" 2>&1
GUARD_MARKER=/tmp/kv-guard-tripped-kvarm bash "$HOME/DynamoBench/common/kv-transport-guard.sh" watch "sgl-disagg72-kv" alisachen-sgl-disagg72-kv-bench 300 &
GW=$!
while true; do
  st=$(kubectl get jobs -n $NS alisachen-sgl-disagg72-kv-bench --no-headers 2>/dev/null | awk '{print $2}')
  [ "$st" = "Complete" ] || [ "$st" = "Failed" ] || [ -z "$st" ] && break
  sleep 300
done
kill $GW 2>/dev/null
[ -f /tmp/kv-guard-tripped-kvarm ] && { say "GUARD TRIPPED DURING KV LADDER - STOPPED"; exit 1; }
say "KV main ladder: ${st:-guard-stopped}; starting router-flag sweep"
bash "$HOME/kv-cache-aware-bench/sglang/scripts/sweep_router_flags.sh"
say "KV LANE COMPLETE (ladder + flag sweep)"
