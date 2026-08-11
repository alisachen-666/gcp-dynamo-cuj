#!/bin/bash
# 72-GPU disagg lane: gate on S2 + clean RDMA transport proof, tear down np-1
# (trtllm 2D + smoke) and np-3 (aggtp8), deploy sgl-disagg72-kv (np-1) +
# sgl-disagg72-rr (np-3), per-arm completion + transport gate, launch benches.
set -u
NS=dynamo-cloud
M="$HOME/kv-cache-aware-bench/sglang/manifests"
P="$HOME/kv-cache-aware-bench/manifests/perf"
LOG=/tmp/sgl_d72_chain.log
say() { echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }

say "gate 1: waiting for S2 job to complete"
for i in $(seq 1 90); do
  st=$(kubectl get jobs -n $NS alisachen-sgl-smoke2-mini --no-headers 2>/dev/null | awk '{print $2}')
  [ "$st" = "Complete" ] && break
  [ "$st" = "Failed" ] && { say "S2 FAILED - stopping (kv-rdma stop policy)"; exit 1; }
  sleep 120
done
[ "$st" = "Complete" ] || { say "S2 timeout"; exit 1; }
say "S2 complete"

say "gate 2: transport proof on smoke workers"
ok=1
for app in sgl-smoke-prefill sgl-smoke-decode; do
  L=$(kubectl logs -n $NS -l app=$app -c ${app##*-} --tail 6000 2>/dev/null)
  rc=$(echo "$L" | grep -c 'rc_mlx5'); er=$(echo "$L" | grep -cE 'NIXL_ERR|REMOTE_DISCONNECT|Foreign traffic')
  say "  $app: rc_mlx5=$rc nixl_errors=$er"
  [ "$er" != "0" ] && ok=0
done
[ "$ok" = "1" ] || { say "TRANSPORT PROOF FAILED - stopping (kv-rdma stop policy)"; exit 1; }
say "transport proof clean"

say "teardown: np-1 trtllm 2D + smoke, np-3 aggtp8"
kubectl delete deployment -n $NS kimi-k25-disagg-kv-tuned-frontend kimi-k25-disagg-kv-tuned-prefill kimi-k25-disagg-kv-tuned-decode --wait=false >> "$LOG" 2>&1
kubectl delete deployment -n $NS sgl-smoke-frontend sgl-smoke-prefill sgl-smoke-decode --wait=false >> "$LOG" 2>&1
kubectl delete statefulset -n $NS kimi-k25-aggtp8-rr-w0 kimi-k25-aggtp8-rr-w1 kimi-k25-aggtp8-rr-w2 kimi-k25-aggtp8-kvt-w0 kimi-k25-aggtp8-kvt-w1 kimi-k25-aggtp8-kvt-w2 --wait=false >> "$LOG" 2>&1
kubectl delete deployment -n $NS kimi-k25-aggtp8-rr-frontend kimi-k25-aggtp8-kvt-frontend --wait=false >> "$LOG" 2>&1
sleep 120

say "deploy: sgl-disagg72-kv (np-1) + sgl-disagg72-rr (np-3)"
kubectl apply -n $NS -f "$M/sgl-disagg72-kv.yaml" -f "$M/sgl-disagg72-rr.yaml" >> "$LOG" 2>&1

for arm in sgl-disagg72-kv sgl-disagg72-rr; do
  say "waiting for $arm (19 pods)"
  for i in $(seq 1 180); do
    ready=$(kubectl get pods -n $NS -l "nvidia.com/dynamo-graph-deployment-name=$arm" \
      --no-headers 2>/dev/null | awk '$2=="2/2"||$2=="1/1"' | wc -l)
    say "[$arm ready-wait $i] $ready/19"
    [ "$ready" -ge 19 ] && break
    sleep 60
  done
  [ "$ready" -ge 19 ] || { say "$arm not ready in 180m; skipping bench"; continue; }
  FE=$(kubectl get pods -n $NS -l app=$arm-frontend -o name | head -1)
  say "$arm smoke: manual completion"
  kubectl exec -n $NS "$FE" -- curl -s -m 120 -X POST http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"alisachen/Kimi-K2.5-NVFP4","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"stream":false}' \
    2>/dev/null | grep -q '"content"' || { say "$arm completion FAILED; skipping bench"; continue; }
  er=$(kubectl logs -n $NS -l app=$arm-prefill -c prefill --tail 2000 2>/dev/null | grep -cE 'NIXL_ERR|REMOTE_DISCONNECT|Foreign traffic')
  [ "$er" != "0" ] && { say "$arm transport errors=$er; skipping bench (stop policy)"; continue; }
  say "$arm gates passed; launching bench"
  kubectl apply -n $NS -f "$P/${arm}-bench.yaml" >> "$LOG" 2>&1
done

say "watching disagg72 benches"
while true; do
  s=$(kubectl get jobs -n $NS --no-headers 2>/dev/null | grep -E "alisachen-sgl-disagg72" | awk '{print $1":"$2}' | tr '\n' ' ')
  say "jobs: $s"
  [ -n "$s" ] && ! echo "$s" | grep -q Running && break
  sleep 300
done
say "DISAGG72 CHAIN DONE: $s"
