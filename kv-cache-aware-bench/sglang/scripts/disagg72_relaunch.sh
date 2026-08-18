#!/bin/bash
# Scale-up chain: wait for both 72-GPU stacks, transfer-test the kv arm,
# apply GDR bypass if transfers fail, then launch both benches gated.
set -u
NS=dynamo-cloud
PF="$HOME/kv-cache-aware-bench/manifests/perf"
LOG=/tmp/d72_relaunch.log
say() { echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }
probe() { # completion probe against an arm; returns 0 on generated content
  local arm=$1
  local FE=$(kubectl get pods -n $NS -l app=$arm-frontend -o name | head -1)
  kubectl exec -n $NS "$FE" -- curl -s -m 300 -X POST http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"alisachen/Kimi-K2.5-NVFP4","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"stream":false}' \
    2>/dev/null | grep -q '"completion_tokens":[1-9]'
}
nixl_errors() {
  local arm=$1 n=0
  for p in $(kubectl get pods -n $NS -l app=$arm-prefill -o name 2>/dev/null); do
    c=$(kubectl logs -n $NS ${p#pod/} -c prefill --since=30m 2>/dev/null | grep -cE "NIXL_ERR|REMOTE_DISCONNECT|Foreign traffic")
    n=$((n+c))
  done
  echo $n
}
for arm in sgl-disagg72-kv sgl-disagg72-rr; do
  say "waiting for $arm (19 pods)"
  for i in $(seq 1 90); do
    ready=$(kubectl get pods -n $NS -l "nvidia.com/dynamo-graph-deployment-name=$arm" --no-headers 2>/dev/null | awk '$2=="2/2"' | wc -l)
    say "[$arm $i] $ready/19"; [ "$ready" -ge 19 ] && break; sleep 120
  done
done
say "transfer test on kv arm"
if probe sgl-disagg72-kv && [ "$(nixl_errors sgl-disagg72-kv)" = "0" ]; then
  say "TRANSFER OK — no GDR bypass needed"
else
  say "transfer FAILED (errors=$(nixl_errors sgl-disagg72-kv)) — applying GDR bypass to all disagg72 workers"
  for d in sgl-disagg72-kv-prefill sgl-disagg72-kv-decode sgl-disagg72-rr-prefill sgl-disagg72-rr-decode; do
    kubectl set env deployment/$d -n $NS UCX_IB_GPU_DIRECT_RDMA=n >> "$LOG" 2>&1
  done
  say "waiting for rollouts after bypass"
  for i in $(seq 1 40); do
    ready=$(kubectl get pods -n $NS --no-headers 2>/dev/null | grep disagg72 | awk '$2=="2/2"' | wc -l)
    say "[bypass-rollout $i] $ready/38"; [ "$ready" -ge 38 ] && break; sleep 120
  done
  if probe sgl-disagg72-kv; then say "TRANSFER OK WITH GDR BYPASS (document caveat)"; else say "STILL FAILING AFTER BYPASS — STOP (kv-rdma policy)"; exit 1; fi
fi
for arm in sgl-disagg72-kv sgl-disagg72-rr; do
  probe $arm && { say "$arm gate passed — launching bench"; kubectl apply -n $NS -f "$PF/${arm}-bench.yaml" >> "$LOG" 2>&1; } || say "$arm completion failed — bench NOT launched"
done
say "DISAGG72 RELAUNCH DONE"
