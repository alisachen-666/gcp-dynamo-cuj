#!/bin/bash
# Fresh-cluster chain: wait for each arm's stack, gate (completion + kv-events
# where applicable), launch its bench independently as it becomes ready.
set -u
NS=dynamo-cloud
PF="$HOME/kv-cache-aware-bench/manifests/perf"
LOG=/tmp/rebootstrap.log
say() { echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }
launch_when_ready() {
  local arm=$1 want=$2 kv=$3 ct=$4
  for i in $(seq 1 150); do
    local ready=$(kubectl get pods -n $NS -l "nvidia.com/dynamo-graph-deployment-name=$arm" --no-headers 2>/dev/null | awk '$2=="2/2"' | wc -l)
    say "[$arm $i] $ready/$want"
    [ "$ready" -ge "$want" ] && break
    sleep 120
  done
  [ "$ready" -ge "$want" ] || { say "$arm TIMED OUT"; return 1; }
  local FE=$(kubectl get pods -n $NS -l app=$arm-frontend -o name | head -1)
  kubectl exec -n $NS "$FE" -- curl -s -m 180 -X POST http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"alisachen/Kimi-K2.5-NVFP4","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"stream":false}' \
    2>/dev/null | grep -q '"content"' || { say "$arm completion FAILED"; return 1; }
  if [ "$kv" = "1" ]; then
    local ev=$(kubectl logs -n $NS -l "nvidia.com/dynamo-graph-deployment-name=$arm" -c $ct --tail -1 2>/dev/null | grep -c "use_kv_events=True")
    say "$arm use_kv_events lines: $ev"
    [ "$ev" -ge 1 ] || { say "$arm KV EVENTS OFF — not launching"; return 1; }
  fi
  say "$arm gates passed — launching bench"
  kubectl apply -n $NS -f "$PF/${arm}-bench.yaml" >> "$LOG" 2>&1
}
launch_when_ready sgl-agg-rr 7 0 worker &
launch_when_ready sgl-agg-kv 7 1 agg &
launch_when_ready sgl-disagg72-kv 19 1 prefill &
launch_when_ready sgl-disagg72-rr 19 0 prefill &
wait
say "REBOOTSTRAP CHAIN DONE"
