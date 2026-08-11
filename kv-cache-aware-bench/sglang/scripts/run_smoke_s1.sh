#!/bin/bash
# SGLang smoke S1: 1P+1D bring-up, model registration, one manual completion.
# Usage: bash run_smoke_s1.sh [ready-budget-minutes]  (default 150)
set -u
NS=dynamo-cloud
M="$HOME/kv-cache-aware-bench/sglang/manifests"
LOG=${LOG:-/tmp/sgl_smoke_s1.log}
BUDGET=${1:-150}
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "S1: applying sgl-smoke (1P+1D+frontend)"
kubectl apply -n $NS -f "$M/sgl-smoke.yaml" >> "$LOG" 2>&1 || { say "apply FAILED"; exit 1; }
say "S1: waiting for worker rollouts (old pods fully replaced)"
for d in sgl-smoke-prefill sgl-smoke-decode; do
  kubectl rollout status deployment/$d -n $NS --timeout=60m >> "$LOG" 2>&1 || { say "rollout $d timed out"; exit 1; }
done

for i in $(seq 1 "$BUDGET"); do
  ready=$(kubectl get pods -n $NS -l 'nvidia.com/dynamo-graph-deployment-name=sgl-smoke' \
    --no-headers 2>/dev/null | awk '$2=="2/2"||$2=="1/1"' | wc -l)
  say "[ready-wait $i] $ready/3"
  [ "$ready" -ge 3 ] && break
  # surface first crash fast: pip/glue failures die in the first minutes
  kubectl get pods -n $NS -l 'nvidia.com/dynamo-graph-deployment-name=sgl-smoke' --no-headers 2>/dev/null \
    | grep -E "CrashLoop|Error" | head -2 | while read -r l; do say "POD ERROR: $l"; done
  sleep 60
done
[ "$ready" -ge 3 ] || { say "S1 FAIL: not ready in ${BUDGET}m"; exit 1; }

FE=$(kubectl get pods -n $NS -l app=sgl-smoke-frontend -o name | head -1)
say "S1: checking /v1/models via $FE"
kubectl exec -n $NS "$FE" -- curl -s http://localhost:8000/v1/models | tee -a "$LOG" | grep -q Kimi \
  || { say "S1 FAIL: model not registered"; exit 1; }

say "S1: manual completion"
kubectl exec -n $NS "$FE" -- curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"alisachen/Kimi-K2.5-NVFP4","messages":[{"role":"user","content":"Say OK and nothing else."}],"max_tokens":8,"stream":false}' \
  | tee -a "$LOG" | grep -q '"content"' || { say "S1 FAIL: completion failed"; exit 1; }

say "S1: engine versions (glue-without-downgrade check)"
for d in prefill decode; do
  P=$(kubectl get pods -n $NS -l app=sgl-smoke-$d -o name | head -1)
  v=$(kubectl exec -n $NS "$P" -- python3 -c "import sglang;print(sglang.__version__)" 2>/dev/null)
  say "  $d sglang=$v (want 0.5.17)"
done
say "S1 PASS"
