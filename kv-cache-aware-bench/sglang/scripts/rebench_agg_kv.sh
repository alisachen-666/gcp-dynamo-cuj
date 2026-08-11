#!/bin/bash
set -u
NS=dynamo-cloud
LOG=/tmp/sgl_aggkv_rebench.log
say() { echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }
say "waiting for sgl-agg-kv rollout with kv-events"
for i in $(seq 1 90); do
  ready=$(kubectl get pods -n $NS -l "nvidia.com/dynamo-graph-deployment-name=sgl-agg-kv" --no-headers 2>/dev/null | awk '$2=="2/2"' | wc -l)
  say "[ready-wait $i] $ready/7"
  [ "$ready" -ge 7 ] && break
  sleep 60
done
[ "$ready" -ge 7 ] || { say "not ready in 90m"; exit 1; }
ev=$(kubectl logs -n $NS -l app=sgl-agg-kv-worker -c agg --tail 400 2>/dev/null | grep -c "use_kv_events=True")
say "workers reporting use_kv_events=True: $ev"
[ "$ev" -ge 1 ] || { say "KV EVENTS STILL OFF - aborting"; exit 1; }
say "relaunching agg-kv bench (full ladder)"
kubectl apply -n $NS -f "$HOME/kv-cache-aware-bench/manifests/perf/sgl-agg-kv-bench.yaml" >> "$LOG" 2>&1
say "launched"
