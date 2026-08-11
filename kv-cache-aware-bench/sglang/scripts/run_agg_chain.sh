#!/bin/bash
# SGLang agg lane: wait for S1 PASS + both agg stacks ready, launch both benches.
set -u
NS=dynamo-cloud
P="$HOME/kv-cache-aware-bench/manifests/perf"
LOG=/tmp/sgl_agg_chain.log
say() { echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }

say "waiting for smoke S1 PASS"
for i in $(seq 1 180); do
  grep -q "S1 PASS" /tmp/sgl_smoke_s1.log 2>/dev/null && break
  grep -q "S1 FAIL" /tmp/sgl_smoke_s1.log 2>/dev/null && { say "S1 FAILED - aborting agg chain"; exit 1; }
  sleep 60
done
grep -q "S1 PASS" /tmp/sgl_smoke_s1.log || { say "S1 timeout"; exit 1; }
say "S1 PASS seen"

for arm in sgl-agg-rr sgl-agg-kv; do
  say "waiting for $arm stack (7 pods)"
  for i in $(seq 1 150); do
    ready=$(kubectl get pods -n $NS -l "nvidia.com/dynamo-graph-deployment-name=$arm" \
      --no-headers 2>/dev/null | awk '$2=="2/2"||$2=="1/1"' | wc -l)
    say "[$arm ready-wait $i] $ready/7"
    [ "$ready" -ge 7 ] && break
    sleep 60
  done
  [ "$ready" -ge 7 ] || { say "$arm not ready in 150m; skipping its bench"; continue; }
  say "launching $arm bench"
  kubectl apply -n $NS -f "$P/${arm}-bench.yaml" >> "$LOG" 2>&1
done

say "benches launched; watching"
while true; do
  s=$(kubectl get jobs -n $NS --no-headers 2>/dev/null | grep -E "alisachen-sgl-agg-(rr|kv)-bench" | awk '{print $1":"$2}' | tr '\n' ' ')
  say "jobs: $s"
  echo "$s" | grep -qE "Complete|Failed" && ! echo "$s" | grep -q Running && break
  sleep 300
done
say "AGG CHAIN DONE: $s"
