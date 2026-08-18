#!/bin/bash
# Live router-flag sweep on the disagg72 KV arm (frontend-only restarts; warm workers).
# Runs AFTER the KV main ladder. Each variant: patch frontend args -> settle ->
# 900s aiperf at conc 192 -> collect. Guard gate enforced per variant.
set -u
NS=dynamo-cloud
LOG=/tmp/flag_sweep.log
say() { echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }
declare -A VARIANTS=(
  [defaults]="--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs"
  [scale2]="--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs --router-prefill-load-scale 2.0"
  [scale3]="--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs --router-prefill-load-scale 3.0"
  [credit08]="--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs --router-kv-overlap-score-credit 0.8"
  [decay08]="--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs --router-kv-overlap-score-credit 1.0 --router-kv-overlap-score-credit-decay 0.8"
  [temp05]="--router-mode kv --router-temperature 0.5 --router-queue-policy fcfs"
)
ORDER="defaults scale2 scale3 credit08 decay08 temp05"
for v in $ORDER; do
  say "=== variant $v: ${VARIANTS[$v]}"
  ARGS_JSON=$(python3 -c "import json;print(json.dumps(['-m','dynamo.frontend']+'''${VARIANTS[$v]}'''.split()+['--request-plane','nats']))")
  kubectl patch deployment sgl-disagg72-kv-frontend -n $NS --type=json \
    -p "[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/args\",\"value\":$ARGS_JSON}]" >> "$LOG" 2>&1
  kubectl rollout status deployment/sgl-disagg72-kv-frontend -n $NS --timeout=300s >> "$LOG" 2>&1
  say "frontend up; settling 300s (router index rebuild via kv-events replay)"
  sleep 300
  kubectl delete job -n $NS alisachen-sgl-d72-flagsweep --ignore-not-found --wait=true >> "$LOG" 2>&1
  kubectl apply -n $NS -f "$HOME/kv-cache-aware-bench/manifests/perf/sgl-d72-flagsweep.yaml" >> "$LOG" 2>&1
  for i in $(seq 1 40); do
    st=$(kubectl get jobs -n $NS alisachen-sgl-d72-flagsweep --no-headers 2>/dev/null | awk '{print $2}')
    [ "$st" = "Complete" ] && break
    [ "$st" = "Failed" ] && { say "variant $v bench FAILED"; break; }
    sleep 120
  done
  say "variant $v done (job=$st); transport gate:"
  bash "$HOME/DynamoBench/common/kv-transport-guard.sh" gate "sgl-disagg72-kv" >> "$LOG" 2>&1 && say "gate PASS" || say "gate FAIL - RDMA policy violation on $v"
  # tag latest artifact dir with the variant name marker
  latest=$(gcloud storage ls gs://alisachen-models/perf/ 2>/dev/null | grep flagsweep | tail -1)
  [ -n "$latest" ] && echo "$v" | gcloud storage cp - "${latest}VARIANT.txt" 2>/dev/null
done
say "FLAG SWEEP DONE"
