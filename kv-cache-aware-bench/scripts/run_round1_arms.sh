#!/usr/bin/env bash
# Round-1 official benchmark chain: arms 2B -> 2C -> 2D on the highlight topology.
# Per arm: fresh stack (cold KV per plan), P:D 3:3 (sim highlight regime), GB300
# engine configs, AIPerf sweep at CONCURRENCIES=32,96,128 on the interleaved trace.
# Designed to run under nohup for ~11h; progress + results append to $LOG.
set -u
M=~/kv-cache-aware-bench/manifests
LOG=${LOG:-/tmp/round1.log}
NS=dynamo-cloud
say() { echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }

run_arm() {
  local armfile=$1 dgd=$2 benchfile=$3
  say "=== ARM $dgd: deploy ==="
  kubectl apply -f "$M/operatorless/$armfile" >> "$LOG" 2>&1
  kubectl scale deployment "$dgd-prefill" "$dgd-decode" --replicas=3 -n $NS >> "$LOG" 2>&1
  # cold KV per plan: force fresh pods even if a stack was already running
  kubectl rollout restart deployment "$dgd-frontend" "$dgd-prefill" "$dgd-decode" -n $NS >> "$LOG" 2>&1
  # wait for 7 ready pods (1 FE + 3P + 3D), up to 70 min (cold weight loads)
  local ok=0
  for i in $(seq 1 70); do
    sleep 60
    local ready total
    ready=$(kubectl get pods -n $NS -l "nvidia.com/dynamo-graph-deployment-name=$dgd" --no-headers 2>/dev/null | awk '$2=="2/2" || $2=="1/1"' | wc -l)
    total=$(kubectl get pods -n $NS -l "nvidia.com/dynamo-graph-deployment-name=$dgd" --no-headers 2>/dev/null | wc -l)
    say "[$dgd ready-wait $i] $ready/$total"
    if [ "$ready" = "7" ] && [ "$total" = "7" ]; then ok=1; break; fi
  done
  if [ "$ok" != "1" ]; then say "ARM $dgd: STACK NOT READY — skipping bench, tearing down"; teardown_arm "$armfile" "$dgd"; return 1; fi

  say "=== ARM $dgd: bench (conc 32,96,128) ==="
  kubectl delete job "$dgd-bench" -n $NS --ignore-not-found >> "$LOG" 2>&1
  kubectl apply -f "$M/perf/$benchfile" >> "$LOG" 2>&1
  for i in $(seq 1 300); do
    sleep 60
    local st restarts
    st=$(kubectl get job "$dgd-bench" -n $NS -o jsonpath='{.status.succeeded},{.status.failed}' 2>/dev/null)
    restarts=$(kubectl get pods -n $NS -l "nvidia.com/dynamo-graph-deployment-name=$dgd" --no-headers 2>/dev/null | awk '{s+=$4} END{print s+0}')
    say "[$dgd bench $i] job=$st worker-restarts=$restarts"
    case "$st" in 1,*) say "ARM $dgd: BENCH COMPLETE"; break;; *,2) say "ARM $dgd: BENCH FAILED"; break;; esac
  done
  say "=== ARM $dgd: results ==="
  kubectl logs "job/$dgd-bench" -n $NS 2>/dev/null | sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' \
    | grep -E 'Concurrency|Time to First Token \(ms\)|Inter Token|Request Latency \(ms\)|Request Count|Completed Requests|Error Rate|Output Token Through|Request Through|artifacts in' >> "$LOG" 2>&1
  teardown_arm "$armfile" "$dgd"
}

teardown_arm() {
  local armfile=$1 dgd=$2
  say "=== ARM $dgd: teardown ==="
  kubectl delete -f "$M/operatorless/$armfile" --ignore-not-found >> "$LOG" 2>&1
  # wait for GPU release before next arm
  for i in $(seq 1 15); do
    sleep 20
    local left
    left=$(kubectl get pods -n $NS -l "nvidia.com/dynamo-graph-deployment-name=$dgd" --no-headers 2>/dev/null | wc -l)
    [ "$left" = "0" ] && break
  done
}

say "ROUND-1 CHAIN START"
run_arm arm2b-disagg-rr.yaml       kimi-k25-disagg-rr       arm2b-disagg-rr-bench.yaml
run_arm arm2c-disagg-kv-nvda.yaml  kimi-k25-disagg-kv-nvda  arm2c-disagg-kv-nvda-bench.yaml
run_arm arm2d-disagg-kv-tuned.yaml kimi-k25-disagg-kv-tuned arm2d-disagg-kv-tuned-bench.yaml
say "ROUND-1 CHAIN DONE"
