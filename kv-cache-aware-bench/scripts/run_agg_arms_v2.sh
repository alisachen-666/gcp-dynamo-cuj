#!/usr/bin/env bash
# Aggregated arms chain (np-2): wait both stacks ready -> bench rr -> bench kvt -> teardown.
set -u
M=~/kv-cache-aware-bench/manifests
LOG=/tmp/agg_chain.log
NS=dynamo-cloud
say(){ echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }
say AGG CHAIN START
for i in $(seq 1 75); do
  sleep 60
  R=$(kubectl get pods -n $NS -l 'nvidia.com/dynamo-graph-deployment-name in (kimi-k25-agg1n-rr,kimi-k25-agg1n-kvt)' --no-headers 2>/dev/null | awk '$2=="2/2"||$2=="1/1"' | wc -l)
  T=$(kubectl get pods -n $NS -l 'nvidia.com/dynamo-graph-deployment-name in (kimi-k25-agg1n-rr,kimi-k25-agg1n-kvt)' --no-headers 2>/dev/null | wc -l)
  say "[ready $i] $R/$T (want 14)"
  [ "$R" = "14" ] && break
done
for arm in rr kvt; do
  dgd=kimi-k25-agg1n-$arm
  say "=== AGG BENCH $arm ==="
  kubectl delete job alisachen-$dgd-bench -n $NS --ignore-not-found >> "$LOG" 2>&1
  kubectl apply -n $NS -f $M/perf/agg1n-$arm-bench.yaml >> "$LOG" 2>&1
  for i in $(seq 1 200); do
    sleep 60
    st=$(kubectl get job alisachen-$dgd-bench -n $NS -o jsonpath='{.status.succeeded},{.status.failed}' 2>/dev/null)
    say "[$arm bench $i] $st"
    case "$st" in 1,*) say "$arm COMPLETE"; break;; *,2) say "$arm FAILED"; break;; esac
  done
  kubectl logs job/alisachen-alisachen-$dgd-bench -n $NS 2>/dev/null | sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' | grep -E 'Concurrency|Time to First Token \(ms\)|Inter Token|Request Count|Completed|Error Rate|Output Token Through' >> "$LOG" 2>&1
done
say AGG CHAIN DONE
