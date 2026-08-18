#!/bin/bash
# KV ladder resume: points 192/288/384, fresh frontend per point (router-state
# stall mitigation, proven by the flag sweep), per-point warmup + guard gate.
set -u
NS=dynamo-cloud
LOG=/tmp/kv_ladder_v2.log
say() { echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }
for C in 192 288 384; do
  say "=== point conc=$C: fresh frontend"
  kubectl rollout restart deployment/sgl-disagg72-kv-frontend -n $NS >> "$LOG" 2>&1
  kubectl rollout status deployment/sgl-disagg72-kv-frontend -n $NS --timeout=300s >> "$LOG" 2>&1
  sleep 300   # router index rebuild via kv-events replay + settle
  python3 - <<PYEOF
import yaml
docs=list(yaml.safe_load_all(open("$HOME/kv-cache-aware-bench/manifests/perf/sgl-d72-flagsweep.yaml")))
job=docs[0]
job["metadata"]["name"]="alisachen-sgl-d72-kv-p$C"
job["spec"]["template"]["metadata"]["labels"]["app"]="alisachen-sgl-d72-kv-p$C"
for e in job["spec"]["template"]["spec"]["containers"][0]["env"]:
    if e.get("name")=="CONCURRENCIES": e["value"]="$C"
    if e.get("name")=="BENCHMARK_DURATION": e["value"]="1800"
open("/tmp/kv-p$C.yaml","w").write(yaml.dump(job,sort_keys=False))
PYEOF
  kubectl delete job -n $NS alisachen-sgl-d72-kv-p$C --ignore-not-found --wait=true >> "$LOG" 2>&1
  kubectl apply -n $NS -f /tmp/kv-p$C.yaml >> "$LOG" 2>&1
  for i in $(seq 1 60); do
    st=$(kubectl get jobs -n $NS alisachen-sgl-d72-kv-p$C --no-headers 2>/dev/null | awk '{print $2}')
    [ "$st" = "Complete" ] || [ "$st" = "Failed" ] && break
    sleep 120
  done
  say "point $C: $st"
  bash "$HOME/DynamoBench/common/kv-transport-guard.sh" gate "sgl-disagg72-kv" >> "$LOG" 2>&1 && say "gate PASS" || say "gate FAIL at $C - RDMA POLICY VIOLATION"
done
say "KV LADDER V2 DONE"
