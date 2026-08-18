#!/bin/bash
# RR knee-location down-sweep: wait for KV ladder v2, swap arms on np-3,
# run conc 12/24/48/72 with the v2 protocol (fresh frontend per point).
set -u
NS=dynamo-cloud
LOG=/tmp/rr_downsweep.log
say() { echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }
say "waiting for KV ladder v2"
until grep -q "KV LADDER V2 DONE" /tmp/kv_ladder_v2.log 2>/dev/null; do sleep 300; done
say "swapping arms: KV down, RR up on np-3"
kubectl scale deployment -n $NS sgl-disagg72-kv-prefill sgl-disagg72-kv-decode sgl-disagg72-kv-frontend --replicas=0 >> "$LOG" 2>&1
sleep 120
kubectl scale deployment -n $NS sgl-disagg72-rr-frontend --replicas=1 >> "$LOG" 2>&1
kubectl scale deployment -n $NS sgl-disagg72-rr-prefill --replicas=6 >> "$LOG" 2>&1
kubectl scale deployment -n $NS sgl-disagg72-rr-decode --replicas=12 >> "$LOG" 2>&1
for i in $(seq 1 60); do
  ready=$(kubectl get pods -n $NS -l "nvidia.com/dynamo-graph-deployment-name=sgl-disagg72-rr" --no-headers 2>/dev/null | awk '$2=="2/2"' | wc -l)
  say "[rr-up $i] $ready/19"; [ "$ready" -ge 19 ] && break; sleep 120
done
[ "$ready" -ge 19 ] || { say "RR STACK TIMEOUT"; exit 1; }
for C in 12 24 48 72; do
  say "=== RR point conc=$C"
  kubectl rollout restart deployment/sgl-disagg72-rr-frontend -n $NS >> "$LOG" 2>&1
  kubectl rollout status deployment/sgl-disagg72-rr-frontend -n $NS --timeout=300s >> "$LOG" 2>&1
  sleep 240
  python3 - <<PYEOF
import yaml
docs=list(yaml.safe_load_all(open("$HOME/kv-cache-aware-bench/manifests/perf/sgl-d72-flagsweep.yaml")))
job=docs[0]
job["metadata"]["name"]="alisachen-sgl-d72-rr-p$C"
job["spec"]["template"]["metadata"]["labels"]["app"]="alisachen-sgl-d72-rr-p$C"
for e in job["spec"]["template"]["spec"]["containers"][0]["env"]:
    if e.get("name")=="CONCURRENCIES": e["value"]="$C"
    if e.get("name")=="BENCHMARK_DURATION": e["value"]="1800"
    if e.get("name")=="TARGET_ENDPOINT" or (e.get("name")=="ENDPOINT"): pass
# retarget the frontend service to the RR arm
s=job["spec"]["template"]["spec"]["containers"][0]["command"][-1] if job["spec"]["template"]["spec"]["containers"][0].get("command") else None
for e in job["spec"]["template"]["spec"]["containers"][0]["env"]:
    if isinstance(e.get("value"),str) and "sgl-disagg72-kv-frontend" in e["value"]:
        e["value"]=e["value"].replace("sgl-disagg72-kv-frontend","sgl-disagg72-rr-frontend")
if s and "sgl-disagg72-kv-frontend" in s:
    job["spec"]["template"]["spec"]["containers"][0]["command"][-1]=s.replace("sgl-disagg72-kv-frontend","sgl-disagg72-rr-frontend")
open("/tmp/rr-p$C.yaml","w").write(yaml.dump(job,sort_keys=False))
PYEOF
  kubectl delete job -n $NS alisachen-sgl-d72-rr-p$C --ignore-not-found --wait=true >> "$LOG" 2>&1
  kubectl apply -n $NS -f /tmp/rr-p$C.yaml >> "$LOG" 2>&1
  for i in $(seq 1 60); do
    st=$(kubectl get jobs -n $NS alisachen-sgl-d72-rr-p$C --no-headers 2>/dev/null | awk '{print $2}')
    [ "$st" = "Complete" ] || [ "$st" = "Failed" ] && break
    sleep 120
  done
  say "RR point $C: $st"
  bash "$HOME/DynamoBench/common/kv-transport-guard.sh" gate "sgl-disagg72-rr" >> "$LOG" 2>&1 && say "gate PASS" || say "gate FAIL at $C"
done
say "RR DOWNSWEEP DONE"
