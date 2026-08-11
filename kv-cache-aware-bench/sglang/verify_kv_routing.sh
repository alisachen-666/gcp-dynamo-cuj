#!/bin/bash
# Post-bench verification that KV-aware routing was REALLY active during a run.
# Usage: verify_kv_routing.sh <kv-arm> <rr-arm> <worker-container> [gcs-perf-root]
#   e.g. verify_kv_routing.sh sgl-agg-kv sgl-agg-rr agg
#        verify_kv_routing.sh sgl-disagg72-kv sgl-disagg72-rr prefill
# Three layers: config -> events -> behavior. Behavior is the verdict; the first
# two localize the failure when behavior fails.
set -u
NS=dynamo-cloud
KV=${1:?kv arm}; RR=${2:?rr arm}; CT=${3:?worker container}
echo "=== Layer 1: config"
kv_on=$(kubectl logs -n $NS -l "nvidia.com/dynamo-graph-deployment-name=$KV" -c $CT --tail 2000 2>/dev/null | grep -c "use_kv_events=True")
fe_mode=$(kubectl get deploy -n $NS $KV-frontend -o jsonpath='{.spec.template.spec.containers[0].args}' 2>/dev/null | grep -o '"kv"' | head -1)
echo "  workers use_kv_events=True lines: $kv_on (want >0)"
echo "  frontend --router-mode kv: ${fe_mode:-MISSING}"

echo "=== Layer 2: events"
pub=$(kubectl logs -n $NS -l "nvidia.com/dynamo-graph-deployment-name=$KV" -c $CT --tail 4000 2>/dev/null | grep -cE "EventPublisher.*kv.?events|KvEventPublisher|topic=kv_events")
echo "  workers publishing kv_events topic: $pub lines (want >0; kv_metrics alone is NOT enough)"

echo "=== Layer 2.5: router telemetry (Prometheus, live during the run)"
FE=$(kubectl get pods -n $NS -l app=$KV-frontend -o name 2>/dev/null | head -1)
kubectl exec -n $NS ${FE#pod/} -- curl -s http://localhost:8000/metrics 2>/dev/null | \
  awk '/^dynamo_component_router_kv_hit_rate_bucket.*le="0"\}/ {z+=$2}
       /^dynamo_component_router_kv_hit_rate_count/ {n+=$2}
       /^dynamo_component_router_kv_hit_rate_sum/ {s+=$2}
       END {if(n>0) printf "  routed=%d mean_predicted_hit_rate=%.3f zero_hit_requests=%d (%.0f%%)\n  VERDICT: %s\n", n, s/n, z, 100*z/n, (s/n>0.3?"KV ROUTING ACTIVE":"KV ROUTING NOT EFFECTIVE (index empty or no reuse)")}'

echo "=== Layer 3: behavior (the verdict)"
# 3a. engine cache-hit ratio per arm from prefill batch logs
for arm in $KV $RR; do
  L=$(kubectl logs -n $NS -l "nvidia.com/dynamo-graph-deployment-name=$arm" -c $CT --tail 6000 2>/dev/null | grep -oE "#new-token: [0-9]+, #cached-token: [0-9]+")
  new=$(echo "$L" | awk -F'[ ,]' '{s+=$2} END {print s+0}')
  cached=$(echo "$L" | grep -oE "cached-token: [0-9]+" | awk '{s+=$2} END {print s+0}')
  tot=$((new+cached))
  [ "$tot" -gt 0 ] && pct=$((100*cached/tot)) || pct=NA
  echo "  $arm: cached=$cached new=$new reuse=${pct}%"
done
# 3b. request-distribution skew across workers (KV affinity concentrates sessions)
for arm in $KV $RR; do
  kubectl logs -n $NS -l "nvidia.com/dynamo-graph-deployment-name=$arm" -c $CT --tail 4000 --prefix 2>/dev/null | \
    grep -oE "^\[pod/[^]]+\].*request received" | awk -F'[][]' '{print $2}' | sort | uniq -c | \
    awk -v arm=$arm '{n++; s+=$1; ss+=$1*$1} END {if(n>0){m=s/n; sd=sqrt(ss/n-m*m); printf "  %s: req/worker mean=%.0f stdev=%.0f cv=%.2f (KV should skew: cv_kv >> cv_rr)\n", arm, m, sd, sd/m}}'
done
echo "=== Interpretation:"
echo "  PASS = kv arm reuse >= 40% AND rr arm reuse < kv arm by >= 20 points"
echo "         AND (from aiperf JSONs) kv TTFT p50 <= 0.7x rr TTFT p50 at same conc"
echo "  Compare aiperf TTFT: gs://alisachen-models/perf/<run>/<point>/profile_export_aiperf.json"
