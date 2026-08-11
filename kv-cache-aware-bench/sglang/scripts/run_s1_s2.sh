#!/bin/bash
set -u
bash "$HOME/kv-cache-aware-bench/sglang/scripts/run_smoke_s1.sh" || exit 1
kubectl apply -n dynamo-cloud -f "$HOME/kv-cache-aware-bench/manifests/perf/sgl-smoke2-mini.yaml" >> /tmp/sgl_smoke_s1.log 2>&1
echo "[$(date -u +%H:%M:%S)] S2 job launched" >> /tmp/sgl_smoke_s1.log
