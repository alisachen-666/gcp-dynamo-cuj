#!/usr/bin/env bash
# RDMA-KV full sweep: after the in-flight collections (m4r 3pts + mxr chain, p8 A/B) release
# the cluster, run every remaining point with GPUDirect RDMA KV transfer:
#   DSR1 llr (low_latency conc 4/8/32/64, one deployment) then DSv4 p1r..p7r (run-point-v2,
#   PASSES=1 — diagnostic sweep; the JIT/stall assertion still stamps VALID/SUSPECT).
# m4r (mid 3 concs), mxr (max_tpt) and p8r come from the earlier chains, completing the matrix.
# Auth-down safe: empty pod listings are only trusted when the API server is reachable.
set -uo pipefail
export KUBECONFIG=$HOME/.kube/REDACTED-GKE-CLUSTER.config
DSR1=$HOME/dsr1-pareto/dsr1-sweep
DSV4=$HOME/dsr1-pareto/dsv4-sweep

echo "=== waiting for m4r/mxr chain AND p8 A/B to release the cluster"
free=0
for i in $(seq 1 400); do   # up to ~33h
  if ! kubectl get ns dynamo-cloud >/dev/null 2>&1; then
    echo "(API/auth unreachable at $(date -u +%H:%MZ); holding)"; free=0; sleep 300; continue
  fi
  busy=$(kubectl get pods -n dynamo-cloud --no-headers 2>/dev/null | grep -E 'dsr1-(m4r|mxr)|dsv4-(p8|p8r)-' | grep -cv Completed || true)
  p8ab=$(pgrep -f run-p8-ab.sh >/dev/null 2>&1 && echo 1 || echo 0)
  if [ "${busy:-0}" = "0" ] && [ "$p8ab" = "0" ]; then free=$((free+1)); else free=0; fi
  [ "$free" -ge 3 ] && break
  sleep 300
done
[ "$free" -ge 3 ] || { echo "!!! cluster never freed for RDMA sweep"; exit 1; }
echo "=== cluster free at $(date -u +%H:%MZ)"

echo "=== [1/8] DSR1 llr"
"$DSR1/run-llr.sh" || echo "!!! llr failed — continuing sweep"

declare -A EXP=( [p1r]=2 [p2r]=7 [p3r]=2 [p4r]=2 [p5r]=3 [p6r]=5 [p7r]=7 )
n=1
for pid in p1r p2r p3r p4r p5r p6r p7r; do
  n=$((n+1))
  echo "=== [$n/8] DSv4 $pid (expect ${EXP[$pid]} workers)"
  "$DSV4/run-point-v2.sh" "$pid" "${EXP[$pid]}" 1 || echo "!!! $pid failed — continuing sweep"
done
echo "=== RDMA SWEEP DONE"
