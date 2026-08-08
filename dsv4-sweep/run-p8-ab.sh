#!/usr/bin/env bash
# P8 A/B: (1) MNNVL record rerun of p8 — recovers the measured total/input throughput lost
# to log rotation (bench-p8.yaml now has the SLIM tail-print); (2) p8r RDMA arm (mooncake
# MC_FORCE_MNNVL=0 + 8x mrdma DRA claims). Both PASSES=2 (cache-filler + record).
# Waits for the DSR1 m4r/mxr chain to free the np-2 pool first. Auth-down safe: an empty
# pod listing is only trusted when the API server is actually reachable.
set -uo pipefail
export KUBECONFIG=$HOME/.kube/REDACTED-GKE-CLUSTER.config
D=$HOME/dsr1-pareto/dsv4-sweep

echo "=== waiting for DSR1 m4r/mxr chain to release nodes"
free=0
for i in $(seq 1 200); do   # up to ~16h40m
  if ! kubectl get ns dynamo-cloud >/dev/null 2>&1; then
    echo "(API/auth unreachable at $(date -u +%H:%MZ); holding)"; free=0; sleep 300; continue
  fi
  busy=$(kubectl get pods -n dynamo-cloud --no-headers 2>/dev/null | grep -E 'dsr1-(m4r|mxr)' | grep -cv Completed || true)
  if [ "${busy:-0}" = "0" ]; then free=$((free+1)); else free=0; fi
  [ "$free" -ge 2 ] && break
  sleep 300
done
[ "$free" -ge 2 ] || { echo "!!! DSR1 chain never released nodes"; exit 1; }

echo "=== nodes free at $(date -u +%H:%MZ); stage 1: p8 MNNVL record (PASSES=2)"
"$D/run-point-v2.sh" p8 9 2
echo "=== stage 1 exit: $?; stage 2: p8r RDMA arm (PASSES=2)"
"$D/run-point-v2.sh" p8r 9 2
echo "=== stage 2 exit: $?"
echo "=== P8 A/B DONE"
