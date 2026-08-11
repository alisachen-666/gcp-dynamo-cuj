#!/bin/bash
# Exits (with labeled reason) on first notable event: crash, chain failure,
# bench job failure, or milestone (S1 pass, bench launch/completion).
NS=dynamo-cloud
seen_s1=""
while true; do
  # 1. container crashes / restart growth on sgl pods
  bad=$(kubectl get pods -n $NS --no-headers 2>/dev/null | grep -E "^sgl-" | \
        awk '$3~/CrashLoop|Error|ImagePull|CreateContainer/ || $4>1 {print $1" "$3" restarts="$4}')
  [ -n "$bad" ] && { echo "EVENT=POD_ERROR"; echo "$bad"; exit 0; }
  # 2. smoke outcome (once: marker file suppresses repeats across monitor restarts)
  if [ ! -f /tmp/.s1_reported ] && grep -qE "S1 (PASS|FAIL)" /tmp/sgl_smoke_s1.log 2>/dev/null; then
    touch /tmp/.s1_reported
    echo "EVENT=S1_$(grep -oE 'S1 (PASS|FAIL)' /tmp/sgl_smoke_s1.log | head -1 | awk '{print $2}')"
    tail -8 /tmp/sgl_smoke_s1.log; exit 0
  fi
  # 3. agg chain failure markers
  grep -qE "aborting|timeout|not ready" /tmp/sgl_agg_chain.log 2>/dev/null && \
    { echo "EVENT=AGG_CHAIN_PROBLEM"; tail -4 /tmp/sgl_agg_chain.log; exit 0; }
  # 4. sgl bench jobs: failure or completion
  js=$(kubectl get jobs -n $NS --no-headers 2>/dev/null | grep "alisachen-sgl-")
  echo "$js" | grep -q Failed && { echo "EVENT=BENCH_FAILED"; echo "$js"; exit 0; }
  n=$(echo "$js" | grep -c Complete)
  [ "$n" -ge 2 ] && { echo "EVENT=ALL_AGG_BENCHES_COMPLETE"; echo "$js"; exit 0; }
  # 5. gcsfuse mount deaths (transport endpoint) on any sgl pod event
  kubectl get events -n $NS --field-selector reason=Failed 2>/dev/null | \
    grep -q "transport endpoint is not connected" && \
    { echo "EVENT=GCSFUSE_MOUNT_DEATH"; exit 0; }
  sleep 180
done
