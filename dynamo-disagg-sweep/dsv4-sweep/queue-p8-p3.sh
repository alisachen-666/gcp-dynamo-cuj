#!/usr/bin/env bash
# Queue: wait for the in-flight p5 run to finish (its runner tears everything down),
# then p8 (corrected config, 1 pass + assertion), then p3 (2 passes: cache-filler + record).
set -uo pipefail
export PATH=$HOME/google-cloud-sdk/bin:$PATH
D=$HOME/dsr1-pareto/dsv4-sweep
echo "[queue] waiting for p5 to finish (poll 120s, max 6h)"
for i in $(seq 1 180); do
  left=$(kubectl get deploy,sts,job -n dynamo-cloud -o name 2>/dev/null | grep -c "dsv4-p5" || true)
  [ "${left:-1}" = "0" ] && break
  sleep 120
done
left=$(kubectl get deploy,sts,job -n dynamo-cloud -o name 2>/dev/null | grep -c "dsv4-p5" || true)
[ "$left" = "0" ] || { echo "[queue] p5 still present after 6h — aborting queue"; exit 1; }
echo "[queue] p5 clear. launching p8 (EXP=9, passes=1)"
bash "$D/run-point-v2.sh" p8 9 1
echo "[queue] p8 done. launching p3 (EXP=2, passes=2: cache-filler + record)"
bash "$D/run-point-v2.sh" p3 2 2
echo "[queue] ALL DONE (p8 + p3)"
