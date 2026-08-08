#!/usr/bin/env bash
# Run one DSR1-FP4 point end-to-end on REDACTED-GKE-CLUSTER: deploy -> registrations -> frontend restart
# -> hard gate -> bench -> extract -> upload (ADC) -> teardown.
# Mirrors dsv4-sweep/run-point-v2.sh structure, adapted for the dsr1-* naming and manifests.
# Usage: run-dsr1-point.sh <name> <expected_workers> <server_manifest> <bench_manifest> <bench_job>
#   e.g. run-dsr1-point.sh m4n 7  manifests/m4n-midcurve-nats.yaml manifests/m4n-bench-conc2048.yaml dsr1-m4n-bench-c2048
#        run-dsr1-point.sh mx  11 manifests/mx-maxtpt.yaml         manifests/mx-bench-conc2048.yaml  dsr1-mx-bench-c2048
set -uo pipefail
NAME=$1; EXP=$2; SRV=$3; BEN=$4; JOB=$5
D=$HOME/dsr1-pareto/dsr1-sweep
DEST="gs://gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs"
S=$(mktemp -d)
PFX="dsr1-$NAME"

echo "=== deploying $NAME"
kubectl apply -f "$D/$SRV" >/dev/null
echo "=== waiting for $EXP worker registrations"
for i in $(seq 1 120); do
  n=$(kubectl exec -n dynamo-cloud dynamo-platform-etcd-0 -- etcdctl --endpoints=localhost:2379 get v1/instances/ --prefix --keys-only 2>/dev/null | grep -c generate)
  [ "${n:-0}" -ge "$EXP" ] && break
  bad=$(kubectl get pods -n dynamo-cloud --no-headers 2>/dev/null | grep -E "$PFX-(prefill|decode)" | grep -cE "CrashLoop|Error" || true)
  if [ "$bad" != "0" ]; then
    echo "!!! $NAME WORKER FAILURE"; kubectl get pods -n dynamo-cloud | grep "$PFX" | grep -E "CrashLoop|Error"
    exit 1
  fi
  sleep 60
done
echo "=== $n registered; restarting frontends"
kubectl rollout restart "deployment/$PFX-frontend" -n dynamo-cloud >/dev/null
kubectl rollout status "deployment/$PFX-frontend" -n dynamo-cloud --timeout=420s >/dev/null
ok=0
for i in $(seq 1 60); do
  r=$(kubectl run "probe-$NAME-$i" -n dynamo-cloud --image=curlimages/curl --restart=Never --rm -q -i --timeout=180s -- \
      -s -m 120 -X POST "http://$PFX-frontend.dynamo-cloud.svc.cluster.local:8000/v1/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"deepseek-ai/DeepSeek-R1","prompt":"hi","max_tokens":4,"stream":false}' 2>/dev/null)
  if echo "$r" | grep -q '"text"'; then ok=$((ok+1)); else ok=0; fi
  [ "$ok" -ge 3 ] && { echo "=== serving verified (3 consecutive completions)"; break; }
  sleep 20
done
[ "$ok" -ge 3 ] || { echo "!!! $NAME never served completions — aborting"; exit 1; }

# RDMA points (name ends in 'r'): KV must ride RDMA — gate + in-bench watchdog (stop policy)
GUARD=$HOME/dsr1-pareto/common/kv-transport-guard.sh
GPID=""
case "$NAME" in *r)
  echo "=== KV-transport gate"
  "$GUARD" gate "$PFX" || { echo "!!! $NAME ABORTED: KV path not clean RDMA"; exit 1; } ;;
esac
echo "=== launching bench $JOB"
kubectl delete job "$JOB" -n dynamo-cloud --ignore-not-found >/dev/null 2>&1
kubectl apply -f "$D/$BEN" >/dev/null
case "$NAME" in *r)
  export GUARD_MARKER="$S/kv-guard-tripped"
  "$GUARD" watch "$PFX" "$JOB" 180 & GPID=$! ;;
esac
for i in $(seq 1 360); do
  st=$(kubectl get job "$JOB" -n dynamo-cloud -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null)
  [ -n "$st" ] && break
  [ -n "$GPID" ] && [ -f "$GUARD_MARKER" ] && break
  sleep 60
done
[ -n "$GPID" ] && kill "$GPID" >/dev/null 2>&1
if [ -n "$GPID" ] && [ -f "$GUARD_MARKER" ]; then
  echo "!!! $NAME FAILED: KV transfer violation mid-bench (guard killed the job)"
fi
echo "=== bench terminal: ${st:-TIMEOUT}"
mkdir -p "$D/results"
kubectl logs -n dynamo-cloud "job/$JOB" > "$S/bench-$NAME.log" 2>/dev/null
python3 - "$NAME" "$S/bench-$NAME.log" "$D/results" <<'EOF'
import json, re, sys
name, logf, out = sys.argv[1], sys.argv[2], sys.argv[3]
log = open(logf).read()
m = re.search(r'--- RESULT JSON\n(\{.*?\})\n', log, re.S)
found = 0
if m:
    d = json.loads(m.group(1)); found = 1
    dims = 'gpus_72_ctx_40_gen_32' if name in ('mx', 'mxk', 'mxt', 'mxr') else 'gpus_72_ctx_24_gen_48'
    suffix = {'m4n': '_nats', 'mxk': '_kvheadroom', 'mxt': '_tcp', 'm4r': '_rdmakv', 'mxr': '_rdmakv'}.get(name, '')
    fn = f"{out}/{name}-results_concurrency_{d['max_concurrency']}_{dims}{suffix}.json"
    json.dump(d, open(fn, 'w'))
    dg = 32 if name in ('mx', 'mxk', 'mxt', 'mxr') else 48
    pg = 40 if name in ('mx', 'mxk', 'mxt', 'mxr') else 24
    print(f"{name} conc {d['max_concurrency']}: out/decode-gpu {d['output_throughput']/dg:.1f} | "
          f"in/prefill-gpu {d['total_input_tokens']/d['duration']/pg:.0f} | "
          f"TPOT {d['mean_tpot_ms']:.2f}ms | TTFT med {d['median_ttft_ms']/1000:.1f}s | E2E {d['mean_e2el_ms']/1000:.1f}s")
print(f"[{name}] extracted {found} result(s)")
EOF
mkdir -p "$S/srv"; for p in $(kubectl get pods -n dynamo-cloud --no-headers | grep "$PFX" | awk '{print $1}'); do kubectl logs -n dynamo-cloud "$p" > "$S/srv/$p.log" 2>/dev/null; done
# transport evidence: confirm request plane actually used
grep -h -m4 -iE "request plane|NetworkManager|rc_mlx5|rc_x|cuda_ipc" "$S/srv/$PFX-decode-0.log" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | head -3
export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token 2>/dev/null)
gcloud storage rsync -r "$S/srv" "$DEST/server-logs/$PFX" 2>&1 | tail -1
gcloud storage cp "$S/bench-$NAME.log" "$DEST/client-logs/$PFX-bench.log" 2>&1 | tail -1
gcloud storage rsync -r "$D/results" "$DEST/results" 2>&1 | tail -1
echo "=== tearing down $NAME"
kubectl delete -n dynamo-cloud "deployment/$PFX-frontend" "deployment/$PFX-prefill" >/dev/null 2>&1
kubectl delete -n dynamo-cloud "statefulset/$PFX-decode" "deployment/$PFX-decode" >/dev/null 2>&1
kubectl delete -n dynamo-cloud "computedomain/$PFX-cd" "service/$PFX-frontend" "service/$PFX-decode" "job/$JOB" >/dev/null 2>&1
rm -rf "$S"
echo "=== $NAME DONE"
