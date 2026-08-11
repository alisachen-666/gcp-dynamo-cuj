#!/usr/bin/env bash
# Run one DSv4 point end-to-end — v2 of run-point.sh. DO NOT edit run-point.sh while a run
# is in flight (bash reads scripts lazily); v2 exists so p5's in-flight run stays untouched.
#
# v2 additions (2026-08-01, after the p3 JIT root-cause):
#   1. PASSES arg: run the full bench sequence N times against ONE standing deployment.
#      Passes 1..N-1 are CACHE FILLERS (results discarded) — they force every batch shape
#      through the DeepGEMM/flashinfer JIT compilers (once-per-shape-per-process) so the
#      final RECORD pass measures a compile-free server. Use PASSES=2 for record-quality
#      reruns (p3); PASSES=1 matches v1 behaviour.
#   2. JIT/stall assertion: after the record pass, every measured window in the bench log is
#      checked against every prefill pod's log for (a) "Entering DeepGEMM JIT" sessions and
#      (b) >5s gaps in the prefill-batch reporter. Result is stamped VALID or SUSPECT —
#      a SUSPECT record run should be re-recorded (the standing deployment makes that cheap).
#   3. GCS uploads authenticate via ADC (application-default token minted at upload time);
#      v1 used the gcloud CLI account, which on this VM is the scope-limited compute SA.
#
# Usage: run-point-v2.sh <pid> <expected_workers> [passes]
set -uo pipefail
PID=$1
EXP=${2:-2}
PASSES=${3:-1}
D=$HOME/dsr1-pareto/dsv4-sweep
DEST="gs://gke-aishared-gsc-dev/mlperf_results/nvfp4_20260519_run5/logs/logs"
S=$(mktemp -d)

echo "=== deploying $PID (passes: $PASSES)"
kubectl apply -f "$D/manifests/$PID.yaml" >/dev/null
echo "=== waiting for $EXP worker registrations"
for i in $(seq 1 120); do
  n=$(kubectl exec -n dynamo-cloud dynamo-platform-etcd-0 -- etcdctl --endpoints=localhost:2379 get v1/instances/ --prefix --keys-only 2>/dev/null | grep -c generate)
  [ "${n:-0}" -ge "$EXP" ] && break
  bad=$(kubectl get pods -n dynamo-cloud --no-headers 2>/dev/null | grep -E "dsv4-$PID-(prefill|decode)" | grep -cE "CrashLoop|Error" || true)
  if [ "$bad" != "0" ]; then
    echo "!!! $PID WORKER FAILURE"; kubectl get pods -n dynamo-cloud | grep "dsv4-$PID" | grep -E "CrashLoop|Error"
    kubectl logs -n dynamo-cloud "deploy/dsv4-$PID-prefill" --tail=20 2>/dev/null | grep -E "Error|Traceback" | head -5
    exit 1
  fi
  sleep 60
done
echo "=== $n registered; restarting frontends"
kubectl rollout restart "deployment/dsv4-$PID-frontend" -n dynamo-cloud >/dev/null
kubectl rollout status "deployment/dsv4-$PID-frontend" -n dynamo-cloud --timeout=420s >/dev/null
# HARD GATE: 3 consecutive real completions through the Service.
ok=0
for i in $(seq 1 60); do
  r=$(kubectl run "probe-$PID-$i" -n dynamo-cloud --image=curlimages/curl --restart=Never --rm -q -i --timeout=180s -- \
      -s -m 120 -X POST "http://dsv4-$PID-frontend.dynamo-cloud.svc.cluster.local:8000/v1/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"deepseek-ai/DeepSeek-V4-Pro","prompt":"hi","max_tokens":4,"stream":false}' 2>/dev/null)
  if echo "$r" | grep -q '"text"'; then ok=$((ok+1)); else ok=0; fi
  [ "$ok" -ge 3 ] && { echo "=== serving verified (3 consecutive completions)"; break; }
  sleep 20
done
[ "$ok" -ge 3 ] || { echo "!!! $PID never served completions — aborting point"; exit 1; }

# RDMA points (pid ends in 'r'): KV path must be RDMA — gate + in-bench watchdog. A run
# whose KV transfer degrades to TCP is killed and fails, never recorded (user directive).
GUARD=$HOME/dsr1-pareto/common/kv-transport-guard.sh
case "$PID" in *r)
  echo "=== KV-transport gate ($PID is an RDMA point)"
  "$GUARD" gate "dsv4-$PID" || { echo "!!! $PID ABORTED: KV path not clean RDMA"; exit 1; } ;;
esac

for pass in $(seq 1 "$PASSES"); do
  if [ "$pass" -lt "$PASSES" ]; then TAG="CACHE-FILLER $pass/$((PASSES-1))"; else TAG="RECORD"; fi
  echo "=== bench pass $pass ($TAG)"
  kubectl delete job "dsv4-bench-$PID" -n dynamo-cloud --ignore-not-found >/dev/null 2>&1
  # wait for job object to be fully gone before re-applying
  for i in $(seq 1 30); do
    kubectl get job "dsv4-bench-$PID" -n dynamo-cloud >/dev/null 2>&1 || break
    sleep 5
  done
  kubectl apply -f "$D/manifests/bench-$PID.yaml" >/dev/null
  GPID=""
  case "$PID" in *r)
    export GUARD_MARKER="$S/kv-guard-tripped-p$pass"
    "$GUARD" watch "dsv4-$PID" "dsv4-bench-$PID" 180 & GPID=$! ;;
  esac
  for i in $(seq 1 240); do
    st=$(kubectl get job "dsv4-bench-$PID" -n dynamo-cloud -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null)
    [ -n "$st" ] && break
    [ -n "$GPID" ] && [ -f "$GUARD_MARKER" ] && break
    sleep 60
  done
  [ -n "$GPID" ] && kill "$GPID" >/dev/null 2>&1
  if [ -n "$GPID" ] && [ -f "$GUARD_MARKER" ]; then
    echo "!!! $PID FAILED: KV transfer degraded to TCP mid-bench (guard killed the job)"
    kubectl delete -n dynamo-cloud "deployment/dsv4-$PID-frontend" "deployment/dsv4-$PID-prefill" >/dev/null 2>&1
    kubectl delete -n dynamo-cloud "statefulset/dsv4-$PID-decode" "deployment/dsv4-$PID-decode" >/dev/null 2>&1
    kubectl delete -n dynamo-cloud "computedomain/dsv4-$PID-cd" "service/dsv4-$PID-frontend" "service/dsv4-$PID-decode" >/dev/null 2>&1
    rm -rf "$S"; exit 1
  fi
  echo "=== bench pass $pass terminal: ${st:-TIMEOUT}"
  if [ "$pass" -lt "$PASSES" ]; then
    kubectl logs -n dynamo-cloud "job/dsv4-bench-$PID" > "$S/bench-$PID-filler$pass.log" 2>/dev/null
  else
    kubectl logs -n dynamo-cloud "job/dsv4-bench-$PID" > "$S/bench-$PID.log" 2>/dev/null
  fi
done

# extract results from the RECORD pass only
python3 - "$PID" "$S/bench-$PID.log" <<'EOF'
import json, re, sys, os
pid, logf = sys.argv[1], sys.argv[2]
log = open(logf).read()
out = os.path.expanduser('~/dsr1-pareto/dsv4-sweep/results')
found = 0
for m in re.finditer(r'--- /tmp/results/(results_concurrency_\d+_gpus_\d+_ctx_\d+_gen_\d+)\.json\n(\{.*?\})\n', log, re.S):
    d = json.loads(m.group(2)); found += 1
    json.dump(d, open(f'{out}/{pid}-{m.group(1)}.json', 'w'))
    g = re.search(r'gpus_(\d+)_ctx_(\d+)_gen_(\d+)', m.group(1)); tot, pg, dg = map(int, g.groups())
    intps = d['total_input_tokens']/d['duration']
    print(f"{pid} conc {d['max_concurrency']}: out/decode-gpu {d['output_throughput']/dg:.1f} | in/prefill-gpu {intps/pg:.0f} | total/gpu {d['total_token_throughput']/tot:.1f} | TPOT {d['mean_tpot_ms']:.2f}ms | TTFT med {d['median_ttft_ms']/1000:.1f}s")
print(f"[{pid}] extracted {found} result(s)")
EOF

# collect server logs BEFORE teardown (needed for the assertion + archive)
mkdir -p "$S/srv"; for p in $(kubectl get pods -n dynamo-cloud --no-headers | grep "dsv4-$PID" | awk '{print $1}'); do kubectl logs -n dynamo-cloud "$p" > "$S/srv/$p.log" 2>/dev/null; done

# JIT/stall assertion over every measured window in the record pass
python3 - "$PID" "$S/bench-$PID.log" "$S/srv" <<'EOF'
import datetime, glob, json, re, sys
pid, benchlog, srvdir = sys.argv[1], sys.argv[2], sys.argv[3]
log = open(benchlog).read()
windows = []
for m in re.finditer(r'\{"date": "(\d{8}-\d{6})".*?"duration": ([0-9.]+)', log):
    end = datetime.datetime.strptime(m.group(1), '%Y%m%d-%H%M%S')
    windows.append((end - datetime.timedelta(seconds=float(m.group(2))), end))
ansi = re.compile(r'\x1b\[[0-9;]*m')
verdict = 'VALID'
for f in glob.glob(f'{srvdir}/*prefill*.log'):
    jits, batches = [], []
    for line in open(f, errors='replace'):
        line = ansi.sub('', line)
        m = re.match(r'(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+)Z', line)
        if not m: continue
        t = datetime.datetime.fromisoformat(m.group(1))
        if 'Entering DeepGEMM JIT' in line: jits.append(t)
        elif 'report_prefill_stats' in line: batches.append(t)
    for (a, b) in windows:
        for t in jits:
            if a <= t <= b:
                print(f"!!! ASSERT: JIT compile at {t.time()} INSIDE measured window {a.time()}-{b.time()} ({f.split('/')[-1]})"); verdict = 'SUSPECT'
        w = [t for t in batches if a <= t <= b]
        for x, y in zip(w, w[1:]):
            gap = (y - x).total_seconds()
            if gap > 5:
                print(f"!!! ASSERT: prefill reporter silent {gap:.1f}s ({x.time()} -> {y.time()}) inside measured window ({f.split('/')[-1]})"); verdict = 'SUSPECT'
print(f"[{pid}] measured windows checked: {len(windows)}; RECORD RUN VERDICT: {verdict}")
if verdict != 'VALID':
    print(f"[{pid}] SUSPECT record: re-run the bench against the standing deployment (cheap) before accepting.")
EOF

# upload with ADC (the CLI account on this VM is the scope-limited compute SA)
export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token 2>/dev/null)
gcloud storage rsync -r "$S/srv" "$DEST/server-logs/dsv4-$PID" 2>&1 | tail -1
for fl in "$S"/bench-$PID*.log; do gcloud storage cp "$fl" "$DEST/client-logs/dsv4-$(basename "$fl")" 2>&1 | tail -1; done
gcloud storage rsync -r "$D/results" "$DEST/results/dsv4" 2>&1 | tail -1
echo "=== tearing down $PID"
kubectl delete -n dynamo-cloud "deployment/dsv4-$PID-frontend" "deployment/dsv4-$PID-prefill" >/dev/null 2>&1
kubectl delete -n dynamo-cloud "statefulset/dsv4-$PID-decode" "deployment/dsv4-$PID-decode" >/dev/null 2>&1
kubectl delete -n dynamo-cloud "computedomain/dsv4-$PID-cd" "service/dsv4-$PID-frontend" "service/dsv4-$PID-decode" "job/dsv4-bench-$PID" >/dev/null 2>&1
rm -rf "$S"
echo "=== $PID DONE"
