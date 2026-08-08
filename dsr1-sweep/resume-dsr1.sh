#!/usr/bin/env bash
# Resume a dsr1 point whose runner aborted pre-bench (deployment standing).
# Usage: resume-dsr1.sh <name> <expected> <bench_manifest> <bench_job>
set -uo pipefail
export PATH=$HOME/google-cloud-sdk/bin:$PATH
NAME=$1; EXP=$2; BEN=$3; JOB=$4
D=$HOME/dsr1-pareto/dsr1-sweep; DEST="gs://REDACTED-GCS-BUCKET/mlperf_results/nvfp4_20260519_run5/logs/logs"
S=$(mktemp -d); PFX="dsr1-$NAME"
echo "=== waiting for $EXP registrations"
for i in $(seq 1 120); do
  n=$(kubectl exec -n dynamo-cloud dynamo-platform-etcd-0 -- etcdctl --endpoints=localhost:2379 get v1/instances/ --prefix --keys-only 2>/dev/null | grep -c generate)
  echo "reg: ${n:-0}"; [ "${n:-0}" -ge "$EXP" ] && break; sleep 60
done
kubectl rollout restart "deployment/$PFX-frontend" -n dynamo-cloud >/dev/null
kubectl rollout status "deployment/$PFX-frontend" -n dynamo-cloud --timeout=420s >/dev/null
ok=0
for i in $(seq 1 90); do
  r=$(kubectl run "probe-$NAME-r$i" -n dynamo-cloud --image=curlimages/curl --restart=Never --rm -q -i --timeout=180s -- \
      -s -m 120 -X POST "http://$PFX-frontend.dynamo-cloud.svc.cluster.local:8000/v1/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"deepseek-ai/DeepSeek-R1","prompt":"hi","max_tokens":4,"stream":false}' 2>/dev/null)
  if echo "$r" | grep -q '"text"'; then ok=$((ok+1)); else ok=0; fi
  [ "$ok" -ge 3 ] && { echo "=== serving verified"; break; }
  sleep 20
done
[ "$ok" -ge 3 ] || { echo "!!! not serving — leaving deployment for debug"; exit 1; }
kubectl delete job "$JOB" -n dynamo-cloud --ignore-not-found >/dev/null 2>&1; sleep 10
kubectl apply -f "$D/$BEN" >/dev/null
for i in $(seq 1 240); do
  st=$(kubectl get job "$JOB" -n dynamo-cloud -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null)
  [ -n "$st" ] && break; sleep 60
done
echo "=== bench terminal: ${st:-TIMEOUT}"
mkdir -p "$D/results"
kubectl logs -n dynamo-cloud "job/$JOB" > "$S/bench-$NAME.log" 2>/dev/null
python3 - "$NAME" "$S/bench-$NAME.log" "$D/results" <<'PYEOF'
import json, re, sys
name, logf, out = sys.argv[1], sys.argv[2], sys.argv[3]
log = open(logf, errors='replace').read()
found = 0; seen=set()
for pat in [r'--- /tmp/results/(results_concurrency_\d+_gpus_\d+_ctx_\d+_gen_\d+\w*)\.json\n(\{.*?\})\n',
            r'=== SLIM-RESULT /tmp/results/(results_concurrency_\d+_gpus_\d+_ctx_\d+_gen_\d+\w*)\.json (\{.*\})']:
    for m in re.finditer(pat, log, re.S if '---' in pat else 0):
        if m.group(1) in seen: continue
        seen.add(m.group(1)); found += 1
        d = json.loads(m.group(2))
        json.dump(d, open(f'{out}/{name}-{m.group(1)}.json', 'w'))
        g = re.search(r'gpus_\d+_ctx_(\d+)_gen_(\d+)', m.group(1)); pg, dg = int(g.group(1)), int(g.group(2))
        print(f"{name} conc {d['max_concurrency']}: out/decode-gpu {d['output_throughput']/dg:.1f} | in/prefill-gpu {d['total_input_tokens']/d['duration']/pg:.0f} | TPOT {d['mean_tpot_ms']:.2f}ms | TTFT med {d['median_ttft_ms']/1000:.1f}s | E2E {d['mean_e2el_ms']/1000:.1f}s")
print(f"[{name}] extracted {found}")
PYEOF
mkdir -p "$S/srv"; for p in $(kubectl get pods -n dynamo-cloud --no-headers | grep "$PFX" | awk '{print $1}'); do kubectl logs -n dynamo-cloud "$p" > "$S/srv/$p.log" 2>/dev/null; done
echo "=== KV transport evidence (rc_mlx5 = RDMA, cuda_ipc = MNNVL) ==="
grep -h -oiE "rc_mlx5\S*|cuda_ipc|dc_mlx5\S*" "$S/srv/$PFX-decode-0.log" 2>/dev/null | sort | uniq -c | head -6
export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token 2>/dev/null)
gcloud storage rsync -r "$S/srv" "$DEST/server-logs/$PFX" 2>&1 | tail -1
gcloud storage cp "$S/bench-$NAME.log" "$DEST/client-logs/$PFX-bench.log" 2>&1 | tail -1
gcloud storage rsync -r "$D/results" "$DEST/results" 2>&1 | tail -1
echo "=== tearing down $NAME"
kubectl delete -n dynamo-cloud "deployment/$PFX-frontend" "deployment/$PFX-prefill" "deployment/$PFX-decode" >/dev/null 2>&1
kubectl delete -n dynamo-cloud "statefulset/$PFX-decode" "computedomain/$PFX-cd" "service/$PFX-frontend" "service/$PFX-decode" "job/$JOB" >/dev/null 2>&1
rm -rf "$S"
echo "=== $NAME RESUME DONE"
