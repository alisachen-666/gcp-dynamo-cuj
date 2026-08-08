#!/usr/bin/env bash
# Resume the m4r rev3 3-point pass after a local process death mid-bench.
# DOES NOT restart frontends or redeploy — the deployment and the conc-2048 bench job
# survived server-side. Waits for auth + the in-flight bench, harvests it, then runs the
# remaining concs (512, 4096) against the standing deployment, audits transport, tears down.
set -uo pipefail
export KUBECONFIG=$HOME/.kube/REDACTED-GKE-CLUSTER.config
D=$HOME/dsr1-pareto/dsr1-sweep
DEST="gs://REDACTED-GCS-BUCKET/mlperf_results/nvfp4_20260519_run5/logs/logs"
S=$(mktemp -d)
PFX="dsr1-m4r"

echo "=== waiting for API/auth"
for i in $(seq 1 200); do
  kubectl get ns dynamo-cloud >/dev/null 2>&1 && break
  sleep 120
done
kubectl get ns dynamo-cloud >/dev/null 2>&1 || { echo "!!! auth never returned"; exit 1; }
echo "=== auth OK at $(date -u +%H:%MZ)"

mkdir -p "$D/results"
extract() {  # $1 = bench log file
  python3 - "$1" "$D/results" <<'EOF'
import json, re, sys
logf, out = sys.argv[1], sys.argv[2]
log = open(logf, errors='replace').read()
found = 0
for m in re.finditer(r'--- /tmp/results/(results_concurrency_\d+_gpus_\d+_ctx_\d+_gen_\d+_rdmakv)\.json\n(\{.*?\})\n', log, re.S):
    d = json.loads(m.group(2)); found += 1
    json.dump(d, open(f'{out}/m4r-{m.group(1)}.json', 'w'))
    print(f"m4r conc {d['max_concurrency']}: out/decode-gpu {d['output_throughput']/48:.1f} | TPOT {d['mean_tpot_ms']:.2f}ms | TTFT med {d['median_ttft_ms']/1000:.1f}s")
if not found:
    for m in re.finditer(r'=== SLIM-RESULT /tmp/results/(results_concurrency_\d+_gpus_\d+_ctx_\d+_gen_\d+_rdmakv)\.json (\{.*\})', log):
        d = json.loads(m.group(2)); found += 1
        json.dump(d, open(f'{out}/m4r-{m.group(1)}.slim.json', 'w'))
        print(f"m4r conc {d['max_concurrency']} (SLIM): out/decode-gpu {d['output_throughput']/48:.1f} | TPOT {d['mean_tpot_ms']:.2f}ms")
print(f'extracted {found}')
EOF
}

# --- harvest the in-flight conc-2048 job (wait for terminal first) ---
if kubectl get job dsr1-m4r-bench-c2048 -n dynamo-cloud >/dev/null 2>&1; then
  echo "=== waiting for in-flight conc-2048 bench to finish"
  st=""
  for i in $(seq 1 200); do
    st=$(kubectl get job dsr1-m4r-bench-c2048 -n dynamo-cloud -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null)
    [ -n "$st" ] && break
    sleep 60
  done
  echo "=== conc 2048 terminal: ${st:-TIMEOUT}"
  kubectl logs -n dynamo-cloud job/dsr1-m4r-bench-c2048 > "$S/bench-m4r-c2048.log" 2>/dev/null
  extract "$S/bench-m4r-c2048.log"
else
  echo "=== no conc-2048 job found (already harvested or deleted) — continuing"
fi

# --- remaining concs against the standing deployment (no frontend restart) ---
for CONC in 512 4096; do
  if ls "$D"/results/m4r-results_concurrency_${CONC}_*.json >/dev/null 2>&1; then
    echo "=== conc $CONC already extracted — skipping"; continue
  fi
  JOB="dsr1-m4r-bench-c$CONC"
  echo "=== bench conc $CONC"
  kubectl delete job "$JOB" -n dynamo-cloud --ignore-not-found >/dev/null 2>&1
  for i in $(seq 1 30); do kubectl get job "$JOB" -n dynamo-cloud >/dev/null 2>&1 || break; sleep 5; done
  kubectl apply -f "$D/manifests/m4r-bench-conc$CONC.yaml" >/dev/null
  st=""
  for i in $(seq 1 240); do
    st=$(kubectl get job "$JOB" -n dynamo-cloud -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null)
    [ -n "$st" ] && break
    sleep 60
  done
  echo "=== conc $CONC terminal: ${st:-TIMEOUT}"
  kubectl logs -n dynamo-cloud "job/$JOB" > "$S/bench-m4r-c$CONC.log" 2>/dev/null
  extract "$S/bench-m4r-c$CONC.log"
done

echo "=== transport-evidence audit"
mkdir -p "$S/srv"
for p in $(kubectl get pods -n dynamo-cloud --no-headers | grep "$PFX" | awk '{print $1}'); do
  kubectl logs -n dynamo-cloud "$p" > "$S/srv/$p.log" 2>/dev/null
done
tot_rc=0; tot_ipc=0; tot_err=0
for f in "$S"/srv/*prefill*.log "$S"/srv/*decode*.log; do
  [ -f "$f" ] || continue
  tot_rc=$((tot_rc+$(grep -c 'rc_mlx5' "$f"))); tot_ipc=$((tot_ipc+$(grep -c 'cuda_ipc' "$f"))); tot_err=$((tot_err+$(grep -cE 'NIXL_ERR|REMOTE_DISCONNECT' "$f")))
done
echo "TRANSPORT TOTALS: rc_mlx5=$tot_rc cuda_ipc=$tot_ipc nixl_errors=$tot_err"
[ "$tot_ipc" = "0" ] && [ "$tot_err" = "0" ] && [ "$tot_rc" != "0" ] && echo "TRANSPORT VERDICT: RDMA-KV CLEAN" || echo "TRANSPORT VERDICT: CHECK LOGS"

export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token 2>/dev/null)
gcloud storage rsync -r "$S/srv" "$DEST/server-logs/dsr1-m4r-rev3-3pts" 2>&1 | tail -1
for fl in "$S"/bench-m4r-c*.log; do [ -f "$fl" ] && gcloud storage cp "$fl" "$DEST/client-logs/$(basename "$fl" .log)-rev3.log" 2>&1 | tail -1; done
gcloud storage rsync -r "$D/results" "$DEST/results" 2>&1 | tail -1
echo "=== tearing down m4r"
kubectl delete -n dynamo-cloud "deployment/$PFX-frontend" "deployment/$PFX-prefill" >/dev/null 2>&1
kubectl delete -n dynamo-cloud "statefulset/$PFX-decode" "deployment/$PFX-decode" >/dev/null 2>&1
kubectl delete -n dynamo-cloud "computedomain/$PFX-cd" "service/$PFX-frontend" "service/$PFX-decode" >/dev/null 2>&1
kubectl delete -n dynamo-cloud job/dsr1-m4r-bench-c512 job/dsr1-m4r-bench-c2048 job/dsr1-m4r-bench-c4096 >/dev/null 2>&1
rm -rf "$S"
echo "=== M4R REV3 3PTS DONE (resumed)"
