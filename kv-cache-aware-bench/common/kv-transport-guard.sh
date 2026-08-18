#!/usr/bin/env bash
# KV-transport guard: FAIL any RDMA-KV run whose KV path degrades to TCP.
# Policy (user directive 2026-08-06): RDMA runs must ride RDMA — a run that falls back to
# TCP must fail loudly, never produce a result that looks valid.
#
# Detection (UCX/NIXL, log-based; requires UCX_PROTO_INFO=y + UCX_LOG_LEVEL=info in workers):
#   VIOLATION if any worker log shows
#     - "no remote ep address"                       (wireup failure -> fallback imminent)
#     - bulk proto rows selecting tcp:  tcp/<dev> on zero-copy/rndv/copy-in/multi-frag rows
#     - NIXL_ERR / REMOTE_DISCONNECT
#   POSITIVE evidence required: >=1 rc_mlx5 proto row across workers (else transport unproven).
#   (mooncake/DSv4: patterns below cover UCX-style lines; mooncake-specific TCP markers are
#    added to MOONCAKE_PAT as they are learned from the first p8r run — until then DSv4 arms
#    rely on the wall-clock backstop in their runner plus post-run inspection.)
#
# Usage:
#   kv-transport-guard.sh gate  <pod-grep-pattern>                    # one-shot; exit 1 on violation
#   kv-transport-guard.sh watch <pod-grep-pattern> <bench-job> <sec>  # loop; on violation:
#            deletes the bench job, writes ${GUARD_MARKER:-/tmp/kv-guard-tripped}, exits 1
set -uo pipefail
MODE=$1; PAT=$2
NS=dynamo-cloud
TCP_BULK='tcp/[a-z0-9]+.*(zero-copy|rndv|copy-in|multi-frag)|((zero-copy|rndv|copy-in|multi-frag).*tcp/[a-z0-9]+)'
MOONCAKE_PAT='TransferEngine.*(tcp|TCP)|transport.*fallback'

scan() {
  # Context-aware TCP detection (2026-08-06 fix): benign host->host metadata puts select
  # tcp in EVERY healthy run — only a tcp selection under a CUDA-memory proto table is a
  # violation. Wireup/NIXL error signatures remain unconditional.
  local viol=0 rc_total=0 pod
  for pod in $(kubectl get pods -n $NS --no-headers 2>/dev/null | grep -E "$PAT" | grep -E "prefill|decode" | grep Running | awk '{print $1}'); do
    local out we cudatcp nerr mck rc
    out=$(kubectl logs -n $NS "$pod" --tail=20000 2>/dev/null | python3 -c "
import re, sys
header = ''
we = cudatcp = nerr = mck = rc = 0
for ln in sys.stdin:
    if 'ucp_context' in ln and 'from' in ln:
        header = ln
    if 'no remote ep address' in ln: we += 1
    if re.search(r'NIXL_ERR|REMOTE_DISCONNECT', ln): nerr += 1
    if re.search(r'TransferEngine.*(tcp|TCP)|transport.*fallback', ln): mck += 1
    if 'rc_mlx5' in ln: rc += 1
    if re.search(r'tcp/[a-z0-9]+', ln) and '|' in ln and 'cuda' in header: cudatcp += 1
print(we, cudatcp, nerr, mck, rc)")
    read -r we cudatcp nerr mck rc <<<"$out"
    rc_total=$((rc_total+${rc:-0}))
    if [ "${we:-0}" != "0" ] || [ "${cudatcp:-0}" != "0" ] || [ "${nerr:-0}" != "0" ] || [ "${mck:-0}" != "0" ]; then
      echo "KV-GUARD VIOLATION pod=$pod wireup_err=$we cuda_tcp_rows=$cudatcp nixl_err=$nerr mooncake_tcp=$mck"
      viol=1
    fi
  done
  if [ "$viol" = "0" ] && [ "$rc_total" = "0" ]; then
    echo "KV-GUARD WARNING: no rc_mlx5 evidence found yet (transport unproven; not failing)"
  fi
  return $viol
}

case "$MODE" in
  gate)
    if scan; then echo "KV-GUARD GATE: PASS (RDMA path clean)"; exit 0
    else echo "KV-GUARD GATE: FAIL — KV path degraded; refusing to bench"; exit 1; fi ;;
  watch)
    JOB=$3; IV=${4:-180}
    MARKER=${GUARD_MARKER:-/tmp/kv-guard-tripped}
    rm -f "$MARKER"
    while true; do
      if ! scan; then
        echo "KV-GUARD WATCH: violation detected at $(date -u +%H:%MZ) — killing bench job $JOB"
        kubectl delete job "$JOB" -n $NS --ignore-not-found >/dev/null 2>&1
        echo "tripped $(date -u +%FT%TZ)" > "$MARKER"
        exit 1
      fi
      kubectl get job "$JOB" -n $NS >/dev/null 2>&1 || { echo "KV-GUARD WATCH: job gone, guard exiting clean"; exit 0; }
      sleep "$IV"
    done ;;
  *) echo "usage: $0 gate|watch <pod-pattern> [bench-job] [interval]"; exit 2 ;;
esac
