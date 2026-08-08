#!/usr/bin/env bash
# Smoke-test ladder for DSR1 Pareto work on pr9-a4xmax (np-2 free node).
# Usage: ./run-smoke.sh [0|1|2]   (default: 0)
set -euo pipefail
cd "$(dirname "$0")"

STAGE="${1:-0}"

echo "== checking free GPUs on np-2 =="
kubectl get pods -A -o json | jq -r '.items[] | select(.status.phase=="Running") |
  select([.spec.containers[].resources.requests["nvidia.com/gpu"] // empty] | length > 0) |
  "\(.metadata.name)\t\(.spec.nodeName)"'

case "$STAGE" in
  0)
    kubectl apply -f manifests/smoke-0-sanity.yaml
    kubectl wait --for=condition=complete --timeout=15m job/dsr1-smoke-sanity || true
    kubectl logs job/dsr1-smoke-sanity
    ;;
  1)
    # flat-file configmap with sa-bench scripts (tokenizer submodule not needed for smoke)
    kubectl create configmap sa-bench-scripts \
      --from-file=../common/sa-bench/backend_request_func.py \
      --from-file=../common/sa-bench/benchmark_dataset.py \
      --from-file=../common/sa-bench/benchmark_serving.py \
      --from-file=../common/sa-bench/benchmark_utils.py \
      --dry-run=client -o yaml | kubectl apply -f -
    kubectl apply -f manifests/smoke-1-tiny-bench.yaml
    kubectl wait --for=condition=complete --timeout=30m job/dsr1-smoke-tiny-bench || true
    kubectl logs job/dsr1-smoke-tiny-bench | tail -60
    ;;
  2)
    kubectl create configmap sa-bench-scripts \
      --from-file=../common/sa-bench/backend_request_func.py \
      --from-file=../common/sa-bench/benchmark_dataset.py \
      --from-file=../common/sa-bench/benchmark_serving.py \
      --from-file=../common/sa-bench/benchmark_utils.py \
      --dry-run=client -o yaml | kubectl apply -f -
    kubectl apply -f manifests/smoke-2-dsr1-fp4-tp4.yaml
    echo "FP4 smoke launched. Weights download takes 30-60 min; follow with:"
    echo "  kubectl logs -f job/dsr1-smoke-fp4-tp4"
    ;;
  *)
    echo "unknown stage: $STAGE (use 0, 1, or 2)"; exit 1 ;;
esac
