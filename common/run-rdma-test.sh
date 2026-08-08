#!/usr/bin/env bash
# RDMA data-path test: 2-node NCCL allreduce FORCED over the mlx5 NICs (NVLink disabled).
# Bypasses DRA (no rdma claims exist on this cluster) via hostNetwork + privileged.
# PASS = "NET/IB" transport in NCCL debug + IB-class bandwidth. FAIL/fallback = socket lines.
set -euo pipefail

POOL="${POOL:-np-2}"
mapfile -t NODES < <(kubectl get nodes -l cloud.google.com/gke-nodepool=$POOL -o json | python3 -c '
import sys, json, re
d = json.load(sys.stdin)
for n in d["items"][:2]:
    ip = next(a["address"] for a in n["status"]["addresses"]
              if a["type"] == "InternalIP" and re.match(r"^\d+\.", a["address"]))
    print(n["metadata"]["name"], ip)')
[ "${#NODES[@]}" -eq 2 ] || { echo "need 2 $POOL nodes"; exit 1; }
NODE0=$(echo "${NODES[0]}" | cut -d' ' -f1); NODE0_IP=$(echo "${NODES[0]}" | cut -d' ' -f2)
NODE1=$(echo "${NODES[1]}" | cut -d' ' -f1)
echo "master: $NODE0 ($NODE0_IP), worker: $NODE1"

gen() { # $1=suffix $2=nodeName $3=node_rank
cat <<YAML
apiVersion: batch/v1
kind: Job
metadata:
  name: rdma-nccl-$1
  namespace: default
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels: {app: rdma-nccl}
    spec:
      restartPolicy: Never
      nodeName: $2
      hostNetwork: true
      tolerations:
      - {key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}
      - {key: kubernetes.io/arch, operator: Equal, value: arm64, effect: NoSchedule}
      containers:
      - name: nccl
        image: lmsysorg/sglang:v0.5.8.post1-cu130-runtime
        securityContext:
          privileged: true
        command: ["bash", "-c"]
        args:
        - |
          set -e
          echo "== rdma NIC addressing =="; ip -br addr show | grep -E "rdma|eth0" || true
          cat > /tmp/allreduce_bw.py <<'EOF'
          import time, torch, torch.distributed as dist
          dist.init_process_group("nccl")
          r, ws = dist.get_rank(), dist.get_world_size()
          torch.cuda.set_device(r % 4)
          assert ws == 8, f"expected 8, got {ws}"
          n = 256 * 1024 * 1024
          x = torch.ones(n, device="cuda")
          for _ in range(5): dist.all_reduce(x)
          torch.cuda.synchronize()
          iters = 20
          t0 = time.perf_counter()
          for _ in range(iters): dist.all_reduce(x)
          torch.cuda.synchronize()
          dt = (time.perf_counter() - t0) / iters
          busbw = 2 * (ws - 1) / ws * n * 4 / dt / 1e9
          if r == 0: print(f"RDMA-path allreduce 1GiB: {dt*1e3:.2f} ms/iter, busbw {busbw:.1f} GB/s", flush=True)
          dist.barrier(); print(f"rank {r}: OK", flush=True)
          EOF
          export NCCL_MNNVL_ENABLE=0
          export NCCL_NET=IB
          export NCCL_IB_HCA=mlx5
          export NCCL_SOCKET_IFNAME=eth0
          export GLOO_SOCKET_IFNAME=eth0
          export NCCL_DEBUG=INFO
          export NCCL_DEBUG_SUBSYS=INIT,NET
          RC=0
          python3 -m torch.distributed.run \
            --nnodes=2 --nproc_per_node=4 --node_rank=$3 \
            --master_addr=$NODE0_IP --master_port=29617 \
            /tmp/allreduce_bw.py > /tmp/nccl.log 2>&1 || RC=\$?
          echo "torchrun exit code: \$RC"
          grep -E "NET/IB : Using|busbw|rank .: OK|NCCL WARN|Error|error" /tmp/nccl.log | head -25
          [ "\$RC" != "0" ] && { echo "--- log tail ---"; tail -30 /tmp/nccl.log; }
          grep -qE "rank .: OK" /tmp/nccl.log || { echo "RDMA TEST NODE $3 FAILED (no rank OK)"; tail -40 /tmp/nccl.log; exit 1; }
          if [ "$3" = "0" ]; then grep -q "busbw" /tmp/nccl.log || { echo "RDMA TEST FAILED (no busbw)"; exit 1; }; fi
          echo "RDMA TEST NODE $3 PASSED"
        resources:
          requests: {nvidia.com/gpu: "4", cpu: "16", memory: 64Gi}
          limits: {nvidia.com/gpu: "4"}
        volumeMounts:
        - {name: dshm, mountPath: /dev/shm}
      volumes:
      - name: dshm
        emptyDir: {medium: Memory, sizeLimit: 16Gi}
YAML
}

gen a "$NODE0" 0 | kubectl apply -f -
gen b "$NODE1" 1 | kubectl apply -f -
echo "launched. follow: kubectl logs -f job/rdma-nccl-a"
