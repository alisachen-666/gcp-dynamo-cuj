"""Generate single-node aggregated arms (operator-less) for GB300, np-2 domain.

Topology: 6 aggregated workers x TP4 (single node each) on 24 GPUs — the
single-node shape from AIC's aggregated SILICON pareto (tp4, MoE-TP4/EP1) so no
operator/Grove is needed. Engine config adapted for mixed prefill+decode:
chunked prefill (long-context requests must not starve decode), ungated batch.
Arms: round-robin and KV-tuned (Strategy C step-1 flags). DYN_NAMESPACE-scoped;
runs concurrently with the disagg chain (np-1) on the idle np-2 domain.
"""
from pathlib import Path

OUT = Path.home() / "kv-cache-aware-bench/manifests/operatorless"
IMAGE = "nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.3.1"

AGG_ENGINE = """    backend: pytorch
    tensor_parallel_size: 4
    pipeline_parallel_size: 1
    enable_attention_dp: false
    moe_tensor_parallel_size: 4
    moe_expert_parallel_size: 1
    moe_config:
      backend: CUTLASS
      use_low_precision_moe_combine: true
    enable_chunked_prefill: true
    max_batch_size: 16
    max_num_tokens: 8192
    max_seq_len: 262144
    trust_remote_code: true
    kv_cache_config:
      free_gpu_memory_fraction: 0.85
      dtype: fp8
      tokens_per_block: 32
      enable_block_reuse: true
    cuda_graph_config:
      enable_padding: true
      batch_sizes: [1, 2, 4, 8, 16]
    print_iter_log: false
"""

ROUTER_FLAGS = {
    "rr": ["--router-mode", "round-robin"],
    "kvt": ["--router-mode", "kv",
            "--router-prefill-load-scale", "2.0",
            "--router-queue-policy", "fcfs",
            "--router-temperature", "0.0",
            "--router-kv-overlap-score-credit", "0.7"],
}

COMMON_POD = """      nodeSelector:
        kubernetes.io/arch: arm64
        cloud.google.com/gke-nodepool: np-2
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
        - key: kubernetes.io/arch
          operator: Equal
          value: arm64
          effect: NoSchedule
"""


def arm(name, flags):
    dgd = f"kimi-k25-agg1n-{name}"
    dyn_ns = f"kvbench-agg1n-{name}"
    frontend_args = "".join(f"\n            - {f}" for f in flags + ["--request-plane", "nats"])
    return f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: {dgd}-config
  namespace: dynamo-cloud
data:
  agg.yaml: |
{AGG_ENGINE}---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {dgd}-frontend
  namespace: dynamo-cloud
  labels: {{app: {dgd}-frontend, nvidia.com/dynamo-graph-deployment-name: {dgd}}}
spec:
  strategy: {{type: Recreate}}
  replicas: 1
  selector: {{matchLabels: {{app: {dgd}-frontend}}}}
  template:
    metadata:
      labels: {{app: {dgd}-frontend, nvidia.com/dynamo-graph-deployment-name: {dgd}}}
      annotations: {{gke-gcsfuse/volumes: "true", gke-gcsfuse/memory-limit: 4Gi}}
    spec:
{COMMON_POD}      containers:
        - name: frontend
          image: {IMAGE}
          command: [python3]
          args:
            - -m
            - dynamo.frontend{frontend_args}
          env:
            - {{name: ETCD_ENDPOINTS, value: "http://dynamo-platform-etcd.dynamo-cloud.svc.cluster.local:2379"}}
            - {{name: NATS_SERVER, value: "nats://dynamo-platform-nats.dynamo-cloud.svc.cluster.local:4222"}}
            - {{name: DYN_NAMESPACE, value: "{dyn_ns}"}}
          volumeMounts:
            - {{name: model-cache, mountPath: /model-cache, readOnly: true}}
      volumes:
        - {{name: model-cache, persistentVolumeClaim: {{claimName: model-cache}}}}
---
apiVersion: v1
kind: Service
metadata:
  name: {dgd}-frontend
  namespace: dynamo-cloud
spec:
  selector: {{app: {dgd}-frontend}}
  ports: [{{name: http, port: 8000, targetPort: 8000}}]
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {dgd}-worker
  namespace: dynamo-cloud
  labels: {{app: {dgd}-worker, nvidia.com/dynamo-graph-deployment-name: {dgd}}}
spec:
  strategy: {{type: Recreate}}
  replicas: 6
  selector: {{matchLabels: {{app: {dgd}-worker}}}}
  template:
    metadata:
      labels: {{app: {dgd}-worker, nvidia.com/dynamo-graph-deployment-name: {dgd}}}
      annotations:
        gke-gcsfuse/volumes: "true"
        gke-gcsfuse/memory-limit: 8Gi
        gke-gcsfuse/cpu-limit: "2"
        gke-gcsfuse/ephemeral-storage-limit: 1200Gi
    spec:
{COMMON_POD}      containers:
        - name: worker
          image: {IMAGE}
          workingDir: /workspace/
          securityContext:
            runAsUser: 0
            capabilities: {{add: [IPC_LOCK]}}
          command: [python3, -m, dynamo.trtllm]
          args:
            - --model-path
            - /model-cache/alisachen/Kimi-K2.5-NVFP4
            - --served-model-name
            - alisachen/Kimi-K2.5-NVFP4
            - --extra-engine-args
            - /config/agg.yaml
            - --publish-events-and-metrics
            - --request-plane
            - nats
            - --kv-block-size
            - "32"
            - --dyn-tool-call-parser
            - kimi_k2
            - --dyn-reasoning-parser
            - kimi_k25
          env:
            - {{name: ETCD_ENDPOINTS, value: "http://dynamo-platform-etcd.dynamo-cloud.svc.cluster.local:2379"}}
            - {{name: NATS_SERVER, value: "nats://dynamo-platform-nats.dynamo-cloud.svc.cluster.local:4222"}}
            - {{name: DYN_NAMESPACE, value: "{dyn_ns}"}}
            - {{name: DYN_SYSTEM_PORT, value: "9090"}}
            - {{name: HF_HOME, value: /model-cache}}
            - {{name: HF_HUB_OFFLINE, value: "1"}}
            - {{name: HF_MODULES_CACHE, value: /tmp/hf_modules}}
            - {{name: UCX_MEMTYPE_CACHE, value: "0"}}
            - {{name: UCX_IB_GID_INDEX, value: "5"}}
            - {{name: UCX_IB_ROCE_LOCAL_SUBNET, value: "1"}}
            - {{name: UCX_IB_ROCE_SUBNET_PREFIX_LEN, value: "64"}}
            - {{name: PYTORCH_CUDA_ALLOC_CONF, value: "expandable_segments:True"}}
            - {{name: TLLM_NUMA_AWARE_WORKER_AFFINITY, value: "1"}}
            - {{name: NCCL_MNNVL_ENABLE, value: "0"}}
            - {{name: NCCL_CUMEM_ENABLE, value: "1"}}
          startupProbe:
            httpGet: {{path: /live, port: 9090}}
            periodSeconds: 60
            failureThreshold: 240
            timeoutSeconds: 20
          resources:
            limits: {{nvidia.com/gpu: "4"}}
          envFrom: [{{secretRef: {{name: hf-token-secret}}}}]
          volumeMounts:
            - {{name: model-cache, mountPath: /model-cache}}
            - {{name: trtllm-config, mountPath: /config, readOnly: true}}
            - {{name: shm, mountPath: /dev/shm}}
      volumes:
        - {{name: model-cache, persistentVolumeClaim: {{claimName: model-cache}}}}
        - {{name: trtllm-config, configMap: {{name: {dgd}-config}}}}
        - {{name: shm, emptyDir: {{medium: Memory, sizeLimit: 250Gi}}}}
"""


for name, flags in ROUTER_FLAGS.items():
    p = OUT / f"agg1n-{name}.yaml"
    p.write_text(arm(name, flags))
    print(f"wrote {p.name}")
