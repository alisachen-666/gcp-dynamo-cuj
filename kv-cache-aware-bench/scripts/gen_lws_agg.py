"""LWS-based multinode aggregated arms: AIC's TP8 GB300 config, operator-less.

Runs AIC's tuned aggregated configuration (3 workers x TP8 = 2 nodes each) via
plain StatefulSets (LWS CRD is not installed on this cluster and its controller
needs the same RBAC the operator does; a 2-replica StatefulSet + headless svc
provides the stable peer DNS and leader convention LWS would). Launch
pattern ported from AI-Hypercomputer/gpu-recipes a4xmax multi-node template
(sshd + mpirun across the pod group, LWS_LEADER_ADDRESS rendezvous), adapted to
launch dynamo.trtllm so the worker registers with the Dynamo router.

Deviations from raw AIC output (same policy as disagg):
  - moe_config CUTLASS + low-precision combine (WIDEEP broken on runtime 1.3.1)
  - enable_chunked_prefill: true (trace p99 input 252k > max_num_tokens)
  - max_seq_len 262144, trust_remote_code
Prereq: ssh keypair secret `alisachen-lws-ssh` (see bottom); np-3 ComputeDomain.
"""
from pathlib import Path

OUT = Path.home() / "kv-cache-aware-bench/manifests/operatorless"
IMAGE = "nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.3.1"

ENGINE = """    backend: pytorch
    tensor_parallel_size: 8
    pipeline_parallel_size: 1
    enable_attention_dp: false
    moe_expert_parallel_size: 4
    moe_tensor_parallel_size: 2
    moe_config:
      backend: CUTLASS
      use_low_precision_moe_combine: true
    enable_chunked_prefill: true
    max_batch_size: 20
    max_num_tokens: 137536
    max_seq_len: 262144
    trust_remote_code: true
    kv_cache_config:
      free_gpu_memory_fraction: 0.8
      dtype: fp8
      tokens_per_block: 32
      enable_block_reuse: true
    cuda_graph_config:
      enable_padding: true
      batch_sizes: [1, 2, 4, 8, 16, 20, 32, 64, 128, 256, 512]
    disable_overlap_scheduler: false
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

LAUNCHER = """                  LEADER="__STS__-0.__STS__.dynamo-cloud.svc.cluster.local"
                  PEER="__STS__-1.__STS__.dynamo-cloud.svc.cluster.local"
                  if [[ $HOSTNAME != *-0 ]]; then sleep infinity; fi
                  until ssh -q -p 2222 $PEER hostname; do echo waiting-for-peer; sleep 5; done
                  export MASTER_HOST=$LEADER
                  sed -i "s/\\"tcp:\\/\\/127.0.0.1:\\*\\"/f\\"tcp:\\/\\/{os.environ.get('MASTER_HOST', '127.0.0.1')}:\\*\\"/g" /usr/local/lib/python3.12/dist-packages/tensorrt_llm/executor/ipc.py || true
                  export OMPI_MCA_orte_keep_fqdn_hostnames=t OMPI_MCA_oob_tcp_if_include=eth0 OMPI_MCA_btl_tcp_if_include=eth0
                  mpirun --allow-run-as-root -np 8 -H "$LEADER:4,$PEER:4" \\
                    --map-by ppr:4:node --bind-to numa \\
                    -mca plm_rsh_args "-p 2222 -o StrictHostKeyChecking=no" \\
                    -x PATH -x LD_LIBRARY_PATH -x NCCL_MNNVL_ENABLE -x NCCL_CUMEM_ENABLE -x MC_FORCE_MNNVL \\
                    -x UCX_MEMTYPE_CACHE -x UCX_IB_GID_INDEX -x UCX_IB_ROCE_LOCAL_SUBNET -x UCX_IB_ROCE_SUBNET_PREFIX_LEN \\
                    -x PYTORCH_CUDA_ALLOC_CONF -x TLLM_NUMA_AWARE_WORKER_AFFINITY -x TRTLLM_ENABLE_PDL \\
                    -x ETCD_ENDPOINTS -x NATS_SERVER -x DYN_NAMESPACE -x DYN_SYSTEM_PORT \\
                    -x HF_HOME -x HF_HUB_OFFLINE -x HF_MODULES_CACHE -x MASTER_HOST \\
                    trtllm-llmapi-launch python3 -m dynamo.trtllm \\
                      --model-path /model-cache/alisachen/Kimi-K2.5-NVFP4 \\
                      --served-model-name alisachen/Kimi-K2.5-NVFP4 \\
                      --extra-engine-args /config/agg.yaml \\
                      --publish-events-and-metrics --request-plane nats \\
                      --kv-block-size 32 --dyn-tool-call-parser kimi_k2 --dyn-reasoning-parser kimi_k25
"""

POD_COMMON = """          nodeSelector:
            kubernetes.io/arch: arm64
            cloud.google.com/gke-nodepool: np-3
          tolerations:
            - {key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}
            - {key: kubernetes.io/arch, operator: Equal, value: arm64, effect: NoSchedule}
          resourceClaims:
            - {name: cd-channel, resourceClaimTemplateName: kv-bench-cd-np3-channel}
            - {name: rdma, resourceClaimTemplateName: mrdma-all}
          volumes:
            - {name: model-cache, persistentVolumeClaim: {claimName: model-cache}}
            - {name: cfg, configMap: {name: __DGD__-config}}
            - {name: shm, emptyDir: {medium: Memory, sizeLimit: 250Gi}}
            - {name: sshkeys, secret: {secretName: alisachen-lws-ssh, defaultMode: 0400}}
"""

SSH_SETUP = """                  mkdir -p /root/.ssh && cp /sshkeys/id_rsa /root/.ssh/ && cp /sshkeys/id_rsa.pub /root/.ssh/authorized_keys
                  chmod 700 /root/.ssh && chmod 600 /root/.ssh/id_rsa /root/.ssh/authorized_keys
                  printf 'Host *\\n  Port 2222\\n  StrictHostKeyChecking no\\n  UserKnownHostsFile /dev/null\\n  ServerAliveInterval 30\\n' > /root/.ssh/config
                  mkdir -p /run/sshd && /usr/sbin/sshd -p 2222
"""

CONTAINER_COMMON = """              image: __IMAGE__
              workingDir: /workspace/
              securityContext: {runAsUser: 0, capabilities: {add: [IPC_LOCK]}}
              env:
                - {name: ETCD_ENDPOINTS, value: "http://dynamo-platform-etcd.dynamo-cloud.svc.cluster.local:2379"}
                - {name: NATS_SERVER, value: "nats://dynamo-platform-nats.dynamo-cloud.svc.cluster.local:4222"}
                - {name: DYN_NAMESPACE, value: "__DYNNS__"}
                - {name: DYN_SYSTEM_PORT, value: "9090"}
                - {name: HF_HOME, value: /model-cache}
                - {name: HF_HUB_OFFLINE, value: "1"}
                - {name: HF_MODULES_CACHE, value: /tmp/hf_modules}
                - {name: NCCL_MNNVL_ENABLE, value: "1"}
                - {name: NCCL_CUMEM_ENABLE, value: "1"}
                - {name: MC_FORCE_MNNVL, value: "1"}
                - {name: UCX_MEMTYPE_CACHE, value: "0"}
                - {name: UCX_IB_GID_INDEX, value: "5"}
                - {name: UCX_IB_ROCE_LOCAL_SUBNET, value: "1"}
                - {name: UCX_IB_ROCE_SUBNET_PREFIX_LEN, value: "64"}
                - {name: PYTORCH_CUDA_ALLOC_CONF, value: "expandable_segments:True"}
                - {name: TLLM_NUMA_AWARE_WORKER_AFFINITY, value: "1"}
                - {name: TRTLLM_ENABLE_PDL, value: "1"}
              envFrom: [{secretRef: {name: hf-token-secret}}]
              volumeMounts:
                - {name: model-cache, mountPath: /model-cache}
                - {name: cfg, mountPath: /config, readOnly: true}
                - {name: shm, mountPath: /dev/shm}
                - {name: sshkeys, mountPath: /sshkeys, readOnly: true}
              resources:
                limits: {nvidia.com/gpu: "4"}
                claims: [{name: cd-channel}, {name: rdma}]
"""


def sts_worker(dgd, w, common, pod):
    sts = f"{dgd}-w{w}"
    launcher = LAUNCHER.replace("__STS__", sts)
    return f"""apiVersion: v1
kind: Service
metadata: {{name: {sts}, namespace: dynamo-cloud}}
spec:
  clusterIP: None
  selector: {{app: {sts}}}
  ports: [{{name: ssh, port: 2222}}]
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: {sts}
  namespace: dynamo-cloud
  labels: {{app: {sts}, nvidia.com/dynamo-graph-deployment-name: {dgd}}}
spec:
  serviceName: {sts}
  replicas: 2
  podManagementPolicy: Parallel
  selector: {{matchLabels: {{app: {sts}}}}}
  template:
    metadata:
      labels: {{app: {sts}, nvidia.com/dynamo-graph-deployment-name: {dgd}}}
      annotations: {{gke-gcsfuse/volumes: "true", gke-gcsfuse/memory-limit: 8Gi, gke-gcsfuse/cpu-limit: "2", gke-gcsfuse/ephemeral-storage-limit: 1200Gi}}
    spec:
{pod}          containers:
            - name: worker
{common}              command: [bash, -c]
              args:
                - |
                  set -x
{SSH_SETUP}{launcher}
              startupProbe:
                exec: {{command: [bash, -c, "if [[ $HOSTNAME == *-0 ]]; then curl -sf http://localhost:9090/live; else pgrep sshd; fi"]}}
                periodSeconds: 60
                failureThreshold: 240
                timeoutSeconds: 20
---
"""


def arm(name, flags):
    dgd = f"kimi-k25-aggtp8-{name}"
    dyn_ns = f"kvbench-aggtp8-{name}"
    fe_args = "".join(f"\n            - {f}" for f in flags + ["--request-plane", "nats"])
    common = CONTAINER_COMMON.replace("__IMAGE__", IMAGE).replace("__DYNNS__", dyn_ns)
    pod = POD_COMMON.replace("__DGD__", dgd)
    sts_docs = "".join(sts_worker(dgd, w, common, pod) for w in range(3))
    body = f"""apiVersion: v1
kind: ConfigMap
metadata: {{name: {dgd}-config, namespace: dynamo-cloud}}
data:
  agg.yaml: |
{ENGINE}---
{sts_docs}apiVersion: apps/v1
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
      nodeSelector: {{kubernetes.io/arch: arm64, cloud.google.com/gke-nodepool: np-3}}
      tolerations:
        - {{key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}}
        - {{key: kubernetes.io/arch, operator: Equal, value: arm64, effect: NoSchedule}}
      containers:
        - name: frontend
          image: {IMAGE}
          command: [python3]
          args:
            - -m
            - dynamo.frontend{fe_args}
          env:
            - {{name: ETCD_ENDPOINTS, value: "http://dynamo-platform-etcd.dynamo-cloud.svc.cluster.local:2379"}}
            - {{name: NATS_SERVER, value: "nats://dynamo-platform-nats.dynamo-cloud.svc.cluster.local:4222"}}
            - {{name: DYN_NAMESPACE, value: "{dyn_ns}"}}
          volumeMounts: [{{name: model-cache, mountPath: /model-cache, readOnly: true}}]
      volumes: [{{name: model-cache, persistentVolumeClaim: {{claimName: model-cache}}}}]
---
apiVersion: v1
kind: Service
metadata: {{name: {dgd}-frontend, namespace: dynamo-cloud}}
spec:
  selector: {{app: {dgd}-frontend}}
  ports: [{{name: http, port: 8000, targetPort: 8000}}]
"""
    return body.replace("{sts_docs}", sts_docs)


CD_NP3 = """apiVersion: resource.nvidia.com/v1beta1
kind: ComputeDomain
metadata: {name: kv-bench-cd-np3, namespace: dynamo-cloud}
spec:
  numNodes: 0
  channel:
    resourceClaimTemplate: {name: kv-bench-cd-np3-channel}
"""

(OUT / "cd-np3.yaml").write_text(CD_NP3)
for name, flags in ROUTER_FLAGS.items():
    (OUT / f"aggtp8-{name}.yaml").write_text(arm(name, flags))
    print(f"wrote aggtp8-{name}.yaml")
print("wrote cd-np3.yaml")
