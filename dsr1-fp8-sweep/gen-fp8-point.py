#!/usr/bin/env python3
"""Generate DSR1-FP8 mid/max GKE manifests directly from recipe values.

Deliberately NOT derived from a sibling manifest by string-replace: that is exactly what
produced the silent config drift in the DSv4 sweep (see ../dsv4-sweep/PLAN.md error log).
Every flag that differs between configs is an explicit entry in CONFIGS below, and every
flag that is shared is written once here. Source of truth:
  reference/srt-slurm-gb300-fp8-8k1k-stp-{mid,max}.yaml   (branch sa-submission-q2-2026 @ deb1dfd9)

Verified: mid and max differ ONLY in prefill_workers, decode nodes, decode tp/dp/ep, and
concurrencies (see `diff` of the two recipe files). Prefill flags are byte-identical.

Usage:  ./gen-fp8-point.py            # writes mid + max server manifests and their bench jobs
"""
import os

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'manifests')
IMAGE = 'lmsysorg/sglang:v0.5.8.post1-cu130'          # InferenceX pin; arm64 verified
DYNAMO = '0.8.0'                                       # srtctl DynamoConfig default (NOT 0.8.1)
MODEL_HOSTPATH = '/mnt/stateful_partition/kube-ephemeral-ssd/dsr1-fp8-model'
DG_HOSTPATH = '/mnt/stateful_partition/kube-ephemeral-ssd/dsr1-fp8-dg-cache'
NS = 'dynamo-cloud'

# decode cuda-graph-bs list, verbatim from the recipe (renders space-separated per srtctl
# _config_to_cli_args: lists become `--flag v1 v2 ...`)
CGBS = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144,
        152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 264, 272, 280,
        288, 296, 304, 312, 320, 328, 336, 344, 352, 360, 368, 376, 384, 416, 448, 480, 512,
        544, 576, 608, 640, 672, 704, 736, 768]

CONFIGS = {
    # id      pworkers  dtp/dp/ep  dnodes  concurrencies
    'mid': dict(pworkers=5, dtp=32, dnodes=8, conc=[128, 256, 512, 1024]),
    'max': dict(pworkers=6, dtp=24, dnodes=6, conc=[2048, 4096]),
}

PREFILL_NODES_PER_WORKER = 2   # DEP8 = tp8 = 8 GPUs = 2 nodes (4 GPUs/node)

# ---- environment blocks (verbatim from recipe, + GKE canonical transport block) ----
COMMON_ENV = [
    ('TORCH_DISTRIBUTED_DEFAULT_TIMEOUT', '1800'),
    ('SGLANG_DG_CACHE_DIR', '/dg-cache'),          # D1: recipe path /configs/dg-10212025 unpublished
    ('DYN_SKIP_SGLANG_LOG_FORMATTING', '1'),
    ('MC_TE_METRIC', 'true'),
    ('SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE', '100000'),
    ('SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT', '100000'),
    ('SGLANG_DISAGGREGATION_WAITING_TIMEOUT', '100000'),
    ('SGLANG_MOONCAKE_CUSTOM_MEM_POOL', 'True'),
    ('MC_FORCE_MNNVL', '1'),
    ('NCCL_MNNVL_ENABLE', '1'),
    ('NCCL_CUMEM_ENABLE', '1'),
    ('SGLANG_USE_MESSAGE_QUEUE_BROADCASTER', '0'),
    ('SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK', '1'),
    ('PYTHONUNBUFFERED', '1'),
]
DECODE_EXTRA_ENV = [
    ('SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK', '768'),
    ('SGLANG_DECODE_BOOTSTRAP_TIMEOUT', '1000'),
    ('SGLANG_HACK_SEQ_BOOTSTRAP_ROOM', '1'),
]
UCX_ENV = [
    ('UCX_TLS', 'cuda_copy,cuda_ipc,tcp'),
    ('UCX_CUDA_IPC_ENABLE_MNNVL', 'y'),
    ('UCX_MEMTYPE_CACHE', 'n'),
    ('UCX_MEMTYPE_REG_WHOLE', 'n'),
    ('UCX_PROTO_INFO', 'y'),
    ('GLOO_SOCKET_IFNAME', 'eth0'),
    ('TP_SOCKET_IFNAME', 'eth0'),
]
INFRA_ENV = [
    ('ETCD_ENDPOINTS', f'http://dynamo-platform-etcd.{NS}.svc.cluster.local:2379'),
    ('NATS_SERVER', f'nats://dynamo-platform-nats.{NS}.svc.cluster.local:4222'),
]


def env_block(pairs, indent=8):
    pad = ' ' * indent
    out = []
    for k, v in pairs:
        out.append(f'{pad}- name: {k}\n{pad}  value: "{v}"')
    return '\n'.join(out)


def pod_index_env(indent=8):
    pad = ' ' * indent
    return (f'{pad}- name: POD_INDEX\n{pad}  valueFrom:\n{pad}    fieldRef:\n'
            f"{pad}      fieldPath: metadata.labels['apps.kubernetes.io/pod-index']")


# ---- server flags, verbatim from the recipe's sglang_config ----
# NOTE: mid/max set NO `quantization` and NO `fp8-gemm-backend` (unlike low-latency) — the
# native FP8 checkpoint is auto-detected from config.json. Verbatim means: do not add them.
# `false` booleans are omitted entirely, matching srtctl _config_to_cli_args.
def prefill_args():
    return """--model-path /model \\
            --served-model-name deepseek-ai/DeepSeek-R1 \\
            --disaggregation-mode prefill \\
            --skip-tokenizer-init \\
            --trust-remote-code \\
            --tensor-parallel-size 8 \\
            --data-parallel-size 8 \\
            --expert-parallel-size 8 \\
            --enable-dp-attention \\
            --attention-backend trtllm_mla \\
            --kv-cache-dtype fp8_e4m3 \\
            --disable-radix-cache \\
            --stream-interval 50 \\
            --max-running-requests 30000 \\
            --context-length 9300 \\
            --watchdog-timeout 1000000 \\
            --disable-shared-experts-fusion \\
            --eplb-algorithm deepseek \\
            --disaggregation-bootstrap-port 30001 \\
            --disaggregation-transfer-backend nixl \\
            --mem-fraction-static 0.75 \\
            --max-total-tokens 524288 \\
            --chunked-prefill-size 131072 \\
            --load-balance-method round_robin \\
            --disable-cuda-graph \\
            --moe-a2a-backend deepep \\
            --deepep-mode normal \\
            --ep-dispatch-algorithm dynamic \\
            --moe-dense-tp-size 1 \\
            --enable-dp-lm-head \\
            --ep-num-redundant-experts 32 \\
            --deepep-config /configs/deepep_config.json"""


def decode_args(dtp):
    return f"""--model-path /model \\
            --served-model-name deepseek-ai/DeepSeek-R1 \\
            --disaggregation-mode decode \\
            --skip-tokenizer-init \\
            --trust-remote-code \\
            --disaggregation-transfer-backend nixl \\
            --tensor-parallel-size {dtp} \\
            --data-parallel-size {dtp} \\
            --expert-parallel-size {dtp} \\
            --enable-dp-attention \\
            --attention-backend trtllm_mla \\
            --kv-cache-dtype fp8_e4m3 \\
            --disable-radix-cache \\
            --stream-interval 50 \\
            --decode-log-interval 1000 \\
            --max-running-requests 45000 \\
            --context-length 9300 \\
            --watchdog-timeout 1000000 \\
            --disable-shared-experts-fusion \\
            --eplb-algorithm deepseek \\
            --disaggregation-bootstrap-port 30001 \\
            --mem-fraction-static 0.82 \\
            --chunked-prefill-size 36864 \\
            --moe-a2a-backend deepep \\
            --deepep-mode low_latency \\
            --ep-dispatch-algorithm static \\
            --moe-dense-tp-size 1 \\
            --enable-dp-lm-head \\
            --prefill-round-robin-balance \\
            --ep-num-redundant-experts 32 \\
            --deepep-config /configs/deepep_config.json \\
            --cuda-graph-bs {' '.join(map(str, CGBS))} \\
            --cuda-graph-max-bs 768"""


VOLUMES = f"""      volumes:
      - name: dshm
        emptyDir: {{medium: Memory, sizeLimit: 128Gi}}
      - name: model
        hostPath: {{path: {MODEL_HOSTPATH}, type: Directory}}
      - name: dg-cache
        hostPath: {{path: {DG_HOSTPATH}, type: DirectoryOrCreate}}
      - name: configs
        configMap: {{name: dsr1fp8-deepep-config}}"""

MOUNTS = """        volumeMounts:
        - {name: dshm, mountPath: /dev/shm}
        - {name: model, mountPath: /model, readOnly: true}
        - {name: dg-cache, mountPath: /dg-cache}
        - {name: configs, mountPath: /configs}"""

RESOURCES = """        resources:
          requests:
            nvidia.com/gpu: "4"
            cpu: "64"
            memory: 512Gi
          limits:
            nvidia.com/gpu: "4"
          claims:
          - name: compute-domain-channel"""

SCHED = """      nodeSelector:
        cloud.google.com/gke-nodepool: np-2
      tolerations:
      - {key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}
      - {key: kubernetes.io/arch, operator: Equal, value: arm64, effect: NoSchedule}"""


def render(cid, cfg):
    pw, dtp, dn = cfg['pworkers'], cfg['dtp'], cfg['dnodes']
    pn = pw * PREFILL_NODES_PER_WORKER
    total_nodes = pn + dn
    p = f'dsr1fp8-{cid}'
    parts = []

    parts.append(f"""# DSR1-FP8 {cid.upper()} — srt-slurm recipes/gb300-fp8/8k1k/stp/{cid}.yaml (verbatim flags).
# GENERATED BY gen-fp8-point.py — edit the generator, not this file.
# {pw} prefill workers (DEP8: tp=dp=ep=8, dp-attn, {PREFILL_NODES_PER_WORKER} nodes each = {pn} nodes)
# + 1 decode worker (DEP{dtp}: tp=dp=ep={dtp}, dp-attn, DeepEP low_latency, {dn} nodes)
# + 10 frontends. {total_nodes} nodes / {total_nodes * 4} GPUs. Concurrencies: {cfg['conc']}, req_rate inf.
# Image {IMAGE} + ai-dynamo {DYNAMO}. Model deepseek-ai/DeepSeek-R1-0528 (native FP8).
# Each prefill worker is its OWN 2-node StatefulSet (independent dynamo worker + dist-init leader).
apiVersion: resource.nvidia.com/v1beta1
kind: ComputeDomain
metadata:
  name: {p}-cd
  namespace: {NS}
spec:
  numNodes: {total_nodes}
  channel:
    resourceClaimTemplate:
      name: {p}-cd-channel
---
apiVersion: v1
kind: Service
metadata:
  name: {p}-frontend
  namespace: {NS}
spec:
  selector:
    app: {p}-frontend
  ports:
  - port: 8000
    targetPort: 8000
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {p}-frontend
  namespace: {NS}
spec:
  replicas: 10
  selector:
    matchLabels:
      app: {p}-frontend
  template:
    metadata:
      labels:
        app: {p}-frontend
    spec:
{SCHED}
      containers:
      - name: frontend
        image: {IMAGE}
        command: ["bash", "-c"]
        args:
        - |
          set -e
          pip install --quiet "ai-dynamo-runtime=={DYNAMO}" "ai-dynamo=={DYNAMO}"
          # small-file mirror at the SAME path workers register (/model). Required for discovery,
          # and doubly so here: mid/max set --skip-tokenizer-init, so the frontend tokenizes.
          until python3 -c "
          from huggingface_hub import snapshot_download
          import shutil, glob, os
          pth = snapshot_download('deepseek-ai/DeepSeek-R1-0528', allow_patterns=['*.json','*.py','tokenizer*','*.txt'])
          os.makedirs('/model', exist_ok=True)
          [shutil.copy(f, '/model/') for f in glob.glob(pth + '/*') if os.path.isfile(f)]
          "; do echo "small-file mirror failed, retry in 60s"; sleep 60; done
          exec python3 -m dynamo.frontend --http-port=8000
        env:
{env_block(INFRA_ENV)}
        - name: HF_HOME
          value: /tmp/hf
        - name: PYTHONUNBUFFERED
          value: "1"
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: "8"
            memory: 16Gi
        volumeMounts:
        - {{name: model, mountPath: /model}}
      volumes:
      - name: model
        emptyDir: {{}}""")

    # one 2-node StatefulSet per prefill worker
    for w in range(pw):
        name = f'{p}-prefill-{w}'
        parts.append(f"""apiVersion: v1
kind: Service
metadata:
  name: {name}
  namespace: {NS}
spec:
  clusterIP: None
  selector:
    app: {name}
  ports:
  - port: 5757
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: {name}
  namespace: {NS}
spec:
  serviceName: {name}
  replicas: {PREFILL_NODES_PER_WORKER}
  podManagementPolicy: Parallel
  selector:
    matchLabels:
      app: {name}
  template:
    metadata:
      labels:
        app: {name}
    spec:
{SCHED}
      resourceClaims:
      - name: compute-domain-channel
        resourceClaimTemplateName: {p}-cd-channel
      containers:
      - name: prefill
        image: {IMAGE}
        command: ["bash", "-c"]
        args:
        - |
          set -e
          pip install --quiet "ai-dynamo-runtime=={DYNAMO}" "ai-dynamo=={DYNAMO}"
          [ -f /model/model.safetensors.index.json ] || {{ echo "MODEL NOT SEEDED on this node"; exit 1; }}
          exec python3 -m dynamo.sglang \\
            {prefill_args()} \\
            --dist-init-addr {name}-0.{name}.{NS}.svc.cluster.local:5757 \\
            --nnodes {PREFILL_NODES_PER_WORKER} \\
            --node-rank $POD_INDEX \\
            --host 0.0.0.0
        env:
{pod_index_env()}
{env_block(INFRA_ENV)}
{env_block(COMMON_ENV)}
{env_block(UCX_ENV)}
{RESOURCES}
{MOUNTS}
{VOLUMES}""")

    # single multi-node decode StatefulSet
    dname = f'{p}-decode'
    parts.append(f"""apiVersion: v1
kind: Service
metadata:
  name: {dname}
  namespace: {NS}
spec:
  clusterIP: None
  selector:
    app: {dname}
  ports:
  - port: 5757
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: {dname}
  namespace: {NS}
spec:
  serviceName: {dname}
  replicas: {dn}
  podManagementPolicy: Parallel
  selector:
    matchLabels:
      app: {dname}
  template:
    metadata:
      labels:
        app: {dname}
    spec:
{SCHED}
      resourceClaims:
      - name: compute-domain-channel
        resourceClaimTemplateName: {p}-cd-channel
      containers:
      - name: decode
        image: {IMAGE}
        command: ["bash", "-c"]
        args:
        - |
          set -e
          pip install --quiet "ai-dynamo-runtime=={DYNAMO}" "ai-dynamo=={DYNAMO}"
          [ -f /model/model.safetensors.index.json ] || {{ echo "MODEL NOT SEEDED on this node"; exit 1; }}
          exec python3 -m dynamo.sglang \\
            {decode_args(dtp)} \\
            --dist-init-addr {dname}-0.{dname}.{NS}.svc.cluster.local:5757 \\
            --nnodes {dn} \\
            --node-rank $POD_INDEX \\
            --host 0.0.0.0
        env:
{pod_index_env()}
{env_block(INFRA_ENV)}
{env_block(COMMON_ENV)}
{env_block(DECODE_EXTRA_ENV)}
{env_block(UCX_ENV)}
        - name: NVSHMEM_DEBUG
          value: "INFO"
{RESOURCES}
{MOUNTS}
{VOLUMES}""")

    out = '\n---\n'.join(parts) + '\n'
    path = os.path.join(D, f'{cid}-{pw}p1d-dep8-dep{dtp}.yaml')
    with open(path, 'w') as f:
        f.write(out)
    return path, total_nodes


if __name__ == '__main__':
    for cid, cfg in CONFIGS.items():
        path, n = render(cid, cfg)
        pn = cfg['pworkers'] * PREFILL_NODES_PER_WORKER
        print(f'{cid}: {os.path.basename(path)}  '
              f'({cfg["pworkers"]}w prefill DEP8 x2n = {pn}n + decode DEP{cfg["dtp"]} x{cfg["dnodes"]}n '
              f'= {n} nodes / {n*4} GPUs; conc {cfg["conc"]})')
