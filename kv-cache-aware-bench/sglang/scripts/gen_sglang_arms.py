"""Generate operator-less sglang arms (GKE) for the selected DynoSim data points.

Arms (selected points, SELECTION.md; disagg goes straight to 72 GPU per 2026-08-11
directive — no 24-GPU disagg arms):
  sgl-smoke         1P+1D + frontend (kv defaults)   — smoke ladder S1/S2
  sgl-disagg72-rr   6P+12D (72 GPU), round-robin     — headline 6:12 / conc 384
  sgl-disagg72-kv   6P+12D (72 GPU), kv (defaults + temp 0) — sweep says defaults
                    capture the 72-GPU win (kv-nvda won 6:12/384; kv-t-c1.0 ≈ defaults
                    won 12:6/192); conservative point = rescale to 12P+6D
  sgl-agg-rr        6x TP4 agg workers, round-robin  — selected conc 16
  sgl-agg-kv        6x TP4 agg workers, kv defaults + temp 0 — sweep: kv-nvda
                    (pure defaults) beats every tuned variant at all agg concs
NOTE: a 72-GPU arm needs the ENTIRE 18-node pool (one worker per node); tear down
everything else on that pool first. Frontend is CPU-only and co-schedules.

Design decisions (validated 2026-08-11):
  - workers: lmsysorg/sglang:v0.5.14-cu130-runtime — the dynamo-1.3.1-tested pair
    (S1 on v0.5.17 crashed in dynamo/sglang/_compat.py: ServerArgs read-only after
    resolution in >=0.5.15; the extra pins sglang==0.5.14 to match).
  - frontend: nvcr trtllm-runtime:1.3.1 image (dynamo.frontend 1.3.1, already on nodes;
    router formula verified on this version). No sglang needed in the frontend.
  - single-node TP4 workers: no ComputeDomain/IMEX claim, NCCL_MNNVL_ENABLE=0;
    mrdma claims kept for NIXL cross-node KV transfer; UCX RoCE pins carried.
  - page-size 64 matches the trace's native hash block size.
"""
import copy
import shlex
import sys
from pathlib import Path

import yaml


def _str_presenter(dumper, data):
    # quote YAML-1.1 boolean lookalikes ("y","n","on","off",...) so k8s env
    # values stay strings instead of becoming booleans in the manifest
    if data.lower() in ("y", "n", "yes", "no", "true", "false", "on", "off"):
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="'")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, _str_presenter)

OUT = Path.home() / "kv-cache-aware-bench/sglang/manifests"
NS = "dynamo-cloud"
MODEL_DIR = "/model-cache/alisachen/Kimi-K2.5-NVFP4"
SERVED = "alisachen/Kimi-K2.5-NVFP4"
SGL_IMAGE = "lmsysorg/sglang:v0.5.14-cu130-runtime"
FE_IMAGE = "nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.3.1"
NODEPOOL = sys.argv[1] if len(sys.argv) > 1 else "np-1"

# v0.5.17 attempt failed S1: dynamo 1.3.1 _compat mutates resolved ServerArgs,
# read-only in 0.5.17 (AttributeError incremental_streaming_output). Fall back to
# the dynamo-tested pair: v0.5.14 image + full extra (pins sglang==0.5.14, matched
# sgl-kernel). Revisit latest image when ai-dynamo >1.3.1 ships 0.5.17 support.
PIP_INSTALL = ('pip install -q "ai-dynamo[sglang]==1.3.1" && '
               'python3 -c "import sglang,dynamo;print(\'sglang\',sglang.__version__)"')

ETCD = "http://dynamo-platform-etcd.dynamo-cloud.svc.cluster.local:2379"
NATS = "nats://dynamo-platform-nats.dynamo-cloud.svc.cluster.local:4222"

# DynamoBench-proven NIXL/UCX env for a4xmax (RDMA-KV-clean verdict manifests):
# without UCX_NET_DEVICES/UCX_TLS pins, UCX free-picks among 9 interfaces (incl.
# gve eth0) for the NIXL data channel -> NIXL_ERR_REMOTE_DISCONNECT + "Foreign
# traffic?" guard assertions (S2 catch).
UCX_ENV = [
    {"name": "UCX_TLS", "value": "cuda_copy,rc_x,tcp"},
    {"name": "UCX_NET_DEVICES",
     "value": "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1"},
    {"name": "UCX_MEMTYPE_CACHE", "value": "n"},
    {"name": "UCX_MEMTYPE_REG_WHOLE", "value": "n"},
    {"name": "UCX_CUDA_IPC_ENABLE_MNNVL", "value": "y"},
    {"name": "UCX_IB_GID_INDEX", "value": "5"},
    {"name": "UCX_IB_ROCE_LOCAL_SUBNET", "value": "y"},
    {"name": "UCX_IB_ROCE_SUBNET_PREFIX_LEN", "value": "64"},
    {"name": "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT", "value": "100000"},
    {"name": "SGLANG_DISAGGREGATION_WAITING_TIMEOUT", "value": "100000"},
    {"name": "SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE", "value": "100000"},
    # remaining proven set (DynamoBench dsr1-sweep rdmakv; LD_LIBRARY_PATH omitted:
    # this image has no /opt/nvidia/nvda_nixl or /usr/local/ucx — pip nixl wheel
    # is self-contained)
    {"name": "SGLANG_MOONCAKE_CUSTOM_MEM_POOL", "value": "True"},
    {"name": "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK", "value": "1"},
    {"name": "SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN", "value": "1"},
    {"name": "SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2", "value": "1"},
    {"name": "SGLANG_USE_MESSAGE_QUEUE_BROADCASTER", "value": "0"},
    {"name": "TORCH_DISTRIBUTED_DEFAULT_TIMEOUT", "value": "1800"},
    {"name": "TP_SOCKET_IFNAME", "value": "eth0"},
    {"name": "PYTHONUNBUFFERED", "value": "1"},
    {"name": "DYN_SKIP_SGLANG_LOG_FORMATTING", "value": "1"},
    {"name": "UCX_PROTO_INFO", "value": "y"},
    {"name": "NCCL_DEBUG", "value": "VERSION"},
]


def base_env(dyn_ns):
    return [
        {"name": "POD_UID", "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}}},
        {"name": "ETCD_ENDPOINTS", "value": ETCD},
        {"name": "NATS_SERVER", "value": NATS},
        {"name": "DYN_NAMESPACE", "value": dyn_ns},
    ]


def worker_env(dyn_ns):
    return base_env(dyn_ns) + UCX_ENV + [
        {"name": "DYN_SYSTEM_PORT", "value": "9090"},
        {"name": "HF_HOME", "value": "/model-cache"},
        {"name": "HF_HUB_OFFLINE", "value": "1"},
        {"name": "HF_MODULES_CACHE", "value": "/tmp/hf_modules"},
        {"name": "NCCL_MNNVL_ENABLE", "value": "0"},  # single-node TP4, no IMEX claim
        {"name": "NCCL_CUMEM_ENABLE", "value": "1"},
        {"name": "NCCL_SOCKET_IFNAME", "value": "eth0"},
        {"name": "GLOO_SOCKET_IFNAME", "value": "eth0"},
        {"name": "PYTORCH_CUDA_ALLOC_CONF", "value": "expandable_segments:True"},
        {"name": "SGLANG_DISABLE_REQUEST_LOGGING", "value": "1"},
    ]


def sgl_args(mode, extra=None):
    """dynamo.sglang passes unknown args through to sglang server args."""
    args = [
        "--model-path", MODEL_DIR,
        "--served-model-name", SERVED,
        "--skip-tokenizer-init",
        "--tp-size", "4",
        "--ep-size", "4",
        "--quantization", "modelopt_fp4",
        "--request-plane", "nats",  # frontend requests via NATS; default tcp -> "no responders" (S1 catch)
        "--host", "0.0.0.0",  # NIXL bootstrap server binds this; default 127.0.0.1 refuses cross-pod decode handshake (S1 catch)
        "--trust-remote-code",
        "--mem-fraction-static", "0.85",
        "--context-length", "262144",
        "--page-size", "64",
        "--watchdog-timeout", "1000000",  # DynamoBench-proven; default 300s can kill scheduler on 250k-token chunked prefills
        # REQUIRED for KV-aware routing: without this the engine publishes no KV
        # block events (glue logs use_kv_events=False), the router's index stays
        # empty and "kv" degenerates to load-based routing (silicon catch: agg
        # KV arm showed TTFT identical to RR, 433/438 batches 0 cached tokens)
        "--kv-events-config", '{"publisher":"zmq","endpoint":"tcp://*:5557","replay_endpoint":"tcp://*:5558"}',
    ]
    if mode == "prefill":
        args += ["--disaggregation-mode", "prefill",
                 "--disaggregation-transfer-backend", "nixl",
                 "--disaggregation-bootstrap-port", "30001",
                 "--chunked-prefill-size", "16384",
                 "--max-running-requests", "8"]
    elif mode == "decode":
        args += ["--disaggregation-mode", "decode",
                 "--disaggregation-transfer-backend", "nixl",
                 "--max-running-requests", "64"]
    else:  # agg: sim says per-worker batch >=8 hits the TPOT cliff at this ISL;
        # cap at 16 to allow RR skew while chunked prefill interleaves
        args += ["--chunked-prefill-size", "16384",
                 "--max-running-requests", "16"]
    return args + (extra or [])


def deployment(name, dyn_ns, container, replicas=1, gpu=False):
    dep = {
        "apiVersion": "apps/v1", "kind": "Deployment",
        "metadata": {"name": name, "namespace": NS,
                     "labels": {"app": name,
                                "nvidia.com/dynamo-graph-deployment-name": dyn_ns}},
        "spec": {
            "strategy": {"type": "Recreate"},
            "progressDeadlineSeconds": 3600,  # worker start ~15min (install+load+autotune); default 600 fails rollout-status waits
            "replicas": replicas,
            "selector": {"matchLabels": {"app": name}},
            "template": {
                "metadata": {"labels": {"app": name,
                                        "nvidia.com/dynamo-graph-deployment-name": dyn_ns},
                             "annotations": {"gke-gcsfuse/volumes": "true",
                                             "gke-gcsfuse/memory-limit": "8Gi",
                                             "gke-gcsfuse/ephemeral-storage-limit": "1200Gi"}},
                "spec": {
                    "containers": [container],
                    "nodeSelector": {"kubernetes.io/arch": "arm64",
                                     "cloud.google.com/gke-nodepool": NODEPOOL},
                    "tolerations": [
                        {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"},
                        {"key": "kubernetes.io/arch", "operator": "Equal",
                         "value": "arm64", "effect": "NoSchedule"}],
                    "volumes": [{"name": "model-cache",
                                 "persistentVolumeClaim": {"claimName": "model-cache"}}],
                },
            },
        },
    }
    if gpu:
        spec = dep["spec"]["template"]["spec"]
        spec["volumes"].append({"name": "shm",
                                "emptyDir": {"medium": "Memory", "sizeLimit": "250Gi"}})
        spec["resourceClaims"] = [{"name": "rdma",
                                   "resourceClaimTemplateName": "mrdma-all"}]
    return dep


def worker(kind, name, dyn_ns, replicas):
    c = {
        "name": kind,
        "image": SGL_IMAGE,
        "command": ["bash", "-c"],
        "args": [PIP_INSTALL + " && exec python3 -m dynamo.sglang " +
                 " ".join(shlex.quote(a) for a in sgl_args(kind))],
        "env": worker_env(dyn_ns),
        "resources": {"limits": {"nvidia.com/gpu": "4"}, "claims": [{"name": "rdma"}]},
        "securityContext": {"runAsUser": 0, "capabilities": {"add": ["IPC_LOCK"]}},
        "startupProbe": {"failureThreshold": 240,
                         "httpGet": {"path": "/live", "port": 9090},
                         "periodSeconds": 60, "timeoutSeconds": 20},
        "volumeMounts": [
            {"mountPath": "/model-cache", "name": "model-cache", "readOnly": True},
            {"mountPath": "/dev/shm", "name": "shm"}],
    }
    return deployment(name, dyn_ns, c, replicas, gpu=True)


def frontend(name, dyn_ns, router):
    flags = {"rr": ["--router-mode", "round-robin"],
             # 72-GPU disagg winner: defaults with temperature pinned (kv-t-c1.0)
             "kv": ["--router-mode", "kv",
                    "--router-temperature", "0.0",
                    "--router-queue-policy", "fcfs"],
             # kept for future tuned arms (trtllm 24-GPU lineage)
             "kvt": ["--router-mode", "kv",
                     "--router-temperature", "0.0",
                     "--router-kv-overlap-score-credit", "1.0",
                     "--router-kv-overlap-score-credit-decay", "0.8",
                     "--router-queue-policy", "fcfs"]}[router]
    c = {
        "name": "frontend",
        "image": FE_IMAGE,
        "command": ["python3"],
        "args": ["-m", "dynamo.frontend"] + flags + ["--request-plane", "nats"],
        "env": base_env(dyn_ns),
        "resources": {},
        "volumeMounts": [{"mountPath": "/model-cache", "name": "model-cache",
                          "readOnly": True}],
    }
    dep = deployment(name, dyn_ns, c)
    svc = {"apiVersion": "v1", "kind": "Service",
           "metadata": {"name": name, "namespace": NS},
           "spec": {"selector": {"app": name},
                    "ports": [{"name": "http", "port": 8000, "targetPort": 8000}]}}
    return [dep, svc]


# arm -> (dyn namespace, router, n_prefill, n_decode, n_agg)
ARMS = {
    "sgl-smoke":        ("sgl-smoke", "kv", 1, 1, 0),
    "sgl-disagg72-rr":  ("sgl-disagg72-rr", "rr", 6, 12, 0),
    "sgl-disagg72-kv":  ("sgl-disagg72-kv", "kv", 6, 12, 0),
    "sgl-agg-rr":       ("sgl-agg-rr", "rr", 0, 0, 6),
    "sgl-agg-kv":       ("sgl-agg-kv", "kv", 0, 0, 6),
}

OUT.mkdir(parents=True, exist_ok=True)
for arm, (dyn_ns, router, np_, nd, na) in ARMS.items():
    docs = frontend(f"{arm}-frontend", dyn_ns, router)
    if np_:
        docs.append(worker("prefill", f"{arm}-prefill", dyn_ns, np_))
    if nd:
        docs.append(worker("decode", f"{arm}-decode", dyn_ns, nd))
    if na:
        docs.append(worker("agg", f"{arm}-worker", dyn_ns, na))
    p = OUT / f"{arm}.yaml"
    p.write_text(yaml.dump_all(docs, sort_keys=False, default_flow_style=False))
    print(f"wrote {p.name}: {np_}P+{nd}D+{na}A, router={router}, pool={NODEPOOL}")
