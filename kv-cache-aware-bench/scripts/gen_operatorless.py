"""Convert disagg DynamoGraphDeployment manifests into operator-less plain
Kubernetes resources (Deployments + Service + ConfigMap).

Why: this cluster's account can create workloads but not RBAC, so the Dynamo
operator can't run. Workers/frontend discover each other via the pre-existing
etcd/NATS in dynamo-cloud, scoped per-arm with DYN_NAMESPACE.

Usage: python3 gen_operatorless.py  # reads manifests/arm2*.yaml, writes manifests/operatorless/
"""
import copy
from pathlib import Path

import yaml

MANIFESTS = Path.home() / "kv-cache-aware-bench/manifests"
OUT = MANIFESTS / "operatorless"
K8S_NAMESPACE = "dynamo-cloud"
# Pin every arm to one node pool = one NVL72 subblock, so prefill->decode KV
# transfer stays inside a single NVLink domain (S1 showed default scheduling
# scatters pods across racks). np-1 is the designated bench domain.
BENCH_NODEPOOL = "np-1"
ETCD = "http://dynamo-platform-etcd.dynamo-cloud.svc.cluster.local:2379"
NATS = "nats://dynamo-platform-nats.dynamo-cloud.svc.cluster.local:4222"

ARMS = ["arm2b-disagg-rr.yaml", "arm2c-disagg-kv-nvda.yaml", "arm2d-disagg-kv-tuned.yaml"]


def discovery_env(dyn_namespace):
    return [
        {"name": "ETCD_ENDPOINTS", "value": ETCD},
        {"name": "NATS_SERVER", "value": NATS},
        {"name": "DYN_NAMESPACE", "value": dyn_namespace},
    ]


def build_deployment(dgd_name, comp_name, svc, dyn_namespace, graph_envs):
    pod = svc["extraPodSpec"]
    c = copy.deepcopy(pod["mainContainer"])
    c["name"] = comp_name
    c.setdefault("env", [])
    c["env"] = graph_envs + c["env"] + discovery_env(dyn_namespace)
    if svc.get("envFromSecret"):
        c["envFrom"] = [{"secretRef": {"name": svc["envFromSecret"]}}]

    gpu = (svc.get("resources") or {}).get("limits", {}).get("gpu")
    resources = c.get("resources") or {}
    if gpu:
        resources.setdefault("limits", {})["nvidia.com/gpu"] = gpu
        # TRT-LLM TP=4 needs large shared memory
        c.setdefault("volumeMounts", []).append({"name": "shm", "mountPath": "/dev/shm"})
        # A4X MAX (per AI-Hypercomputer/gpu-recipes@fef5ad27 + DynamoBench m2r/m4r):
        # RDMA memory registration needs IPC_LOCK; UCX pins keep the KV-transfer
        # path off cross-rail pairs (8 rails = 8 disjoint /64s, GID index 5).
        sc = c.setdefault("securityContext", {})
        sc.setdefault("capabilities", {}).setdefault("add", []).append("IPC_LOCK")
        c["env"] += [
            # "0"/"1" (not y/n) — bare y/n round-trip as YAML booleans and break EnvVar
            {"name": "UCX_MEMTYPE_CACHE", "value": "0"},
            {"name": "UCX_IB_GID_INDEX", "value": "5"},
            {"name": "UCX_IB_ROCE_LOCAL_SUBNET", "value": "1"},
            {"name": "UCX_IB_ROCE_SUBNET_PREFIX_LEN", "value": "64"},
            {"name": "PYTORCH_CUDA_ALLOC_CONF", "value": "expandable_segments:True"},
            {"name": "TLLM_NUMA_AWARE_WORKER_AFFINITY", "value": "1"},
        ]
    c["resources"] = resources
    claims = (svc.get("resources") or {}).get("claims")
    if claims:
        claims = list(claims)
        if not any(cl.get("name") == "rdma" for cl in claims):
            claims.append({"name": "rdma"})
        c["resources"]["claims"] = claims

    node_selector = dict(pod.get("nodeSelector") or {})
    node_selector["cloud.google.com/gke-nodepool"] = BENCH_NODEPOOL
    spec = {
        "containers": [c],
        "nodeSelector": node_selector,
        "tolerations": pod.get("tolerations", []),
        "volumes": copy.deepcopy(pod.get("volumes", []) or []),
    }
    if gpu:
        spec["volumes"].append({
            "name": "shm",
            "emptyDir": {"medium": "Memory", "sizeLimit": "250Gi"},
        })
    if pod.get("resourceClaims"):
        # all 8 RDMA NICs per GPU pod (official a4xmax recipe pattern; template
        # mrdma-all pre-exists in dynamo-cloud and matches gpu-recipes' all-mrdma)
        rc = list(pod["resourceClaims"])
        if not any(r.get("name") == "rdma" for r in rc):
            rc.append({"name": "rdma", "resourceClaimTemplateName": "mrdma-all"})
        spec["resourceClaims"] = rc

    labels = {
        "app": f"{dgd_name}-{comp_name}",
        "nvidia.com/dynamo-graph-deployment-name": dgd_name,
    }
    annotations = {}
    if any(v.get("persistentVolumeClaim", {}).get("claimName") == "model-cache"
           for v in spec["volumes"]):
        # model-cache is a gcsfuse CSI volume: the sidecar must be injected
        annotations["gke-gcsfuse/volumes"] = "true"
        annotations["gke-gcsfuse/memory-limit"] = "8Gi"
        annotations["gke-gcsfuse/ephemeral-storage-limit"] = "1200Gi"
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": f"{dgd_name}-{comp_name}", "namespace": K8S_NAMESPACE,
                     "labels": labels},
        "spec": {
            # Recreate: with 30+ min weight loads, RollingUpdate strands old-template
            # pods (they must stay until new ones pass startup probes)
            "strategy": {"type": "Recreate"},
            "replicas": svc.get("replicas", 1),
            "selector": {"matchLabels": {"app": f"{dgd_name}-{comp_name}"}},
            "template": {"metadata": {"labels": labels, "annotations": annotations},
                         "spec": spec},
        },
    }


OUT.mkdir(exist_ok=True)
for arm_file in ARMS:
    docs = list(yaml.safe_load_all((MANIFESTS / arm_file).read_text()))
    dgd = next(d for d in docs if d and d.get("kind") == "DynamoGraphDeployment")
    others = [d for d in docs if d and d.get("kind") != "DynamoGraphDeployment"]
    dgd_name = dgd["metadata"]["name"]
    dyn_namespace = dgd_name.replace("kimi-k25-", "kvbench-")
    graph_envs = dgd["spec"].get("envs", [])

    out_docs = []
    for d in others:  # ConfigMap(s)
        d["metadata"]["namespace"] = K8S_NAMESPACE
        out_docs.append(d)

    for comp_name, svc in dgd["spec"]["services"].items():
        name = comp_name.lower()
        if svc.get("multinode"):
            raise SystemExit(f"{arm_file}: multinode service {comp_name} not supported operator-less")
        out_docs.append(build_deployment(dgd_name, name, svc, dyn_namespace, graph_envs))
        if svc.get("componentType") == "frontend":
            out_docs.append({
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {"name": f"{dgd_name}-frontend", "namespace": K8S_NAMESPACE},
                "spec": {
                    "selector": {"app": f"{dgd_name}-{name}"},
                    "ports": [{"name": "http", "port": 8000, "targetPort": 8000}],
                },
            })

    out_path = OUT / arm_file
    out_path.write_text(yaml.safe_dump_all(out_docs, sort_keys=False, default_flow_style=False))
    print(f"wrote {out_path.name}: {len(out_docs)} resources, DYN_NAMESPACE={dyn_namespace}")
