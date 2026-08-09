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
    c["resources"] = resources
    claims = (svc.get("resources") or {}).get("claims")
    if claims:
        c["resources"]["claims"] = claims

    spec = {
        "containers": [c],
        "nodeSelector": pod.get("nodeSelector", {}),
        "tolerations": pod.get("tolerations", []),
        "volumes": copy.deepcopy(pod.get("volumes", []) or []),
    }
    if gpu:
        spec["volumes"].append({
            "name": "shm",
            "emptyDir": {"medium": "Memory", "sizeLimit": "64Gi"},
        })
    if pod.get("resourceClaims"):
        spec["resourceClaims"] = pod["resourceClaims"]

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
        annotations["gke-gcsfuse/ephemeral-storage-limit"] = "500Gi"
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": f"{dgd_name}-{comp_name}", "namespace": K8S_NAMESPACE,
                     "labels": labels},
        "spec": {
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
