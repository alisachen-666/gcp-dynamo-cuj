#!/usr/bin/env bash
# Seed the DSR1 NVFP4 checkpoint from GCS onto all 18 np-2 nodes' local SSD (REDACTED-GKE-CLUSTER).
# Adapted from seed-model-all-nodes.sh with two changes for this VM:
#   - auth via ADC token (the gcloud CLI account here is the scope-limited compute SA)
#   - idempotent: rsync no-ops on already-seeded nodes
set -euo pipefail
TOKEN=$(gcloud auth application-default print-access-token)
kubectl delete job dsr1-seed-np2 -n dynamo-cloud --ignore-not-found >/dev/null 2>&1
cat <<YAML | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: dsr1-seed-np2
  namespace: dynamo-cloud
spec:
  completions: 18
  parallelism: 18
  completionMode: Indexed
  backoffLimit: 6
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels: {app: dsr1-seed-np2}
    spec:
      restartPolicy: OnFailure
      nodeSelector: {cloud.google.com/gke-nodepool: np-3}
      tolerations:
      - {key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}
      - {key: kubernetes.io/arch, operator: Equal, value: arm64, effect: NoSchedule}
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchLabels: {app: dsr1-seed-np2}
            topologyKey: kubernetes.io/hostname
      containers:
      - name: seed
        image: gcr.io/google.com/cloudsdktool/google-cloud-cli:slim
        command: ["bash", "-c"]
        args:
        - |
          set -e
          gcloud storage rsync -r gs://alisachen-models/deepseek-ai/DeepSeek-R1-0528-NVFP4-v2 /model
          echo "seeded: \$(ls /model/*.safetensors 2>/dev/null | wc -l) shards on \$(hostname)"
        env:
        - {name: CLOUDSDK_AUTH_ACCESS_TOKEN, value: "$TOKEN"}
        resources:
          requests: {cpu: "16", memory: 16Gi}
        volumeMounts:
        - {name: model, mountPath: /model}
      volumes:
      - name: model
        hostPath: {path: /mnt/stateful_partition/kube-ephemeral-ssd/dsr1-model, type: DirectoryOrCreate}
YAML
echo "dsr1-fp4 seeder launched (18 np-2 nodes)"
until [ "$(kubectl get job dsr1-seed-np2 -n dynamo-cloud -o jsonpath='{.status.succeeded}' 2>/dev/null)" = "18" ]; do sleep 30; done
echo "all 18 seed pods succeeded"
kubectl logs -n dynamo-cloud job/dsr1-seed-np2 --tail=1 --prefix 2>/dev/null | grep -c "seeded:" || true
