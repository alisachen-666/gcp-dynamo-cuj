#!/usr/bin/env bash
# Seed DSR1-NVFP4 from gs://alisachen-models onto ALL np-2 nodes' local SSD
# (/mnt/stateful_partition/kube-ephemeral-ssd/dsr1-model, plain files).
# Auth: short-lived token minted from this VM's SA, injected at apply time.
set -euo pipefail
TOKEN=$(gcloud auth print-access-token)
kubectl delete job dsr1-model-seed -n dynamo-cloud --ignore-not-found >/dev/null 2>&1
cat <<YAML | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: dsr1-model-seed
  namespace: dynamo-cloud
spec:
  completions: 18
  parallelism: 18
  completionMode: Indexed
  backoffLimit: 6
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels: {app: dsr1-model-seed}
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
              matchLabels: {app: dsr1-model-seed}
            topologyKey: kubernetes.io/hostname
      containers:
      - name: seed
        image: gcr.io/google.com/cloudsdktool/google-cloud-cli:slim
        command: ["bash", "-c"]
        args:
        - |
          set -e
          N=\$(ls /model/*.safetensors 2>/dev/null | wc -l)
          if [ "\$N" = "163" ]; then echo "already seeded (\$N shards)"; exit 0; fi
          gcloud storage rsync -r gs://alisachen-models/deepseek-ai/DeepSeek-R1-0528-NVFP4-v2 /model
          echo "seeded: \$(ls /model/*.safetensors | wc -l) shards"
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
echo "seeder launched (18 nodes)"
