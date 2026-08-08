#!/usr/bin/env bash
# Seed DSv4 model + dynamo wheels from GCS onto all np-2 nodes' local SSD.
set -euo pipefail
TOKEN=$(gcloud auth application-default print-access-token)
kubectl delete job dsv4-seed -n dynamo-cloud --ignore-not-found >/dev/null 2>&1
cat <<YAML | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: dsv4-seed
  namespace: dynamo-cloud
spec:
  completions: 18
  parallelism: 18
  completionMode: Indexed
  backoffLimit: 6
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels: {app: dsv4-seed}
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
              matchLabels: {app: dsv4-seed}
            topologyKey: kubernetes.io/hostname
      containers:
      - name: seed
        image: gcr.io/google.com/cloudsdktool/google-cloud-cli:slim
        command: ["bash", "-c"]
        args:
        - |
          set -e
          gcloud storage rsync -r gs://REDACTED-MODELS-BUCKET/deepseek-ai/DeepSeek-V4-Pro /model
          gcloud storage rsync -r gs://REDACTED-MODELS-BUCKET/dynamo-wheels/81d0555 /wheels
          echo "seeded: \$(ls /model/*.safetensors | wc -l) shards, \$(ls /wheels/*.whl | wc -l) wheels on \$(hostname)"
        env:
        - {name: CLOUDSDK_AUTH_ACCESS_TOKEN, value: "$TOKEN"}
        resources:
          requests: {cpu: "16", memory: 16Gi}
        volumeMounts:
        - {name: model, mountPath: /model}
        - {name: wheels, mountPath: /wheels}
      volumes:
      - name: model
        hostPath: {path: /mnt/stateful_partition/kube-ephemeral-ssd/dsv4-model, type: DirectoryOrCreate}
      - name: wheels
        hostPath: {path: /mnt/stateful_partition/kube-ephemeral-ssd/dsv4-wheels, type: DirectoryOrCreate}
YAML
echo "dsv4 seeder launched (18 nodes)"
# per-node verification (lesson from decode-9): count nodes with full seed
sleep 30
until [ "$(kubectl get job dsv4-seed -n dynamo-cloud -o jsonpath='{.status.succeeded}' 2>/dev/null)" = "18" ]; do sleep 30; done
echo "all 18 seed pods succeeded; verifying per-node coverage:"
kubectl logs -n dynamo-cloud job/dsv4-seed --tail=1 --prefix 2>/dev/null | grep -c "seeded:" || true
