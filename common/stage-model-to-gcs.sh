#!/usr/bin/env bash
# Stage the NVFP4 model from the M2 decode pod to gs://alisachen-models/deepseek-ai/.
# Waits for the pod's download to finish, materializes the HF snapshot (deref symlinks),
# installs cloud CLI in-pod, uploads with a VM-minted token (re-mints per attempt).
set -uo pipefail
NS=dynamo-cloud
DEST=gs://alisachen-models/deepseek-ai/DeepSeek-R1-0528-NVFP4-v2

POD=$(kubectl get pods -n $NS -l app=dsr1-m2-decode -o jsonpath='{.items[0].metadata.name}')
echo "using pod: $POD"

echo "waiting for model download to complete..."
until kubectl logs -n $NS "$POD" 2>/dev/null | grep -q "model ready at"; do sleep 60; done
MODEL_DIR=$(kubectl logs -n $NS "$POD" | grep -oE "model ready at .*" | head -1 | sed 's/model ready at //')
echo "model dir in pod: $MODEL_DIR"

echo "materializing snapshot (deref symlinks) in pod..."
kubectl exec -n $NS "$POD" -- bash -c "mkdir -p /data/hf/export && cp -rL '$MODEL_DIR/.' /data/hf/export/ && du -sh /data/hf/export"

echo "installing cloud CLI in pod (arm64 tarball)..."
kubectl exec -n $NS "$POD" -- bash -c '
  [ -x /tmp/google-cloud-sdk/bin/gcloud ] && exit 0
  cd /tmp && curl -sSL https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-arm.tar.gz | tar xz'

echo "uploading to $DEST ..."
for attempt in 1 2 3 4 5 6; do
  TOKEN=$(gcloud auth print-access-token)
  kubectl exec -n $NS "$POD" -- bash -c \
    "CLOUDSDK_AUTH_ACCESS_TOKEN='$TOKEN' /tmp/google-cloud-sdk/bin/gcloud storage rsync -r /data/hf/export $DEST" \
    && { echo "UPLOAD COMPLETE"; break; }
  echo "attempt $attempt incomplete; re-minting token and resuming..."
  sleep 10
done

echo "verifying..."
gcloud storage du -s "$DEST" 2>/dev/null
gcloud storage ls "$DEST/**" 2>/dev/null | wc -l
echo "STAGING DONE"
