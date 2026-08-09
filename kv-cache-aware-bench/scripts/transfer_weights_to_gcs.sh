#!/usr/bin/env bash
# Stream Kimi-K2.5-NVFP4 from the public HF bucket mirror into alisachen's GCS
# registry, without touching local disk (curl | gcloud storage cp -).
# Resumable: files already in GCS with matching size are skipped.
set -uo pipefail

HF_BUCKET="Alisa233/Kimi-K2.5-NVFP4-bucket"
DEST="gs://alisachen-models/alisachen/Kimi-K2.5-NVFP4"
PARALLEL=6
LIST=$(mktemp)

curl -s "https://huggingface.co/api/buckets/${HF_BUCKET}/tree" \
  | python3 -c "
import json,sys
for e in json.load(sys.stdin):
    if e['type']=='file': print(e['size'], e['path'])" > "$LIST"

total=$(awk "{s+=$1} END {printf \"%.0f\", s}" "$LIST")
echo "$(wc -l < "$LIST") files, $((total/1000000000)) GB -> $DEST"

transfer_one() {
  size=$1 path=$2
  url="https://huggingface.co/buckets/${HF_BUCKET}/resolve/${path}"
  dst="${DEST}/${path}"
  export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token)
  have=$(gcloud storage ls -l "$dst" 2>/dev/null | awk 'NR==1{print $1}')
  if [ "${have:-}" = "$size" ]; then echo "skip $path"; return 0; fi
  for attempt in 1 2 3; do
    export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token)
    if curl -sfL "$url" | gcloud storage cp - "$dst" 2>/dev/null; then
      have=$(gcloud storage ls -l "$dst" 2>/dev/null | awk 'NR==1{print $1}')
      if [ "${have:-}" = "$size" ]; then echo "ok   $path ($((size/1000000)) MB)"; return 0; fi
      echo "size-mismatch $path (got ${have:-none}, want $size), retry $attempt"
    else
      echo "fail $path attempt $attempt"
    fi
    sleep 5
  done
  echo "GIVEUP $path"; return 1
}
export -f transfer_one
export HF_BUCKET DEST

xargs -a "$LIST" -n2 -P"$PARALLEL" bash -c 'transfer_one "$@"' _
rc=$?

echo "== verification =="
export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token)
gcs_count=$(gcloud storage ls "${DEST}/**" 2>/dev/null | wc -l)
gcs_bytes=$(gcloud storage du -s "$DEST" 2>/dev/null | awk '{print $1}')
echo "GCS: ${gcs_count} objects, ${gcs_bytes} bytes (expect $(wc -l < "$LIST") files, ${total} bytes)"
if [ "$gcs_bytes" = "$total" ]; then echo "TRANSFER COMPLETE"; else echo "TRANSFER INCOMPLETE"; exit 1; fi
