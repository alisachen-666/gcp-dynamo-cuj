#!/usr/bin/env python3
"""Add the NCCL gIB plugin (apt method) to Dynamo worker workloads in raw manifests.

Raw-manifest translation of gpu-recipes' `network.gibInstall.method: apt` Helm block:
  - initContainer `nccl-gib-apt-installer` (worker image): installs nccl-gib from
    packages.cloud.google.com gpudirect-gib-apt, stages /usr/local/gib into an emptyDir
  - volume `gib` mounted read-only at /usr/local/gib in the worker container
  - worker env: LD_LIBRARY_PATH (gib + ucx + nixl + nvidia), NCCL_DEBUG=VERSION

Text-based (preserves comments). Only patches YAML docs that request nvidia.com/gpu
(frontends untouched); idempotent. Usage: add-gib-init.py <manifest.yaml> [...]
"""
import re, sys

LDP = ("/usr/local/gib/lib64:/usr/local/ucx/lib:/usr/local/ucx/lib/ucx:"
       "/opt/nvidia/nvda_nixl/lib/aarch64-linux-gnu:"
       "/opt/nvidia/nvda_nixl/lib/aarch64-linux-gnu/plugins:/usr/local/nvidia/lib64")

INIT_BLOCK = """      initContainers:
      # NCCL gIB plugin (GPUDirect RoCE transport for NCCL) — raw-manifest port of
      # gpu-recipes network.gibInstall method=apt. Stages /usr/local/gib for the worker.
      - name: nccl-gib-apt-installer
        image: __IMAGE__
        imagePullPolicy: Always
        command: ["/bin/bash", "-c"]
        args:
        - |
          set -ex
          export DEBIAN_FRONTEND=noninteractive
          apt-get update
          apt-get install -y --no-install-recommends ca-certificates curl gnupg
          curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /etc/apt/trusted.gpg.d/cloud.google.gpg
          echo 'deb https://packages.cloud.google.com/apt gpudirect-gib-apt main' > /etc/apt/sources.list.d/nccl-gib.list
          apt-get update
          apt-get install -y --no-install-recommends nccl-gib
          test -f /usr/local/gib/scripts/set_nccl_env.sh
          mkdir -p /target/usr/local/gib
          cp -a /usr/local/gib/. /target/usr/local/gib/
        volumeMounts:
        - {name: gib, mountPath: /target/usr/local/gib}
"""

GIB_MOUNT = '{name: gib, mountPath: /usr/local/gib, readOnly: true}'
ENV_LINES = ('        - name: LD_LIBRARY_PATH\n          value: "%s"\n'
             '        - name: NCCL_DEBUG\n          value: "VERSION"\n') % LDP


def patch_doc(doc):
    # require an actual GPU resource request (`nvidia.com/gpu: "N"`), not just the
    # toleration string, so frontends are skipped
    if not re.search(r'nvidia\.com/gpu:\s*"', doc) or 'nccl-gib-apt-installer' in doc:
        return doc, False
    m = re.search(r'^(\s+)image:\s*(\S+)', doc, re.M)
    image = m.group(2) if m else 'lmsysorg/sglang:v0.5.8.post1-cu130-runtime'
    # 1. initContainers before the pod's containers: line
    doc, n = re.subn(r'^      containers:\n', INIT_BLOCK.replace('__IMAGE__', image) + '      containers:\n', doc, count=1, flags=re.M)
    if not n:
        return doc, False
    # 2. gib volume (block or flow list under pod-level volumes:)
    doc, nv = re.subn(r'^      volumes:\n', '      volumes:\n      - {name: gib, emptyDir: {}}\n', doc, count=1, flags=re.M)
    if not nv:
        doc = doc.rstrip('\n') + '\n      volumes:\n      - {name: gib, emptyDir: {}}\n'
    # steps 3-4 apply to the WORKER container only — scope to the text after the pod's
    # `containers:` line so the init container's own volumeMounts are never matched
    head, sep, tail = doc.partition('      containers:\n')
    # 3. worker volumeMount: block style first, else extend a flow-style list
    tail2, nm = re.subn(r'^        volumeMounts:\n', '        volumeMounts:\n        - %s\n' % GIB_MOUNT, tail, count=1, flags=re.M)
    if not nm:
        tail2, nm = re.subn(r'(^        volumeMounts:\s*\[)(.*?)(\]\s*$)',
                            lambda mo: mo.group(1) + mo.group(2) + ', ' + GIB_MOUNT + mo.group(3),
                            tail, count=1, flags=re.M)
    if not nm:
        raise SystemExit('could not place gib volumeMount (no volumeMounts on worker container)')
    tail = tail2
    # 4. env additions after the worker's env: line
    tail, ne = re.subn(r'^        env:\n', '        env:\n' + ENV_LINES, tail, count=1, flags=re.M)
    if not ne:
        raise SystemExit('could not place env entries (no env: block on worker container)')
    return head + sep + tail, True


def main():
    import subprocess
    for path in sys.argv[1:]:
        raw = open(path).read()
        docs = raw.split('\n---\n')
        patched = 0
        out = []
        for d in docs:
            nd, ok = patch_doc(d)
            out.append(nd)
            patched += ok
        if patched:
            open(path, 'w').write('\n---\n'.join(out))
            # validate
            import yaml
            list(yaml.safe_load_all(open(path)))
        print(f'{path}: patched {patched} worker workload(s), YAML valid')


if __name__ == '__main__':
    main()
