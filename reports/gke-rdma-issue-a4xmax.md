# GKE Issue Report: RDMA fabric non-functional on `REDACTED-GKE-CLUSTER-OLD` (A4X Max / GB300)

- **Cluster**: `REDACTED-GKE-CLUSTER-OLD`, project `REDACTED-GCP-PROJECT`, region `us-east5` (endpoint 136.83.38.72)
- **Node pools affected**: `np-1` (17 nodes) and `np-2` (18 nodes) — i.e. **all GPU nodes, both NVLink domains**
- **Machine type**: `a4x-maxgpu-4g-metal` (arm64 Grace + 4x NVIDIA GB300, 8x 400G CX RDMA NICs per node)
- **GKE version**: v1.35.6-gke.1127000
- **Date observed**: 2026-07-29
- **Reported by**: repo owner (DeepSeek-R1 InferenceX-parity benchmarking on the np-2 NVL72 domain)
- **Severity**: Medium for single-domain NVLink workloads (unaffected); **High for anything
  needing RDMA** — cross-domain jobs, official A4X Max recipes with mRDMA claims, RoCE fallback.

## TL;DR

The RDMA hardware is present and healthy at the PCIe/verbs level (8x mlx5 RoCE NICs per node),
but RDMA is unusable end-to-end, in three mutually consistent ways:

1. **DRA does not expose the NICs**: the `dra.net` (DraNet) driver publishes every NIC with
   `rdma=false`, so the `mrdma.google.com` DeviceClass matches **zero devices** and any pod
   requesting the standard `all-mrdma` ResourceClaim is permanently unschedulable.
2. **The NICs have no IP addressing**: only link-local IPv6 GIDs (`fe80::`) exist, so RoCEv2
   has no routable GIDs.
3. **The data path is dead**: a forced NCCL-over-RoCE allreduce between two nodes fails on
   every HCA with `status=12` (IBV_WC_RETRY_EXC_ERR, vendor err 129) — packets are sent,
   nothing is ever acknowledged, collectives time out.

All three symptoms point at one root cause: **the GKE RDMA additional-network configuration
(additional node networks / GKENetworkParamSet for the `gpu{0-3}rdma{0,1}` NICs) appears to
have never been applied to this cluster's GPU node pools.** NVLink (MNNVL via ComputeDomain /
IMEX, `nvidia-dra-driver-gpu`) is fully functional — validated at 730 GB/s across all 72 GPUs
of np-2 — so this is specifically the Ethernet/RoCE side.

## Evidence trace (commands + observed output)

### 1. Symptom: pods with the standard mRDMA claim cannot schedule

Pod spec used the claim exactly as in AI-Hypercomputer/gpu-recipes
`src/helm-charts/a4xmax/inference-templates/common/templates/all-mrdma.yaml`
(deviceClassName `mrdma.google.com`, ExactCount 8):

```
kubectl describe pod <pod>
  Warning  FailedScheduling  ... 0/36 nodes are available:
  18 cannot allocate all claims, 18 node(s) didn't match Pod's node affinity/selector.

kubectl get resourceclaims
  <pod>-rdma-...   pending
```

### 2. The DeviceClass is a filter over DraNet attributes

```
kubectl get deviceclass mrdma.google.com -o jsonpath='{.spec}'
  selectors[0].cel.expression:
    device.driver == "dra.net" && has(device.attributes['dra.net'].rdma)
      && device.attributes['dra.net'].rdma == true
```

### 3. DraNet publishes NO rdma-capable devices — on any GPU node, either pool

```
kubectl get resourceslices -o json | jq '[.items[].spec.driver] | unique'
  ["compute-domain.nvidia.com", "dra.net"]      # note: no mrdma.google.com slices

# per-node summary (identical for ALL 17 np-1 and ALL 18 np-2 nodes):
  gke-REDACTED-GKE-CLUSTER-OLD-np-2-9d137666-3xp9: devices=4 rdma_true=0
  ...
  gke-REDACTED-GKE-CLUSTER-OLD-np-1-c198133d-3w4j: devices=4 rdma_true=0
  ...

# sample device from a np-2 node's dra.net ResourceSlice:
  { "name": "pci-0016-01-00-0",
    "rdma": {"bool": false},
    "pciDevice": "Infrastructure Data Path Function" }
```

Only 4 netdev-class devices per node are published (IDPF functions), none flagged rdma.
The 8 RDMA NICs are absent from the DRA inventory entirely.

### 4. The hardware IS present and healthy (privileged diagnostic pod on np-2)

```
ls /sys/class/infiniband/   -> mlx5_0 mlx5_1 mlx5_2 mlx5_3 mlx5_4 mlx5_5 mlx5_6 mlx5_7
ls /dev/infiniband/         -> rdma_cm uverbs0..uverbs7
ls /sys/class/net/          -> ... gpu0rdma0 gpu0rdma1 gpu1rdma0 gpu1rdma1
                                   gpu2rdma0 gpu2rdma1 gpu3rdma0 gpu3rdma1 ...
```

8 HCAs, 8 uverbs devices, 8 host netdevs (2 per GPU) — matching the A4X Max spec and the
`count: 8` the official recipes expect.

### 5. Data-path test: forced NCCL-over-RoCE allreduce FAILS (both domains)

Test: 2 nodes x 4 GPUs, hostNetwork + privileged (to bypass DRA), `NCCL_MNNVL_ENABLE=0`,
`NCCL_NET=IB`, `NCCL_IB_HCA=mlx5` (so NCCL must use RoCE or fail — no silent fallback),
1 GiB fp32 allreduce. Harness: `~/dsr1-pareto/common/run-rdma-test.sh` (POOL=np-1|np-2).

NCCL device init succeeds — all 8 NICs enumerated as RoCE 400G:

```
NCCL INFO NET/IB: [0] mlx5_0:uverbs0:1/RoCE provider=Mlx5 speed=400000 ...
NCCL INFO NET/IB : Using [0]mlx5_0:1/RoCE [1]mlx5_1:1/RoCE ... [7]mlx5_7:1/RoCE [RO];
                   OOB eth0:192.168.63.133<0>
```

...but every transfer fails with transport retry exhaustion, and the GID table shows
link-local-only addressing:

```
transport/net_ib.cc:2472 NCCL WARN NET/IB: Got completion from peer 192.168.0.20
  with status=12 opcode=0 len=0 vendor err 129 (Recv)
  localGid  fe80::92e3:17ff:fe1c:647c
  remoteGid fe80::92e3:17ff:fe20:8c8c   hca mlx5_6
[rank0] ncclRemoteError: A call failed possibly due to a network error ...
  Collective WorkNCCL(... OpType=ALLREDUCE ...) timeout
```

- `status=12` = IBV_WC_RETRY_EXC_ERR: the QP retried to exhaustion with zero completions —
  no packet ever made the round trip.
- `fe80::` GIDs only = the `gpuXrdmaY` interfaces have **no IP addresses**, so no routable
  RoCEv2 GIDs exist. (An RoCEv2-ready NIC shows IPv4-mapped GIDs.)

Result identical on np-2 (`3xp9` <-> `6l20`) and np-1 (`3w4j` <-> `6sgc`).

### 6. Control (proves the rest of the stack is fine)

- Same test methodology over NVLink (ComputeDomain channel claims, `NCCL_MNNVL_ENABLE=1`):
  **PASS** — 2-node: 723 GB/s busbw; **18-node/72-GPU single clique: 730 GB/s** (np-2).
- So: NCCL, drivers, GPUs, DRA plumbing for `compute-domain.nvidia.com`, and the pod
  scheduling path all work. The failure is isolated to the RoCE/Ethernet RDMA layer.

## NEW (2026-07-29 late): probable mechanism — `asapd-lite` crashloop in kube-system

The mRDMA NIC management daemon is failing fleet-wide, and it is the component responsible
for exactly the missing pieces:

```
kubectl get pods -n kube-system | grep asapd
  asapd-lite-*   0/1  CrashLoopBackOff/Running(not Ready)   ~278-285 restarts / 18h  (ALL 36 pods)

kubectl logs -n kube-system asapd-lite-2dq5n --tail
  controller.cc:28] Detected firmware version: 40.46.7004
  asapd-lite.cc:46] Build label: doca_3.1.0105_commit_
  metric-collector: /mrdma_nic/* metrics registered
  main.cc:73] Running ipam in ipvlan mode
  main.cc:51] Running command: /usr/local/sbin/ipam --mode=ipvlan --setup_systemd_timer=true
  <no further output; pod never becomes Ready>

kubectl describe pod ...
  Last State: Terminated, Reason: Error, Exit Code: 143 (SIGTERM ~90s after start)
```

`ipam` is the component that assigns IP addresses to the `gpuXrdmaY` RDMA NICs. It never
completes -> NICs never get IPs (fe80:: GIDs only) -> RoCE data path dead -> DraNet publishes
`rdma=false`. Every symptom in this report is downstream of this loop. Suggested first
diagnostic for the owner: why does ipam hang/fail on these nodes — most plausibly the RDMA
additional-network objects it expects are missing (consistent with the root-cause hypothesis
below).

## Root cause — CONFIRMED: GKE multi-network objects for RDMA are absent

Direct check (2026-07-29):

```
kubectl get networks.networking.gke.io   -> No resources found
kubectl get gkenetworkparamsets          -> No resources found
```

A correctly configured A4X Max cluster carries one `Network` + `GKENetworkParamSet` pair per
RDMA NIC (the GKE multi-networking objects that drive NIC IP assignment and pod attachment).
This cluster has ZERO. Everything else follows:

- `asapd-lite`'s `ipam --mode=ipvlan` has no network definitions to apply -> crashloops,
  NICs never get IPs (only `fe80::` link-local GIDs) -> RoCE data path dead (status=12).
- DraNet builds its inventory from kernel netlink state and cloud network associations
  (source: github.com/google/dranet `pkg/inventory/db.go` — `discoverNetworkInterfaces` via
  netlink `LinkList`, `discoverRDMADevices` sets `rdma := rdmamap.IsRDmaDeviceForNetdevice()`,
  `addCloudAttributes` maps the cloud network). With the NICs unconfigured/unattached, the
  `gpuXrdmaY` interfaces never appear as allocatable rdma devices in ResourceSlices.
- Reference for the healthy state: DraNet `examples/demo_gke_rdma/README.md` shows a working
  GKE RDMA cluster publishing `gpu6rdma0` with `dra.net/rdma: true` AND `dra.net/cloudNetwork`
  attributes — exactly what is missing here.

Fix: create the RDMA `Network`/`GKENetworkParamSet` objects (and ensure the node pools'
additional-network attachment) per the A4X Max provisioning guide, then confirm asapd-lite
pods go Ready and ResourceSlices show `rdma=true` devices.

Reference point: a sibling cluster's working deployment (values file for
`dynamo-dsr1-fp4-8k1k-mid`) uses the identical DeviceClass (`mrdma.google.com`, count 8) with
`rdma.enabled: true` successfully — the recipe layer is fine; it is this cluster's network
configuration that differs.

## Impact

| Workload type | Impact |
|---|---|
| Single-NVL72-domain jobs (e.g. our 18-node InferenceX mid_curve parity benchmark) | **None** — all data planes (NCCL, DeepEP/NVSHMEM, NIXL/mooncake KV transfer) are pinned to NVLink and validated. We schedule pods with the mRDMA claim stripped. |
| Official A4X Max recipes as-published (all include the `all-mrdma` claim) | **Blocked** — pods Pending forever until the claim is removed by hand. |
| Cross-domain / multi-rack topologies (np-1 + np-2 together, >72 GPU jobs) | **Impossible** — no NVLink between domains and no working RDMA. |
| RoCE as a fallback path for any misconfigured NVLink component | **Unavailable** — the practical fallback becomes TCP (~10 GB/s), a silent performance cliff. |

## Requested fixes

1. Attach/repair the RDMA additional node networks for pools `np-1` and `np-2` so the 8
   `gpuXrdmaY` NICs per node receive IP addressing (RoCEv2 GIDs).
2. Ensure DraNet then publishes those NICs with `rdma=true` so `mrdma.google.com`
   ResourceClaims allocate (official recipe compatibility).
3. (While in there) Consider granting cluster RBAC to the benchmarking service account
   `alisa-gcs-sa@REDACTED-GCS-BUCKET.iam.gserviceaccount.com` (numeric ID
   `103697796161254948063` — tokens lack email scope), so automation does not depend on
   daily-expiring user ADC.

## Verification (we can run this on request — ~5 minutes, 2 nodes)

1. `kubectl get resourceslices -o json | jq ...` -> expect `rdma_true=8` per GPU node.
2. `POOL=np-2 ~/dsr1-pareto/common/run-rdma-test.sh` -> expect `NET/IB` transfers to complete
   with IB-class busbw and `rank N: OK` from all 8 ranks (instead of status=12).
3. Schedule a pod with the standard `all-mrdma` claim -> expect Running (instead of Pending).

## Artifacts

- Test harness: `~/dsr1-pareto/common/run-rdma-test.sh` (self-contained; picks 2 nodes from
  POOL, forces NCCL over RoCE, fails loudly)
- Raw logs: session task outputs (hardware diag; np-2 and np-1 failure traces with full NCCL
  WARN lines)
- Related validation data: MNNVL 730 GB/s full-domain result (proves everything-but-RoCE is
  healthy)

## RESOLUTION ADDENDUM (2026-08-05) — FIXED on the recreated `REDACTED-GKE-CLUSTER`

The cluster `REDACTED-GKE-CLUSTER` was recreated (new endpoint; nodes rebuilt) with managed
networking-DRA (`gke-managed-networking-dra-driver`), and the RDMA fabric is now
**functional end-to-end — via DRA claims**:

1. **DRA layer**: DraNet publishes 288 devices with `dra.net/rdma=True` (36 nodes x 8 NICs);
   the legacy `networking.gke.io` Network-objects path is gone. `mrdma.google.com` claims
   now allocate (`allocated,reserved`) and pods schedule — previously permanently Pending.
2. **Addressing**: inside a pod holding an 8-NIC mrdma claim, `mlx5_*` carries **global
   RoCEv2 GIDs** (`2a03:83e4:...`, GID indexes 2-5) — previously link-local `fe80::` only.
3. **Data path**: cross-node `ib_write_bw` between two DRA-claimed pods:
   **387.8 Gb/s average (388.1 peak) on a single mlx5 NIC** — line rate for 400G.
   (Evidence: results/rdma-dra3-evidence.log in the GCS archive.)

Operational caveats:
- **The claim is mandatory.** hostNetwork / unclaimed access still sees link-local-only GIDs
  and fails QP->RTR ("Unable to Connect the HCA's through the link") — dead by design; NICs
  are configured into a pod's namespace only at claim time.
- **DRA allocation happens at scheduling.** Pods pinned with `nodeName:` bypass the scheduler
  and hang in ContainerCreating ("not allowed to use ResourceClaim") — use nodeSelector.
- Implications: the `all-mrdma` claim pattern from the team's helm values now works here;
  `worker-env-canonical.yaml`'s dormant `rc_x` UCX transport can activate (with the claim
  added to worker pods); cross-NVL72-domain scaling and RoCE KV-transfer experiments are
  unblocked.
