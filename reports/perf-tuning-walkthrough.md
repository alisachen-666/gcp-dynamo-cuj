# E2E Performance Tuning Walkthrough — DSR1-FP4 & DSv4 on GKE A4X Max (GB300)

A case-study record of every performance gap we hit reproducing InferenceX's published
GB300 NVL72 numbers, how each was diagnosed, what the root cause turned out to be, and the
verified outcome of the fix. Written 2026-08-01. Sources: `README.md`, `dsv4-sweep/PLAN.md`,
`dsr1-sweep/reference/COMPARISON.md`, `reports/benchmark-report-{dsr1,dsv4}-8k1k.md`, and the
evidence files in `*/results-summary/`.

**Stack under test**: DeepSeek-R1 NVFP4 and DeepSeek-V4-Pro mxfp4, NVIDIA Dynamo
disaggregated prefill/decode (operator-less: etcd + NATS + plain Deployments/StatefulSets),
sglang workers, NIXL/mooncake KV transfer over MNNVL, 18-node NVL72 domains
(`REDACTED-GKE-CLUSTER-OLD` → `REDACTED-GKE-CLUSTER`), benchmarked with srt-slurm's vendored sa-bench at ISL 8192 /
OSL 1024. Parity target: InferenceX `dsr1-fp4-gb300-dynamo-sglang` and
`dsv4-fp4-gb300-dynamo-sglang-mtp`.

## Where we ended up (out tok/s per decode GPU, ours vs InferenceX)

| Point | First result | Final result | Fix that closed it |
|---|---|---|---|
| DSR1 ll conc-64 | −6.0% | **+4.4%** | measurement methodology (case 1) |
| DSR1 mid conc-2048 | −2.7% (TPOT +41%) | −2.7% (TPOT +4.4% w/ NATS) | request-plane parity (case 5) |
| DSR1 mid conc-4096 | +9.4% | +9.4% | none needed (their curve regresses) |
| DSR1 max_tpt conc-2048 | −9.0% | **−7.6% (TCP), OPEN** | dispatch burstiness, plane-independent (case 6) |
| DSv4 p3 conc-256 | −8.0% (TTFT p99 +71%) | **+0.3%** | JIT cache persistence (case 4) |
| DSv4 p4 conc-256 | +0.0% | +0.0% | — (the anchor point) |
| DSv4 p5 conc-512 | −18.2% | **+1.8%** | config-drift fix (case 3) |
| DSv4 p6 conc-1024 | −1.6% (drift) | **+0.5%** | config-drift fix (case 3) |
| DSv4 p7 conc-4096 | −2.1% (drift) | **−0.4%** (TPOT −1.9%) | config-drift fix (case 3) |
| DSv4 p8 conc-8192 | −17.9% | ≈ parity | config-drift fix (case 3) |
| DSv4 p2 conc-8/32/64 | uncollected | **+18.6% / +9.5% / +8.0%** | final collection (2026-08-05) — DSv4 10/10 |

Latency ended at or better than parity on every clean point; interactivity better than theirs
on all comparable DSv4 points and all DSR1 low-latency points.

---

## Case 1 — The gap that was a ruler error: measurement methodology (DSR1 ll)

**Symptom.** Low-latency conc-64 read 272.9 out tok/s/decode-GPU vs their 290.3 (−6%), with no
config difference we could find.

**Investigation.** Compared our bench invocation against srt-slurm's `bench.sh` defaults:
theirs sends 2× concurrency warmup prompts @ rate 250 and measures over **10×** concurrency
prompts; we were warming with 1× and measuring 4×. Throughput = tokens ÷ wall-time *including*
the ramp (0→conc) and drain (conc→0) edges — short runs overweight those partial-load edges.

**Root cause.** Not the server: measurement fairness. Proven by rerunning the same server both
ways: throughput +11.1% (272.9 → 303.1) with **TPOT flat (+1.4%)** — a faster server would
have shown faster tokens, a fairer ruler doesn't.

**Fix & outcome.** Adopted 2×-warmup/10×-measured as mandatory for every recorded point
(README "Benchmark measurement methodology"); the warmup half separately cut TTFT median 26%
(kernel autotune/allocator warmth). Conc-64 flipped from −6% to **+4.4%**. DSv4 went further:
3× warmup + a 5× full-pressure "long-run warm" pass (discarded) before the 10× record, because
throttled warmup does not exercise the batch shapes full pressure produces.

**Lesson.** Before debugging the machine, verify the ruler. Any point measured under a
different protocol is marked `(base)` and never compared silently.

## Case 2 — The gap that wasn't there: KV transport (MNNVL vs RDMA)

**Symptom.** Standing suspicion that our KV path (cuda_ipc over MNNVL, because the cluster's
RoCE fabric is broken — see `reports/gke-rdma-issue-REDACTED-GKE-CLUSTER-OLD.md`) cost throughput vs
InferenceX's RDMA-capable Slurm racks.

**Investigation.** Instrumented every disagg worker with `UCX_PROTO_INFO=y` (transport
evidence policy, README): protocol tables showed all CUDA transfers on `zero-copy cuda_ipc`,
tcp on control lanes only. Sized the traffic: ~1.2 GB/s KV ingress at conc 64 against a
measured 723–730 GB/s NVLink fabric — <0.5% utilization, ~1–3 ms per request inside 6+ s
requests.

**Root cause.** None — the case-1 rerun doubled as the controlled experiment: methodology
change moved throughput +11.1% while TPOT stayed flat, which a transport bottleneck cannot do.

**Lesson.** Every disagg run carries transport evidence so this class of question is answered
from logs, not re-litigated. The MNNVL path is throughput-neutral for this workload.

## Case 3 — Silent config drift: DSv4 p5–p8 (−18%)

**Symptom.** p5 −18.2% and p8 −17.9% vs InferenceX, while p4 matched to 4 significant figures
— same stack, wildly different gaps.

**Investigation.** Line-diffed the generated manifests against the vendored srt-slurm recipes.
`gen-point.py` derived p5–p8 from p3's manifest, parameterizing only TP/EP/memory knobs —
silently inheriting p3's values for everything else. But the recipes *deliberately* retune with
concurrency: decode DeepGEMM tokens/rank 4096 (we ran p3's 2048), p8 MTP 1-step/2-draft (we
ran 3/4), context-length 9216 (we ran 16384), plus mem-fraction/cuda-graph/swa deltas.

**Root cause.** Inheritance-based manifest generation. The p8 drift signature was
self-diagnosing once read correctly: −17.9% throughput with **+45% interactivity** — deep
speculation buying per-user latency with the batch's compute.

**Fix & outcome.** `gen-point.py` extended to parameterize every drifted knob from recipe
values; drift results quarantined (`results-summary/drift/`, `*` in every table). Verified:
p8 corrected rerun took TTFT p99 from +109% to +3.7% and TPOT to −2.1% (throughput ≈parity by
log-plateau analysis, case 7); p5 clean make-up landed **+1.8%** (1650.2 vs 1620.9). The FP8
generator was later written inversely — every value an explicit recipe-sourced entry — so this
class of drift cannot recur silently.

**Lesson.** Deriving configs by copy + partial substitution is how parity dies quietly. Diff
generated artifacts against source recipes, mechanically, every time.

## Case 4 — JIT compiles inside the measured window: DSv4 p3 (−8%, TTFT p99 +71%)

**Symptom.** The one clean-parity point with a gap: −8.0% throughput and p99 TTFT 54.6 s vs
their 32.0 s — while *median* TTFT was within +4%. A pure tail effect.

**Investigation.** Reconstructed the bench timeline (measured window 12:02:13→12:08:19 from
the result JSON's date−duration) and swept the prefill log: DeepGEMM `TF32_HC_PRENORM_GEMM`
JIT sessions fired repeatedly — during warmup (28.9 s stall), during long-warm (36.7 s stall
ending 4 s before measurement), and **at 12:07:08, inside the measured window**, freezing the
*single* prefill worker 12.3 s + 12.5 s. Evidence:
`dsv4-sweep/results-summary/p3-jit-stall-evidence.txt`.

**Root cause.** The manifests persisted only the flashinfer cache; `/root/.cache/deep_gemm`
was cold in every pod, and DeepGEMM compiles lazily **per batch shape** — new shapes kept
appearing 11 minutes into traffic. ~25 s of prefill outage in a 366 s window ≈ 6.8% ≈ the −8%
(the 8:1 workload is prefill-bound); 12 s freezes ≈ the p99 tail. srt-slurm hosts accumulate
this cache across their whole sweep; our pods started from zero every time.

**Fix & outcome.** (a) `dsv4-dg-cache` hostPath mounted at `/root/.cache/deep_gemm` in every
manifest; (b) record protocol v2: a full **discarded bench pass** (cache filler) before the
record pass — strictly stronger than longer warmup because shape coverage grows with total
prompts seen, not warm minutes; (c) a post-run **assertion** that scans every prefill log for
JIT sessions and >5 s reporter gaps inside the measured window, stamping VALID/SUSPECT.
Re-record: **+0.3%** throughput, TTFT p99 gap +70.7% → +6.8%, assertion VALID. The +5-day
image deviation (forced by Docker Hub pruning the pinned nightly) was thereby exonerated.

**Lesson.** Warm state is part of the configuration. And a record run should carry a
machine-checkable validity verdict, not a vibe.

## Case 5 — The TPOT "deficit" that was a scheduling profile: DSR1 mid_curve (+24–41% TPOT)

**Symptom.** At mid_curve our TPOT ran 16.2–18.7 ms vs their eerily flat ~13.1–13.6 ms across
an 8× concurrency sweep — yet throughput was within ±3% and E2E within ±8%, and our TTFT
median was 16–77% *better*.

**Investigation, in three layers.**
1. *Data hygiene first*: their exported CSVs label latency columns "(ms)" but the values are
   **seconds** (proven by the identity Interactivity = 1/TPOT holding exactly on every row),
   and our earlier report compared our mean-derived interactivity against their median —
   both corrected before any tuning conclusions were drawn.
2. *Arithmetic*: E2E = TTFT + TPOT×(N−1). With completion times equal and our first tokens
   earlier, our measured TPOT is **forced** higher — an identity, not a defect. Decode logs
   confirmed the mechanism: ~15 requests resident per dp rank (eager admission → fatter,
   slower decode batches); their flat TPOT implies a leaner held batch with the queue
   absorbing load (their TTFT median 3.4 s vs our 0.79 s at conc 512).
3. *A real parity break underneath*: srt-slurm's runner injects `--request-plane nats` into
   every worker (schema default — the submission-branch recipes don't even carry the key); our
   manifests omitted it and dynamo 0.8.1 defaulted to **TCP** (confirmed:
   `NetworkManager with TCP request plane` in the archived decode logs). Explicit config got
   translated; implicit config got dropped.

**Fix & outcome.** One-variable A/B (`m4n`): identical mid_curve conc-2048 with NATS. Result:
TPOT 18.71 → **13.83 ms** (gap vs theirs +41% → +4.4%) and TTFT median 23.6 → 33.8 s — our
profile moved almost exactly onto theirs. Throughput 788.8 vs TCP's 816.5.

**Lesson.** The "gap" was profile shaping, not kernel speed — and on this stack TCP is
arguably the *better* operating point (more throughput, much better TTFT); NATS is retained as
the like-for-like mode for InferenceX comparisons. Also: check what the reference runner
*injects*, not just what its recipes say.

## Case 6 — Bursty dispatch starves prefill: DSR1 max_tpt (−9%)

**Symptom.** Newly collected max_tpt (10 prefill workers, DEP32 decode, conc 2048, NATS):
TPOT +0.4%, interactivity −0.4%, TTFT p99 **−5%** — but TTFT *median* +65.8% and throughput
−9.0%.

**Investigation (elimination chain).** Config: mechanical flag+env diff vs recipe — exact.
Balance: 13.8–14.9 M tokens per worker (1.08× skew) — clean. Compute: 9.4–10.2 k tok/s/GPU
while batching — at speed. The defect: **26 staggered stalls of an oddly uniform 18–21 s**
across the 10 workers (~15% idle each), and the bracketing lines show `#queue-req: 0` before
each gap and **110–140 requests arriving in one burst** after it — workers idled while ~2 k
requests queued upstream. Evidence:
`dsr1-sweep/results-summary/mx-prefill-starvation-evidence.txt`.

**Root cause (working).** Request delivery from the frontend/router layer is periodic-bursty
on the NATS path rather than streaming — consistent with case 5's observation that NATS shifts
queueing upstream (m4n: TTFT median +43% under NATS at mid_curve). Median-waiting requests eat
the burst period; the p99 doesn't change because worst-case requests span a cycle either way.

**Fix & status: the pre-registered A/B FALSIFIED the plane hypothesis** — and that is the
value of pre-registering. `mxt` (identical run, TCP): TTFT median 17.0 → 15.8 s (predicted
~10 s), throughput −9.0% → −7.6%, TPOT unchanged, and the stalls *persisted* (28 gaps of the
same 19–21 s character vs NATS's 26). The burstiness is plane-independent and unique to the
10-way prefill fan-out (6-way mid_curve shows none under TCP). Remaining suspects: router
dispatch cadence at high worker fan-out, k8s Service LB vs their nginx, and bench-client
connection handling. Follow-up decode-side analysis sharpened the mechanism: only **~30 of the offered 64
requests/rank actually decode**; the rest sit in the bootstrap/prefill pipeline (NATS: KV
92-96% full with 42% pre-allocated for transfers awaiting prefill; TCP: same halved batch with
longer prefill queues instead). The point is **residency-bound**: throughput = conc ÷ E2E
residency (Little's law: ours 2048/33.3 s ≈ 61 req/s ceiling vs their 2048/30.7 ≈ 67), so the
~5-6 s median-TTFT excess from wave-like admission IS the −7.6%. Status: OPEN — next probes
are admission smoothing and a KV-headroom diagnostic; all evidence archived.

## Case 7 — When the harvest is the bottleneck: measurement ops

Not a server gap, but it cost us one full p5 run and nearly cost p8's:

- **Result lost end-to-end (p5 first make-up attempt)**: extraction wrote to a gitignored
  directory that didn't exist on a fresh clone; GCS upload used the VM's scope-limited service
  account (403); teardown then deleted the only copies. Fixes: extraction *before* teardown
  with `mkdir -p`, uploads via ADC-minted tokens, and never editing a bash script that a
  runner is currently executing (bash reads lazily — v2 was created as a new file for exactly
  this reason).
- **Container log rotation ate a result (p8)**: the result JSON is a single ~35 MB line
  (per-request arrays); kubelet rotation kept only the tail. Recovery: sa-bench writes the
  aggregate stats *last*, so all latency aggregates survived in the tail; throughput was
  reconstructed from the decode logs' per-batch throughput timeline — steady-state plateau
  55.1 k tok/s vs their 54.4 k (**+1.4%**), measured-window ≈ −2.5%. (An earlier Little's-law
  estimate of −10% was calibration error — its edge factor came from the config-different
  drift run.) Fix: every bench now prints a compact scalar-only `=== SLIM-RESULT` line
  immediately before its final marker — the log tail always survives rotation.
- **Every record run now self-certifies**: hard serving gate (3 consecutive real completions
  through the Service — registration counts race), JIT/stall assertion (case 4), transport
  evidence in-log (case 2/5), and the request-plane line grepped into the run summary.

## The distilled playbook

1. **Match the ruler before touching the machine**: warmup and measured-prompt multipliers
   change reported throughput by ~10% with zero server change.
2. **Verbatim is a diff, not a feeling**: mechanically compare generated configs against
   source recipes — including what the reference *runner injects* beyond the recipe file.
3. **Warm state is config**: persist JIT/kernel caches; fill them with a discarded
   full-pressure pass; assert the record window was compile-free and stall-free.
4. **Latency identities before latency theories**: E2E = TTFT + TPOT×(N−1) at fixed
   concurrency — many "TPOT gaps" are TTFT policy mirrored through this identity.
5. **One-variable A/Bs with pre-registered predictions** (m4n, mxt): write the expected
   direction down before the run so the conclusion can't be fitted afterward.
6. **Evidence files or it didn't happen**: every root cause above has a verbatim log extract
   archived next to the results it explains.
7. **Treat the harvest path as production**: extract locally before teardown, rotation-proof
   the logs, and validate external data (their "(ms)" columns were seconds; caught by an
   internal identity check).

## Case 8 — KV-over-RDMA: a three-stage investigation (2026-08-05)

**Setup.** Cluster RDMA became functional (388 Gb/s verbs between DRA-claimed pods), inviting
the question: does moving the NIXL KV path from MNNVL to GPUDirect RDMA close the remaining
DSR1 gaps (mid-2048 −6.0%, max_tpt −9.0%)?

**Stage 1 — naive env swap fails invisibly.** `UCX_TLS=cuda_copy,rc_x,tcp` + mrdma claims ran
"successfully" at 6x lower throughput: proto tables showed all lanes on tcp with *software
emulation* for CUDA — rc never engaged (UCX ifindex errors on host-side netdev names).
*Lesson: a transport experiment without transport evidence is not an experiment.*

**Stage 2 — the fix was in the repo all along.** sysfs GID table: indexes 0-3 -> host-ns
netdevs, 4/5 -> the in-pod ipvlan. `UCX_IB_GID_INDEX=5` (canonical rev2, July) makes a 2-node
smoke run 100% rc_mlx5. *Lesson: 20-minute smoke before every 18-node run — the smoke-first
sequencing here saved a second invalid full-scale run.*

**Stage 3 — scale kills it on rail isolation.** Full-scale crash-looped with
NIXL_ERR_REMOTE_DISCONNECT; a socket matrix proved the 8 ipvlans live in 8 disjoint /64s with
no cross-rail routes. UCX multipath pairs rails obliviously -> unroutable connections at
fan-out. *Conclusion: RDMA-KV under DraNet needs rail-aware pairing in NIXL/UCX (NCCL does
this internally). MNNVL stays the production KV path; documented as future work.*

**Stage 4 — RESOLUTION (2026-08-06/07): `UCX_NET_DEVICES` closes the case.** The stage-3
"rail-aware pairing" conclusion was refined by a minutes-scale smoke ladder (rxr: Qwen3-4B
TP1 pairs with `CUDA_VISIBLE_DEVICES` GPU pinning) plus a mesh-warm burst harness (conc-512
rate-∞ salvo forcing every rank-pair to wire simultaneously). Forensics overturned two
beliefs: (a) no cuda KV ever rode TCP — the "tcp bulk" rows in failure logs are benign
host→host metadata present in healthy runs too (the transport guard was tightened to
cuda-context-only detection accordingly); (b) failed pairs *hang* (cuda rendezvous has no
fallback), which is what turned a scattered wire-up failure into 6-hour benches. The true
root cause: `no remote ep address for lane[k]` = **lane-index mismatch** — UCX derives lane
maps from device discovery order, which varies across DRA-injected pods. The fix, from the
official reference recipe: `UCX_NET_DEVICES=mlx5_0:1..mlx5_7:1` pins an identical ordered
device list on every worker. Controlled A/B on the same gate: rev3 = 5 wireup errors from
probe traffic; rev4 = clean through the full 24×48 mesh storm (1024/1024 requests in 100.6 s,
rc_mlx5=4663, zero errors). At benchmark scale (mult10, guard-audited 10,425 rc_mlx5 ops,
0 cuda_ipc): mid-512 **491.1 (+11.1% vs InferenceMax)**, mid-2048 **828.7 (−1.2%)** with
TPOT 13.75 ms — collapsing the +24–41% mid-curve TPOT gap of Case 5 to ~3% — mid-4096 830.8
(+8.5%). Mechanism: MNNVL KV handoffs contend with decode compute on the NVLink fabric;
RDMA offloads them to the NICs. RDMA-KV is now the *final* result at mid 512/2048 under the
optimal-run policy, and the per-network A/B lives in `benchmark-report-rdma-kv.md`.
Ops lessons added along the way: kill a bench client and the workers' stream clients may
panic (cycle workers after aborting a bench); session-scoped background orchestrators die
with the CLI (daemonize long runs; VM reboots still kill them — resume harvest-first);
guard every RDMA run with the transport gate + in-bench watchdog (stop policy).
