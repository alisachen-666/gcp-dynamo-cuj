# DeepSeek-V4-Pro FP4 GB300 Sweep — Benchmarking Plan (GKE)

Goal: reproduce the InferenceX **`dsv4-fp4-gb300-dynamo-sglang-mtp`** data points on GKE A4X Max
(`REDACTED-GKE-CLUSTER-OLD`, np-2 NVL72 domain), following config, benchmark tool, topology, and images
**strictly**.

## Pinned sources (copies in `reference/`)

- InferenceX matrix entry: SemiAnalysisAI/InferenceX @ `5885a434`,
  `.github/configs/nvidia-master.yaml` L10920 (`inferencex-dsv4-fp4-gb300-dynamo-sglang-mtp.yaml`)
- Recipes: NVIDIA/srt-slurm `recipes/dsv4-pro/sglang/gb300-fp4/8k1k/disagg/` (8 files vendored;
  note InferenceX references them via `recipes/sglang/deepseek-v4/8k1k/` — same content, the
  runner maps the path)

## STRICT parity requirements

| Aspect | Value (do not substitute) |
|---|---|
| Image | ~~`lmsysorg/sglang:nightly-dev-20260527-14f81a67`~~ **`nightly-dev-20260601-373cadc9`** (original pin DELETED from Docker Hub — see Error log; nearest surviving nightly, forced deviation) |
| Model | **`deepseek-ai/DeepSeek-V4-Pro`**, precision **mxfp4**, loaded from a shared path (`/model/` on slurm -> gcsfuse/local mirror on GKE) |
| Dynamo | installed **from git hash `81d0555ee23519cea80a42b4fe824e30368b7300`** (`dynamo.install: true` in recipes — NOT a PyPI release), request plane **NATS** |
| Spec decode (MTP) | EAGLE: `speculative-algo EAGLE`, `speculative-num-steps 3`, `speculative-eagle-topk 1`, `speculative-num-draft-tokens 4` (decode side, per recipe) |
| Extra dsv4 flags | `tool-call-parser: deepseekv4`, env `SGLANG_DEFAULT_THINKING=1`, `SGLANG_DSV4_REASONING_EFFORT=max`, DeepGEMM mega-MoE opts (per recipe env blocks — take each file verbatim) |
| Benchmark | **sa-bench** (already vendored in `../common/sa-bench/`), ISL 8192 / OSL 1024, `req_rate=inf`, concurrencies per point (below) |
| Transport | NVLink everywhere: `MC_FORCE_MNNVL=1`, `NCCL_MNNVL_ENABLE=1`, `NCCL_CUMEM_ENABLE=1` (in recipes) |

## Sweep matrix (8 points; ALL fit inside our 18-node np-2 domain)

| # | Point (recipe file) | Prefill | Decode | Nodes | Conc (sa-bench) |
|---|---|---|---|---|---|
| 1 | `disagg-low-latency-1p1d-tp4-tp4-mtp` | 1w TP4 EP1 | 1w TP4 EP1 | 2 | 1 |
| 2 | `disagg-low-latency-1p6d-dep4-tp4-mtp` | 1w DEP4 (dp-attn) | 6w TP4 | 7 | 8x32x64 |
| 3 | `disagg-mid-curve-1p1d-dep4-dep8-mtp` | 1w DEP4 | 1w DEP8 (TP8/EP8, dp-attn) | 3 | 256 |
| 4 | `disagg-mid-curve-1p1d-dep4-dep16-mtp` | 1w DEP4 | 1w DEP16 (TP16/EP16) | 5 | 256 |
| 5 | `disagg-mid-curve-2p1d-dep4-dep8-mtp` | 2w DEP4 | 1w DEP8 | 4 | 512 |
| 6 | `disagg-mid-curve-4p1d-dep4-dep8-mtp` | 4w DEP4 | 1w DEP8 | 6 | 1024 |
| 7 | `disagg-high-conc-6p1d-dep4-dep8-mtp` | 6w DEP4 | 1w DEP8 | 8 | 4096 |
| 8 | `disagg-high-conc-8p1d-dep4-dep8-mtp` | 8w DEP4 | 1w DEP8 | 10 | 8192 |

All decode DEP points use dp-attention + DeepEP (wide-EP); low-latency points are plain TP.
Server flags per point: take the recipe file's `sglang_config` prefill/decode sections VERBATIM
(they differ per point — mem fractions, chunked-prefill, cuda-graph-bs, deepep modes).

## GKE translation (apply every M2/DSR1 lesson)

1. **Operator-less Dynamo**: plain Deployments per worker + `python3 -m dynamo.frontend`;
   etcd + NATS from the chart install in `dynamo-cloud` (reuse). No DGD CRs (RBAC-blocked
   anyway), matching srt-slurm structure.
2. **Image**: the pinned nightly must be verified on arm64/GB300 first (nightlies are
   multi-arch but unproven here) — smoke-0-style sanity job before anything else.
   **Bake dynamo@`81d0555` into a derived image** pushed to Artifact Registry (source build —
   Rust toolchain; do NOT pip-install per pod: the hash build is too heavy for startup).
3. **Model staging**: download `deepseek-ai/DeepSeek-V4-Pro` ONCE -> stage to
   `gs://alisachen-models/deepseek-ai/DeepSeek-V4-Pro/` (reuse `../common/stage-model-to-gcs.sh`
   pattern); workers read from hostPath mirror or gcsfuse. NEVER per-pod HF downloads
   (429 + dynamo fetch_llm dies on first 429). CHECK FIRST: HF availability/gating + size of
   V4-Pro (likely ~400+ GB at fp4; EAGLE draft weights may be a separate artifact — check
   recipe `speculative-draft-model-path` / model repo).
4. **Worker args**: always `--host 0.0.0.0` (bootstrap server must bind pod IP — M2 lesson);
   `--model-path` = local mirror path; pass recipe flags verbatim.
5. **Frontend**: needs the model's small files at the SAME path as workers (discovery resolves
   the worker-registered path; `ignore_weights=true`) — mirror config/tokenizer via
   snapshot_download allow_patterns, or gcsfuse-mount the same path read-only.
6. **ComputeDomain**: one CR per deployment sized to its node count (e.g. point 8: numNodes=10);
   every worker pod carries the channel claim. NO mrdma claim (cluster RDMA broken; NVLink-only
   matches the recipes' MC_FORCE_MNNVL anyway).
7. **UCX env**: defaults + `UCX_CUDA_IPC_ENABLE_MNNVL=y`, `UCX_TLS=cuda_copy,cuda_ipc,tcp`
   (tested), `UCX_PROTO_INFO=y` evidence. NEVER `UCX_NET_DEVICES=mlx5...` here (crashes NIXL).
8. **Multi-node workers** (DEP8/DEP16 decode spans 2-4 nodes): Indexed pattern with
   `--dist-init-addr <leader>:5757 --nnodes N --node-rank $JOB_COMPLETION_INDEX` (srt-slurm
   passes the same); leader DNS via headless service; all pods of one worker share the
   ComputeDomain.
9. **Cache**: hostPath `/mnt/stateful_partition/kube-ephemeral-ssd/dsv4-cache` (11.6T local SSD;
   NOT /var/tmp — 94G boot disk) + node pinning for cheap iteration.
10. **Transport evidence per point**: UCX_PROTO_INFO logs (bulk KV on cuda_ipc), NVSHMEM_DEBUG=INFO
    on first DEP point (DeepEP over MNNVL), archive with results.

## Execution phases

- **P0 image/env validation** (1 node): sanity job on the pinned nightly — imports, GB300
  kernels, dynamo@hash import, NIXL agent creation. Gate for everything else.
- **P1 model staging**: V4-Pro -> GCS + node mirrors. Verify EAGLE/MTP draft assets.
- **P2 point 1** (2 nodes, conc 1): smallest e2e — proves dsv4 + MTP + disagg on GKE.
- **P3 points 3,5,6** (3-6 nodes): mid-curve family — shares decode DEP8 shape, iterate fast.
- **P4 points 2,4** (7,5 nodes), **P5 points 7,8** (8,10 nodes): remaining coverage.
- Points can run CONCURRENTLY when node budget allows (e.g. point 1 + point 3 = 5 nodes);
  but for RECORD runs, run one point at a time on an otherwise-idle domain (no interference).
- Each point: warmup per sa-bench convention, then measured run ->
  `results/dsv4/<point>/results_concurrency_<c>_gpus_<g>.json` + transport-evidence log slice.

## Open items / risks — status (2026-07-30)

- [x] `deepseek-ai/DeepSeek-V4-Pro`: PUBLIC, not gated, 64 safetensors shards; NO separate
      MTP/draft files -> EAGLE weights are in-checkpoint (like DSR1 NextN). Staging to
      `gs://alisachen-models/deepseek-ai/DeepSeek-V4-Pro/` in progress (dsv4-model-stage job).
- [~] nightly image arm64: pull + python/sglang import validated CPU-side in dsv4-dynamo-build
      job; GPU kernel validation pending first GPU point.
- [~] dynamo@`81d0555` aarch64 build: in progress (dsv4-dynamo-build job: rustup + maturin
      build of lib/bindings/python + ai-dynamo wheel -> `gs://alisachen-models/dynamo-wheels/81d0555/`).
      Distribution to pods: seed wheels to node hostPath (same pattern as model seeding);
      pods `pip install /wheels/*.whl` (no in-pod git builds).
- [ ] 0.8.1 etcd/NATS chart services vs dynamo@hash workers (recipes force NATS request
      plane pre-39d2a68 -> expected compatible; verify at point 1)
- [ ] node budget: DSR1 mid_curve occupies all 18 np-2 nodes until its sweep completes;
      DSv4 GPU phases start after teardown.

## KEY RECIPE FACTS extracted (point 1, applies broadly)

- Transfer backend is **mooncake** (NOT nixl as in DSR1) -> `MC_FORCE_MNNVL=1` directly
  governs the KV path; verify mooncake-transfer-engine exists in the nightly image.
- Bench needs `use_chat_template: true` + custom tokenizer
  `sa_bench_tokenizers.sglang_deepseek_v4.SGLangDeepseekV4Tokenizer` -> the bench ConfigMap
  must include the vendored `sa_bench_tokenizers/` package (flat configmap is NOT enough).
- Frontends: 1+8 for every point (even 2-node).
- Decode-only EAGLE MTP: steps 3, topk 1, draft 4.
- Per-goal directive: use MORE warmup — warmup 3x conc @ rate 250 + a long-run warm
  (extra measured-length pass discarded) before the recorded run.

## Error log (running; per goal directive)

- **[FIXED 2026-07-30] CONFIG PARITY DRIFT in generated points p5-p8 (self-inflicted).**
  `gen-point.py` derived p4-p8 from p3 but only parameterized TP/DP/EP, mem-fraction,
  max-running-requests and cuda-graph-max-bs. Recipe diffs showed the following were silently
  inherited from p3 instead of taken from each recipe:
    * decode `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` (p5-p8 want **4096**, p3=2048)
    * p6 prefill max-running-requests / cuda-graph-max-bs (want 1024, ran 512); decode cgbs 1024
    * p7 decode **speculative-num-steps 2 / draft-tokens 3** (ran 3/4) and context-length 9216 (ran 16384)
    * p8 decode **speculative-num-steps 1 / draft-tokens 2** (ran 3/4), mem-fraction 0.85 (ran 0.9),
      cuda-graph-max-bs 1280 (ran 1024), swa-full-tokens-ratio 0.1 (ran 0.15), context-length 9216
  IMPACT: p5/p6 mildly off; **p7/p8 materially off** — the recipes deliberately REDUCE MTP depth
  at high concurrency (spec decode loses value as batch grows and wastes compute), and shrink
  context-length to free KV. Results from the first p5-p8 runs are labelled `_drift` and
  superseded by parity reruns.
  FIX: `gen-point.py` extended with dtokrank/spsteps/spdraft/ctxlen/swaratio parameters; all
  points regenerated from recipe values and rerun.

- **[FIXED 2026-07-30] p5/p6 bench failed `Initial test run failed ... Error: Not Found`.**
  The runner waited for a fixed registration count (2) and for `/v1/models` to list the model,
  then started the bench. For multi-prefill-worker points the true count is
  `prefill_workers + 1` (p5=3, p6=5, p7=7, p8=9), so the bench began while workers were still
  registering -> frontend 404s on /v1/completions. FIX: replaced the count/model-list gate with
  a HARD GATE requiring 3 consecutive successful `/v1/completions` through the Service before
  benching. Points p5/p6 rerun after the fix.

- **[FIXED 2026-07-30] DSv4 bench: `python3: can't open file '/tmp/benchmark_serving.py'`.**
  Two distinct bugs: (a) bench pod lacked `nodeSelector: np-2` so its `/model` hostPath check
  failed (`hostPath type check failed ... is not a directory`) and it sat ContainerCreating 60
  min; (b) `cp -r /sa-bench/*.py /tmp/` copies ConfigMap SYMLINKS verbatim (`-> ..data/x.py`),
  which dangle outside the mount — cp exits 0, python then reports file-not-found.
  FIX: add np-2 nodeSelector to bench jobs + use `cp -L` (dereference) + post-copy `ls` guard.

- **[NOTED 2026-07-30] DSv4 worker first-boot is slow (~35-60 min)** on the nightly: mxfp4/
  DeepGEMM JIT kernel compilation for sm_103a at startup. Mitigation added from point 3 on:
  persist `/root/.cache/flashinfer` to node hostPath (`dsv4-jit-cache`) so each node compiles
  once across all sweep points.

- **[FIXED 2026-07-30] dynamo@hash build: bindgen "Unable to find libclang".**
  nixl-sys uses bindgen which requires libclang; nightly image ships none.
  FIX: `apt-get install clang libclang-dev` + `LIBCLANG_PATH=/usr/lib/llvm-18/lib`.
  (Ops note: avoid `pkill -f <pattern>` where the pattern matches the kubectl-exec'd shell's
  own command string — it kills the exec session, exit 143.)

- **[FIXED 2026-07-30] dynamo@hash build: nixl-sys cannot find libnixl.so.**
  The nightly image ships NIXL only as the pip `nixl_cu13` package
  (`dist-packages/.nixl_cu13.mesonpy.libs/libnixl.so`), but `nixl-sys v0.10.1`'s build script
  searches `/opt/nvidia/nvda_nixl/...` and its stub fallback also fails. FIX: symlink the pip
  package's libs into `/opt/nvidia/nvda_nixl/lib/aarch64-linux-gnu/` before `maturin build`.
  RUNTIME IMPLICATION: every DSv4 worker/frontend pod must create the same symlink layout
  before installing/running the wheels (added to manifests).

- **[FIXED 2026-07-30] dynamo@hash wheel build failures (2 attempts).**
  Attempt 1 (Job): silent — pods GC'd with logs. Lesson: build pods must tee to hostPath.
  Attempt 2: rustup installer fails inside the nightly image with `Invalid cross-device link
  (os error 18)` — the image ALREADY BUNDLES rustc/cargo 1.96.0 in /root/.cargo and rustup's
  cross-overlayfs rename of the preexisting toolchain is illegal. FIX: skip rustup entirely,
  build with the bundled toolchain (PATH=/root/.cargo/bin). Attempt 3 running with
  submodules + cmake + persistent /work/build3.log.

- **[FIXED 2026-07-30] Pinned nightly image DELETED from Docker Hub.**
  `lmsysorg/sglang:nightly-dev-20260527-14f81a67` returns 404 (nightly-dev tags are pruned);
  verified NO May-2026 nightlies survive. Nearest available: **`nightly-dev-20260601-373cadc9`**
  (+5 days of sglang master vs the pin; arm64 manifest present, 12.6 GB).
  DECISION: proceed with 20260601 as a FORCED image deviation — documented parity break.
  If point-1 results look anomalous vs InferenceX, fallback is building sglang @ commit
  `14f81a67` from source (exact code parity, multi-hour sm103 kernel build).


## Directory layout

- `PLAN.md` — this file
- `reference/` — pinned InferenceX block + 8 verbatim srt-slurm recipes
- `manifests/` — (to create) per-point GKE manifests, generated from recipes
- `results/` — per-point sa-bench JSONs + transport evidence

- **[ROOT-CAUSED 2026-08-01] p3's -8% throughput / +70% p99 TTFT vs InferenceX: DeepGEMM JIT
  compiles during the MEASURED window.** Server logs (`server-logs/dsv4-p3/`): prefill worker
  entered `TF32_HC_PRE` JIT sessions repeatedly (lazy per-shape), including at 12:07:08 — inside
  the measured window 12:02:13-12:08:19 — freezing the SINGLE prefill worker for 12.3s + 12.5s
  (~6.8% of the window; matches -8% on the prefill-bound 8k1k mix). A 36.7s stall ended 4s
  before measurement began. Cause: the `jit-cache` hostPath persists only
  `/root/.cache/flashinfer`; `/root/.cache/deep_gemm` was cold every pod. srt-slurm hosts
  accumulate this cache across the sweep. FIX: all point manifests + point3 template now also
  mount `dsv4-dg-cache` -> `/root/.cache/deep_gemm`. Expect p3's residual gap to close on rerun
  with a warm DG cache; the image deviation (+5 days) is now a secondary suspect at most.

## Record-run protocol v2 (adopted 2026-08-01, after the p3 JIT root-cause)

Runner: `run-point-v2.sh <pid> <expected_workers> [passes]` (v1 kept intact — never edit a
bash script while a run is in flight; bash reads scripts lazily).

1. **DeepGEMM cache persistence**: all manifests now mount
   `dsv4-dg-cache -> /root/.cache/deep_gemm` (in addition to the flashinfer mount). Compiles
   accumulate per node across pods, as on srt-slurm hosts.
2. **Double-bench for record-quality points** (`passes=2`): one full bench sequence is run and
   DISCARDED against the standing deployment (cache filler — forces every batch shape through
   the JIT once-per-process), then the record pass runs compile-free. Strictly stronger than
   longer warmup: shape coverage grows with total prompts seen, not warm minutes (p3 leaked a
   compile after 8x conc of warm traffic).
3. **JIT/stall assertion**: after the record pass, every measured window (from the result
   JSON's date+duration) is checked against every prefill pod log for "Entering DeepGEMM JIT"
   sessions and >5s gaps in `report_prefill_stats`. Verdict VALID/SUSPECT is printed; SUSPECT
   record runs must be re-benched (cheap — deployment still standing).
4. Uploads authenticate via ADC token (v1 used the VM's scope-limited compute SA and 403'd).

Queued 2026-08-01 (`queue-p8-p3.sh`, running): p5 rerun (in flight, v1, corrected config) ->
p8 rerun (v2, corrected config, 1 pass + assertion) -> p3 re-record (v2, 2 passes).
