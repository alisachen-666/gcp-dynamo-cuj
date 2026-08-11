# SGLang backend — Kimi-K2.5 on GB300 (KV-aware routing bench)

Second backend for the study, tuned with the same layered pipeline proven on
trtllm (see ../SWEEP_METHODOLOGY.md): Profiler → AIC → DynoSim → silicon,
SILICON database mode only, sims parallel with live work, every deviation
from generated configs documented inline.

## Pipeline mapping (trtllm → sglang)

| Layer | trtllm (done) | sglang (this dir) |
|---|---|---|
| AIC support check | PASS agg+disagg | `--backend sglang` — gb300/sglang perf DB exists (0.5.9/0.5.10 rows seen) |
| AIC solves | warm/cold × agg/disagg, top-5 configs | same brackets, same trace shape (ISL 137k/OSL 1.1k, prefix 133k) |
| DynoSim | rates from AIC + live recalibration | reuse simulator as-is (router formula is backend-agnostic); re-seed PREFILL_TOKRATE / TPOT curve from sglang AIC rows, recalibrate vs sglang smoke |
| Router | dynamo frontend flags (identical) | identical — dynamo router doesn't care about the engine |
| Manifests | dynamo.trtllm workers | dynamo sglang workers (`dynamo.sglang`); NIXL KV transfer for disagg |
| Smoke ladder | S0-S4 | reuse jobs with sglang stacks; S2/S3 transport proof via NIXL/UCX (DynamoBench kv-transport-guard) |
| Bench | AIPerf sliced trace, alisachen- jobs | identical harness, new endpoints |

## Assets already proven on this hardware family (DynamoBench)

- Image lineage: `lmsysorg/sglang:v0.5.8.post1-cu130-runtime` (pinned; smoke-passed
  on a4xmax incl. NVFP4 DeepSeek), `:kimi-k3` variant existed for Kimi-K3
- Dynamo+sglang disagg with NIXL KV transfer over RDMA: working manifests in
  ~/DynamoBench/dsr1-sweep/manifests (m4r/llr rdmakv variants), incl. the UCX
  rail pins we already carry and NIXL-specific env
- kv-transport-guard.sh + gen-rdma-report.py for the transport-evidence policy

## Open questions to resolve in order

1. AIC sglang solve output (running) — engine shapes/batching for GB300 sglang
2. Kimi-K2.5-NVFP4 support in a pinned sglang image (KimiK25ForConditionalGeneration
   + modelopt nvfp4 on sm_103); candidate images: lmsysorg/sglang latest pins,
   or the dynamo sglang-runtime image from nvcr (`ai-dynamo/sglang-runtime`)
3. dynamo.sglang worker flags parity (kv events, disagg mode, kv-block-size)
4. Whether sglang's radix cache + dynamo KV events emit block hashes compatible
   with our 64-token trace blocks (kv-block-size alignment)

## Status log

- 2026-08-11: dir scaffolded; AIC sglang support check + warm/cold solves launched.
