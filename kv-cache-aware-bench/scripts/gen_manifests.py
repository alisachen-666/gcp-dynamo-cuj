"""Generate benchmark arm manifests from the NVIDIA kimi-k2.5 recipe.

Transforms applied to every arm:
  - pin runtime image tag (1.3.1)
  - strip Eagle3 speculative_config blocks (no speculative decoding)
  - strip host-memory KV offload (host_cache_size / secondary_offload_min_priority)
  - max_seq_len 200000 -> 262144 (trace p99 input is 252k)
  - add tolerations for the GPU node taints (nvidia.com/gpu, kubernetes.io/arch)
  - point compute-domain claim template at kv-bench-compute-domain-channel
  - per-arm resource renames and router-mode/flag overrides
"""
import re
from pathlib import Path

RECIPE = Path.home() / "kv-cache-aware-bench/recipes/recipes/kimi-k2.5/trtllm"
OUT = Path.home() / "kv-cache-aware-bench/manifests"
IMAGE_TAG = "1.3.1"

TOLERATIONS = """tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
  - key: kubernetes.io/arch
    operator: Equal
    value: arm64
    effect: NoSchedule"""

STRATEGY_C_STEP1 = [
    ("--router-prefill-load-scale", "2.0"),
    ("--router-queue-policy", "fcfs"),
    ("--router-temperature", "0.0"),
    ("--router-kv-overlap-score-credit", "1.0"),  # sweep 0.7-1.0 at run time
]

# ---------------------------------------------------------------------------
# Round-1 GB300-native engine configs (2026-08-09): derived from the
# aiconfigurator SILICON warm solve (prefix 133k, ISL 137k, gb300, 24 GPUs).
# Topology: P:D = 1:5 single-node workers (aic pareto conc-20/60 shape, sized
# for our conc-32 bench). Replaces the recipe's GB200-tuned engine yamls,
# which stalled decode (S4: PyExecutor hang at min 21). Deviations from raw
# aic output are commented inline.
AIC_PREFILL_YAML = """\
    backend: pytorch
    moe_expert_parallel_size: 4
    moe_tensor_parallel_size: 1
    moe_config:
      backend: TRTLLM  # aic said WIDEEP; its runner API (1.3.0rc14) crashes on runtime 1.3.1 (run_moe arg mismatch)
    tensor_parallel_size: 4
    pipeline_parallel_size: 1
    enable_attention_dp: true
    # aic: false (one-shot 137k prefill); we enable it because trace p99 input
    # is 252k > max_num_tokens — tail requests need chunking
    enable_chunked_prefill: true
    max_batch_size: 1
    max_num_tokens: 137504
    max_seq_len: 262144
    trust_remote_code: true
    kv_cache_config:
      free_gpu_memory_fraction: 0.8
      dtype: fp8
      tokens_per_block: 32
      enable_block_reuse: true
    cache_transceiver_config:
      backend: DEFAULT
      max_tokens_in_buffer: 138624
    disable_overlap_scheduler: true
    print_iter_log: false
"""
AIC_DECODE_YAML = """\
    backend: pytorch
    tensor_parallel_size: 4
    pipeline_parallel_size: 1
    enable_attention_dp: true
    moe_tensor_parallel_size: 4
    moe_expert_parallel_size: 1
    moe_config:
      backend: CUTLASS  # aic said WIDEEP; incompatible with runtime 1.3.1 — see prefill note
    enable_chunked_prefill: false
    # per-rank cap; global 16/worker = headroom over conc32/5 workers
    max_batch_size: 4
    max_num_tokens: 64
    max_seq_len: 262144
    trust_remote_code: true
    kv_cache_config:
      free_gpu_memory_fraction: 0.8
      dtype: fp8
      tokens_per_block: 32
      enable_block_reuse: true
    cache_transceiver_config:
      backend: DEFAULT
      max_tokens_in_buffer: 138624
    cuda_graph_config:
      enable_padding: true
      batch_sizes:
      - 1
      - 2
      - 3
      - 4
      - 8
    print_iter_log: false
"""


def gb300_disagg(text):
    """Swap recipe GB200 engine configs for aic GB300 ones; P:D 3:3 -> 1:5."""
    pre, sep, post = text.partition("\n    decode:")
    pre = pre.replace("replicas: 3", "replicas: 1")     # prefill service
    post = post.replace("replicas: 3", "replicas: 5", 1)  # decode service
    text = pre + sep + post
    text = re.sub(r"  prefill\.yaml: \|\n(?:    .*\n?|\n)*?(?=  decode\.yaml)",
                  "  prefill.yaml: |\n" + AIC_PREFILL_YAML, text)
    text = re.sub(r"  decode\.yaml: \|\n(?:    .*\n?|\n)*$",
                  "  decode.yaml: |\n" + AIC_DECODE_YAML, text)
    return text


def strip_speculative(text):
    # drop each speculative_config: block (mapping key + its indented children)
    return re.sub(
        r"\n( *)speculative_config:\n(?:\1 +.*\n?)*", "\n", text
    )


def add_tolerations(text):
    # replace each commented tolerations hint with a real tolerations block
    def repl(m):
        indent = m.group(1)
        block = "\n".join(
            indent + line for line in TOLERATIONS.splitlines()
        )
        return block
    return re.sub(r"( *)# tolerations:.*", repl, text)


def base_transforms(text, keep_eagle=False):
    text = text.replace("<IMAGE_TAG>", IMAGE_TAG)
    # weights come from the public bucket mirror staged in alisachen's GCS registry;
    # model identity is alisachen/Kimi-K2.5-NVFP4 end-to-end
    text = text.replace(
        "- --model-path\n            - nvidia/Kimi-K2.5-NVFP4",
        "- --model-path\n            - /model-cache/alisachen/Kimi-K2.5-NVFP4",
    )
    text = text.replace(
        "- --served-model-name\n            - nvidia/Kimi-K2.5-NVFP4",
        "- --served-model-name\n            - alisachen/Kimi-K2.5-NVFP4",
    )
    if keep_eagle:
        # recipe-direct arms (1A): Eagle3 retained as shipped; draft weights must
        # be staged at this path (gated on HF — mirror like the base model)
        text = text.replace(
            "speculative_model_dir: nvidia/Kimi-K2.5-Thinking-Eagle3",
            "speculative_model_dir: /model-cache/alisachen/Kimi-K2.5-Thinking-Eagle3",
        )
    else:
        text = strip_speculative(text)
        text = text.replace("with Eagle3 speculative decoding and", "with")
        text = text.replace("Eagle3 speculative decoding, ", "")
    # no host-memory KV offload: drop the config keys and the comment block about them
    text = re.sub(r" *host_cache_size:.*\n", "", text)
    text = re.sub(r" *secondary_offload_min_priority:.*\n", "", text)
    text = text.replace(",\n# and host-memory KV offloading on prefill workers.", ".")
    text = re.sub(r"#\n# Prefill workers offload spilled KV blocks to host memory[^\n]*\n#[^\n]*\n", "", text)
    text = text.replace("max_seq_len: 200000", "max_seq_len: 262144")
    # first-time weight load via gcsfuse can exceed 1h; probe budget 60->240 min
    text = text.replace("failureThreshold: 60", "failureThreshold: 240")
    text = add_tolerations(text)
    text = text.replace(
        "resourceClaimTemplateName: your-compute-domain-channel",
        "resourceClaimTemplateName: kv-bench-compute-domain-channel",
    )
    # A4X MAX: claim all 8 RDMA NICs per GPU pod (gpu-recipes@fef5ad27 pattern)
    text = text.replace(
        "resourceClaims:\n          - name: compute-domain-channel\n"
        "            resourceClaimTemplateName: kv-bench-compute-domain-channel",
        "resourceClaims:\n          - name: compute-domain-channel\n"
        "            resourceClaimTemplateName: kv-bench-compute-domain-channel\n"
        "          - name: rdma\n"
        "            resourceClaimTemplateName: mrdma-all",
    )
    text = text.replace(
        "claims:\n          - name: compute-domain-channel",
        "claims:\n          - name: compute-domain-channel\n          - name: rdma",
    )
    return text


def rename(text, old, new):
    return text.replace(old, new)


def set_router_mode(text, mode):
    return re.sub(
        r"(- --router-mode\n\s+- )\S+", r"\g<1>" + mode,
        re.sub(r"(- kv|- round-robin)(\n\s+- --request-plane)", r"- " + mode + r"\g<2>", text),
    ) if False else re.sub(
        r"(--router-mode\n( +)- )(kv|round-robin)", r"\g<1>" + mode, text
    )


def add_router_flags(text, flags):
    # insert extra frontend args right after the router-mode value
    m = re.search(r"( +)- --router-mode\n\1- \S+", text)
    if not m:
        raise SystemExit("router-mode arg not found")
    indent = m.group(1)
    extra = "".join(
        f"\n{indent}- {k}\n{indent}- \"{v}\"" for k, v in flags
    )
    return text[: m.end()] + extra + text[m.end():]


OUT.mkdir(exist_ok=True)

agg_rr = base_transforms((RECIPE / "agg-round-robin/deploy.yaml").read_text())
agg_kv = base_transforms((RECIPE / "agg-eagle-kv-router/deploy.yaml").read_text())
disagg_kv = base_transforms((RECIPE / "disagg-eagle-kv-router/deploy.yaml").read_text())
# recipe-direct 1A arms: NVDA config as shipped incl. Eagle3 (only env adaptations)
agg_eagle_kv = base_transforms(
    (RECIPE / "agg-eagle-kv-router/deploy.yaml").read_text(), keep_eagle=True)
agg_eagle_rr = base_transforms(
    (RECIPE / "agg-eagle-round-robin/deploy.yaml").read_text(), keep_eagle=True)

arms = {
    # Phase 1 (24-GPU aggregated): 3 workers x TP8/EP8 (2 nodes each)
    # 1A: NVDA recipe direct (Eagle3 ON) — both routing modes as shipped
    "arm1a-agg-eagle-rr.yaml": agg_eagle_rr,
    "arm1a-agg-eagle-kv.yaml": agg_eagle_kv,
    "arm1b-agg-rr.yaml": agg_rr,
    "arm1c-agg-kv-nvda.yaml": rename(agg_kv, "kimi-k25-agg-kv-eagle", "kimi-k25-agg-kv-nvda"),
    "arm1d-agg-kv-tuned.yaml": add_router_flags(
        rename(agg_kv, "kimi-k25-agg-kv-eagle", "kimi-k25-agg-kv-tuned"), STRATEGY_C_STEP1
    ),
    # Phase 2 round-1 (24-GPU disaggregated, GB300-native aic configs):
    # 1 prefill + 5 decode, single-node 4-GPU workers
    "arm2b-disagg-rr.yaml": gb300_disagg(set_router_mode(
        rename(disagg_kv, "kimi-k25-disagg-kv-eagle", "kimi-k25-disagg-rr"), "round-robin"
    )),
    "arm2c-disagg-kv-nvda.yaml": gb300_disagg(
        rename(disagg_kv, "kimi-k25-disagg-kv-eagle", "kimi-k25-disagg-kv-nvda")),
    "arm2d-disagg-kv-tuned.yaml": gb300_disagg(add_router_flags(
        rename(disagg_kv, "kimi-k25-disagg-kv-eagle", "kimi-k25-disagg-kv-tuned"), STRATEGY_C_STEP1
    )),
}

for name, text in arms.items():
    if "arm1a-" in name:
        assert "/model-cache/alisachen/Kimi-K2.5-Thinking-Eagle3" in text, \
            f"{name}: eagle draft path not repointed"
    else:
        assert "speculative_config" not in text and "speculative_model_dir" not in text, \
            f"{name}: eagle not fully stripped"
    assert "host_cache_size" not in text and "secondary_offload" not in text, \
        f"{name}: KV offload not fully stripped"
    assert "<IMAGE_TAG>" not in text, f"{name}: image tag placeholder left"
    (OUT / name).write_text(text)
    print(f"wrote {name}: {len(text.splitlines())} lines, "
          f"router={'round-robin' if 'round-robin' in text else 'kv'}, "
          f"eagle={'ON' if 'speculative_config' in text else 'off'}")
