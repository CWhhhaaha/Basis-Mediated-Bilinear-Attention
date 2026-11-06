"""Controlled full-model prefill and long-cache decoding measurements.

The long-cache path uses preallocated synthetic explicit K/V tensors and excludes
cache/model initialization from timing. It does not perform a 1M-token prefill.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from bmb.attention import fa4_backend_info
from bmb.llama_bmb import BMBConfig, BMBForCausalLM


def timed_call(fn, warmup: int, steps: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / steps


def allocate_cache(model: BMBForCausalLM, length: int, extra: int, device):
    config = model.config
    cache = []
    for layer in model.layers:
        score_dim = layer.attn.score_dim
        shape_k = (1, config.n_kv_head, length + extra, score_dim)
        shape_v = (1, config.n_kv_head, length + extra, config.value_dim)
        cache.append({
            "k": torch.randn(shape_k, device=device, dtype=torch.bfloat16),
            "v": torch.randn(shape_v, device=device, dtype=torch.bfloat16),
            "length": length,
        })
    return cache


def logical_cache_gib(config: BMBConfig, length: int):
    score_dim = config.factor_rank if config.attention_type == "bmbuv" else config.head_dim
    elements = config.n_layer * config.n_kv_head * length * (score_dim + config.value_dim)
    return elements * 2 / 2**30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefill-length", type=int, default=16384)
    parser.add_argument("--cache-length", type=int, default=1000000)
    parser.add_argument("--decode-steps", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--skip-prefill", action="store_true")
    parser.add_argument("--skip-long-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--backend",
        choices=("sdpa", "fa4"),
        default="sdpa",
        help="Attention implementation; FA4 never falls back to SDPA",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This systems benchmark requires a CUDA GPU")

    raw = json.loads(Path(args.config).read_text())
    backend_info = (
        fa4_backend_info()
        if args.backend == "fa4"
        else {"package": "torch", "version": torch.__version__}
    )
    config = BMBConfig(**raw["model"])
    config.attention_backend = args.backend
    if args.checkpoint:
        model = BMBForCausalLM.from_pretrained(args.checkpoint, config=config)
    else:
        model = BMBForCausalLM(config)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    result = {
        "config": raw,
        "checkpoint": args.checkpoint,
        "device": torch.cuda.get_device_name(),
        "dtype": "bfloat16",
        "batch_size": 1,
        "logical_kv_at_cache_gib": logical_cache_gib(config, args.cache_length),
        "protocol": "synthetic static explicit-K/V cache; initialization excluded",
        "attention_backend": args.backend,
        "attention_interface": (
            "FlashAttention-4: flash_attn.cute.flash_attn_func"
            if args.backend == "fa4"
            else "PyTorch SDPA; backend selected by PyTorch"
        ),
        "backend_package": backend_info,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_capability": list(torch.cuda.get_device_capability()),
    }

    with torch.inference_mode():
        if not args.skip_prefill:
            inputs = torch.randint(0, config.vocab_size, (1, args.prefill_length), device=device)
            def prefill():
                model(inputs, use_cache=True, logits_to_keep=1)
            seconds = timed_call(prefill, args.warmup, max(1, args.decode_steps // 16))
            result["prefill_tokens_per_second"] = args.prefill_length / seconds
            past = model(inputs, use_cache=True, logits_to_keep=1).past_key_values
            short_cache = allocate_cache(
                model, args.prefill_length, args.decode_steps + args.warmup, device
            )
            for static, dynamic in zip(short_cache, past):
                static["k"][:, :, :args.prefill_length].copy_(dynamic[0])
                static["v"][:, :, :args.prefill_length].copy_(dynamic[1])

            def short_decode():
                for static in short_cache:
                    static["length"] = args.prefill_length
                model(token, past_key_values=short_cache, use_cache=True, logits_to_keep=1)

            token = torch.zeros((1, 1), dtype=torch.long, device=device)
            seconds = timed_call(short_decode, args.warmup, args.decode_steps)
            result["short_cache_decode_ms_per_token"] = seconds * 1000
            del inputs, past, short_cache

        if not args.skip_long_cache:
            torch.cuda.empty_cache()
            cache = allocate_cache(model, args.cache_length, args.decode_steps + args.warmup, device)
            token = torch.zeros((1, 1), dtype=torch.long, device=device)

            def long_decode():
                for static in cache:
                    static["length"] = args.cache_length
                model(token, past_key_values=cache, use_cache=True, logits_to_keep=1)

            torch.cuda.reset_peak_memory_stats()
            seconds = timed_call(long_decode, args.warmup, args.decode_steps)
            result["long_cache_decode_ms_per_token"] = seconds * 1000
            result["peak_allocation_gib"] = torch.cuda.max_memory_allocated() / 2**30

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
