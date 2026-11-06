"""Minimal dynamic/static-cache example; requires only PyTorch and bmb."""

import torch

from bmb import BMBAttention


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    attention = BMBAttention(
        d_model=256,
        n_heads=8,
        n_kv_heads=2,
        latent_rank=64,
        factor_rank=32,
        value_dim=64,
    ).to(device=device, dtype=dtype).eval()
    x = torch.randn(2, 16, 256, device=device, dtype=dtype)

    with torch.inference_mode():
        # Ordinary training/prefill call: [B,T,d_model] -> [B,T,d_model].
        full, _ = attention(x)

        # Dynamic cache: convenient for generation and debugging.
        prefix, cache = attention(x[:, :12], use_cache=True)
        decoded, cache = attention(
            x[:, 12:13], past_key_value=cache, use_cache=True
        )
        dynamic_error = (decoded - full[:, 12:13]).abs().max().item()

        # Static cache: preallocate once per layer for bounded-memory inference.
        static = attention.create_static_cache(
            batch_size=2, max_length=16, device=device, dtype=dtype
        )
        attention(x[:, :12], past_key_value=static, use_cache=True)
        static_decoded, static = attention(
            x[:, 12:13], past_key_value=static, use_cache=True
        )
        static_error = (static_decoded - full[:, 12:13]).abs().max().item()

    print(f"output: {tuple(full.shape)}")
    print(f"dynamic K cache: {tuple(cache[0].shape)}")
    print(f"static cache length: {static['length']}")
    print(f"full vs dynamic max abs error: {dynamic_error:.6g}")
    print(f"full vs static max abs error: {static_error:.6g}")


if __name__ == "__main__":
    main()
