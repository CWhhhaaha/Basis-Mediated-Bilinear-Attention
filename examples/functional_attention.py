"""FlashAttention-style functional call with explicit tensor layouts."""

import torch

from bmb import BMBAttention, bmb_uv_attention


def main():
    torch.manual_seed(0)
    d_model, heads, kv_heads, rank, width, value_dim = 256, 8, 2, 64, 32, 64
    parameters = BMBAttention(
        d_model=d_model,
        n_heads=heads,
        n_kv_heads=kv_heads,
        latent_rank=rank,
        factor_rank=width,
        value_dim=value_dim,
    ).eval()
    x = torch.randn(2, 32, d_model)

    with torch.inference_mode():
        z = parameters.basis(x)  # [B,T,r]
        value = parameters.v_proj(x).view(2, 32, kv_heads, value_dim)
        cos_sin = parameters.rotary(x, position_offset=0, length=32)

        # Functional output is [B,T,H,d_v]; the caller owns the output projection.
        y, cache = bmb_uv_attention(
            z,
            parameters.q_factor,
            parameters.k_factor,
            value,
            cos_sin=cos_sin,
            use_cache=True,
        )
        output = parameters.o_proj(y.reshape(2, 32, heads * value_dim))

    print(f"functional output: {tuple(y.shape)}")
    print(f"projected output: {tuple(output.shape)}")
    print(f"K cache: {tuple(cache[0].shape)}")
    print(f"V cache: {tuple(cache[1].shape)}")


if __name__ == "__main__":
    main()
