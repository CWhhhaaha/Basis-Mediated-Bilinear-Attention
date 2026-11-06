# BMB Attention API

`bmb.BMBAttention` is the user-facing GQA+BMB-UV attention sublayer. It depends only on PyTorch and is also used by the included FineWeb model.

## Installation

```bash
# Attention API
python -m pip install -e .

# Training and evaluation programs
python -m pip install -e '.[train,eval]'

# Optional FlashAttention-4 backend
python -m pip install -e '.[fa4]'
python -m pip install --pre 'flash-attn-4[cu13]'  # CUDA 13
```

## Module API

```python
import torch
from bmb import BMBAttention

attn = BMBAttention(
    d_model=768,
    n_heads=22,
    n_kv_heads=2,
    latent_rank=64,
    factor_rank=32,
    value_dim=64,
)

x = torch.randn(2, 128, 768)
y, _ = attn(x)  # [B,T,d_model]
```

The module includes the shared basis, per-query-head Q factors, per-KV-head K factors, V and output projections, RoPE, grouped attention, and cache handling. Normalization, residual connections, and the MLP remain in the surrounding block.

- `n_kv_heads=n_heads` gives an MHA-style KV-head count, `n_kv_heads=1` gives MQA, and an intermediate divisor gives GQA.
- `latent_rank` is $r$ and `factor_rank` is $s$.
- The default is causal attention. Use `is_causal=False` for bidirectional attention.
- `get_config()` and `BMBAttention.from_config(...)` support configuration serialization.

## Functional API

```python
from bmb import bmb_uv_attention

z = attn.basis(x)                       # [B,T,r]
v = attn.v_proj(x).view(2, 128, 2, 64) # [B,T,H_kv,d_v]
cos_sin = attn.rotary(x, 0, 128)

y, cache = bmb_uv_attention(
    z, attn.q_factor, attn.k_factor, v,
    cos_sin=cos_sin, use_cache=True,
)                                      # [B,T,H,d_v]
y = attn.o_proj(y.reshape(2, 128, 22 * 64))
```

Use `project_bmb_uv` and `grouped_attention` separately when the host model owns position transforms or cache management. See [integration.md](integration.md) and `examples/functional_attention.py`.

## Position encoding

Three modes are available:

1. **Built-in split-half RoPE:** pass optional `position_ids`; cached chunks infer contiguous positions from the cache length.
2. **External RoPE:** pass `cos_sin=(cos,sin)`, broadcastable to `[B,T,H,s]`.
3. **No internal rotation:** construct with `rope=False` and apply the host model's position transform through the lower-level API.

`factor_rank` must be even when built-in RoPE is enabled.

## Padding and custom masks

A 2D key-padding mask covers past and current positions. `True` or `1` marks valid keys:

```python
mask = torch.tensor([
    [0, 0, 1, 1, 1],
    [1, 1, 1, 1, 1],
])
positions = torch.tensor([
    [0, 0, 0, 1, 2],
    [0, 1, 2, 3, 4],
])
y, cache = attn(
    x[:, :5], attention_mask=mask, position_ids=positions, use_cache=True
)
```

Boolean and additive 4D masks use shape `[B,1,T,L]` or `[B,H,T,L]`, where `L` is past length plus current chunk length. For left-padded generation, extend the 2D mask for each new token and pass the corresponding position IDs.

## Dynamic cache and incremental decoding

```python
with torch.inference_mode():
    y_prefix, cache = attn(x[:, :127], use_cache=True)
    y_next, cache = attn(
        x[:, 127:128], past_key_value=cache, use_cache=True
    )
```

The returned `(K,V)` cache has shapes

$$
K\in\mathbb{R}^{B\times H_{kv}\times L\times s},\qquad
V\in\mathbb{R}^{B\times H_{kv}\times L\times d_v}.
$$

K already includes RoPE. Multi-token chunks use an offset-aware causal mask, so cached prefill and incremental decoding retain the correct causal alignment.

## Preallocated static cache

```python
cache = attn.create_static_cache(
    batch_size=2,
    max_length=2048,
    device=x.device,
    dtype=x.dtype,
)

with torch.inference_mode():
    attn(x, past_key_value=cache, use_cache=True)
    next_x = torch.randn(2, 1, 768)
    y_next, cache = attn(
        next_x, past_key_value=cache, use_cache=True
    )
```

Allocate one cache dictionary per layer. Static-cache writes are intended for inference; dynamic caches support autograd.

## Attention backends

### SDPA

`backend="sdpa"` is the default and uses PyTorch's selected SDPA implementation. It supports the mask and cache interfaces shown above.

### Reference

`backend="reference"` materializes attention scores and probabilities. It is useful for correctness checks and debugging.

### FlashAttention-4

```python
attn = BMBAttention(
    d_model=768, n_heads=22, n_kv_heads=2,
    latent_rank=64, factor_rank=32, value_dim=64,
    dropout=0.0, backend="fa4",
)
y, cache = attn(x.cuda().bfloat16(), use_cache=True)
```

The FA4 adapter retains the same projections, RoPE, explicit K/V cache, and output contract. It requires CUDA FP16/BF16 tensors, zero attention dropout, and no padding or custom mask. Unsupported calls raise an error instead of silently switching backends. Use SDPA for CPU execution, padded/custom-masked calls, or attention dropout.

Run `PYTHONPATH=src python examples/plug_and_play.py` for a complete module and cache example.
