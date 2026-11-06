# BMB-UV integration guide

BMB-UV can be integrated as a tensor operator, an attention sublayer, or the included causal language model. Replacing an existing trained attention layer requires training or fine-tuning because BMB-UV changes the Q/K parameterization.

## PyTorch attention sublayer

```python
import torch
from torch import nn
from bmb import BMBAttention

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.RMSNorm(768)
        self.attn = BMBAttention(
            d_model=768,
            n_heads=22,
            n_kv_heads=2,
            latent_rank=64,
            factor_rank=32,
            value_dim=64,
        )

    def forward(self, x, attention_mask=None, position_ids=None):
        y, _ = self.attn(
            self.norm(x),
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return x + y
```

Input and output are `[B,T,d_model]`. Set `n_kv_heads` to the desired MHA, MQA, or GQA-style KV-head count.

## Custom projection and cache pipelines

Use `bmb_uv_attention` when the caller owns projections or cache management:

```python
from bmb import BMBAttention, bmb_uv_attention

params = BMBAttention(
    d_model=768, n_heads=22, n_kv_heads=2,
    latent_rank=64, factor_rank=32, value_dim=64,
)
z = params.basis(hidden)                    # [B,T,r]
v = params.v_proj(hidden).view(B, T, 2, 64) # [B,T,H_kv,d_v]
cos_sin = params.rotary(hidden, 0, T)

y, cache = bmb_uv_attention(
    z, params.q_factor, params.k_factor, v,
    cos_sin=cos_sin, use_cache=True,
)
y = params.o_proj(y.reshape(B, T, 22 * 64))
```

For a custom position transform, use the lower-level stages:

```python
from bmb import project_bmb_uv, grouped_attention

q, k = project_bmb_uv(z, q_factor, k_factor)  # [B,T,H,s]
q = custom_position_transform(q)
k = custom_position_transform(k)

# Append or transform K/V in the host system, then pass BHTD tensors.
y = grouped_attention(
    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
    attention_mask=host_mask,
    past_len=host_past_length,
)
```

`grouped_attention` does not append caches in this low-level path.

## Hugging Face AutoModel and generation

Install the training extra and register the included model:

```python
import torch
from transformers import AutoModelForCausalLM
from bmb.transformers_integration import register_bmb

BMBConfig, BMBForCausalLM = register_bmb()
config = BMBConfig(
    vocab_size=50304,
    n_layer=12,
    n_head=22,
    n_kv_head=2,
    n_embd=768,
    latent_rank=64,
    factor_rank=32,
    value_dim=64,
)
model = AutoModelForCausalLM.from_config(config).eval()

input_ids = torch.randint(0, config.vocab_size, (1, 8))
with torch.inference_mode():
    output_ids = model.generate(
        input_ids, max_new_tokens=20, do_sample=False, use_cache=True
    )
```

Registration covers the included `BMBConfig` and `BMBForCausalLM`. Other architectures need an adapter for their projection, position-encoding, mask, and cache contracts.

## Cache integration

Dynamic cache:

```python
y, cache = attn(prefix, use_cache=True)
y_next, cache = attn(next_x, past_key_value=cache, use_cache=True)
```

Static cache, with one dictionary per layer:

```python
cache = attn.create_static_cache(
    batch_size=B, max_length=max_positions,
    device=device, dtype=dtype,
)
with torch.inference_mode():
    attn(prefix, past_key_value=cache, use_cache=True)
    attn(next_x, past_key_value=cache, use_cache=True)
```

K and V are explicit and position-encoded:

$$
K\in\mathbb{R}^{B\times H_{kv}\times L\times s},\qquad
V\in\mathbb{R}^{B\times H_{kv}\times L\times d_v}.
$$

Dynamic caches support autograd but concatenate on append. Static caches avoid repeated allocation and are intended for inference.

## Cache and parameter calculations

For the same batch, length, layer count, KV-head count, V width, and dtype, the explicit K/V payload reduction is

$$
1-\frac{s+d_v}{d_k+d_v}.
$$

For the paper's XL setting (`s=32`, `d_k=d_v=64`), this is **25.00%**. Calculate another configuration with:

```python
from bmb import compare_gqa_bmb_uv_cache

report = compare_gqa_bmb_uv_cache(
    batch_size=1,
    sequence_length=1_000_000,
    n_layers=48,
    n_kv_heads=2,
    head_dim=64,
    factor_rank=32,
    value_dim=64,
)
```

Per-layer Q/K projection counts, excluding V and output projections, are

$$
P_{\mathrm{GQA}}=d(H+H_{kv})d_k,
\qquad
P_{\mathrm{BMB-UV}}=dr+(H+H_{kv})rs.
$$

For XL (`d=1600`, `H=48`, `H_kv=2`, `d_k=64`, `r=128`, `s=32`), this changes Q/K parameters from **5,120,000 to 409,600 per layer**, a **92.00% reduction**. Use `compare_gqa_bmb_uv_qk_parameters` for other shapes.

Caching the shared latent Z instead of explicit K is smaller only when $r<H_{kv}s$. The paper's XL configuration does not satisfy this condition and therefore caches explicit K/V.

## Backend selection

- **SDPA:** default; supports padding and custom masks, CPU or CUDA, dynamic/static caches, and attention dropout.
- **Reference:** explicit scores and probabilities for debugging.
- **FA4:** optional `backend="fa4"` path for CUDA FP16/BF16, zero attention dropout, and causal calls without padding or custom masks.

The module and functional APIs share the same backend and cache semantics. The systems benchmark measures prefill and decode separately for GQA and GQA+BMB-UV.
