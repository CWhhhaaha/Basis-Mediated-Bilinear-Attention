"""Portable BMB-UV attention with module and functional APIs.

Explicit, position-encoded K/V cache layout:
    K [batch,n_kv_heads,sequence,factor_rank]
    V [batch,n_kv_heads,sequence,value_dim]

The default backend is PyTorch SDPA. The reference backend favors portability
and inspection, not memory efficiency. FlashAttention-4 is an explicit optional
backend; importing or using the default API never imports it.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import math

import torch
import torch.nn.functional as F
from torch import nn


class RotaryEmbedding(nn.Module):
    """Rotary frequencies with float32 trigonometric evaluation."""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim <= 0 or dim % 2:
            raise ValueError("RoPE dimension must be a positive even integer")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_offset: int = 0,
        length: int | None = None,
        position_ids: torch.Tensor | None = None,
    ):
        if position_ids is None:
            if length is None:
                length = x.shape[-2]
            position_ids = torch.arange(
                position_offset,
                position_offset + length,
                device=x.device,
                dtype=torch.long,
            )
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.ndim != 2:
            raise ValueError("position_ids must have shape [T] or [B,T]")
        positions = position_ids.to(device=x.device, dtype=torch.float32)
        inv_freq = self.inv_freq.to(device=x.device, dtype=torch.float32)
        freqs = torch.einsum("bt,f->btf", positions, inv_freq)
        cos = freqs.cos()[:, :, None, :]
        sin = freqs.sin()[:, :, None, :]
        return cos.to(x.dtype), sin.to(x.dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply split-half RoPE to [B,T,H,d] tensors."""
    if x.shape[-1] % 2:
        raise ValueError("RoPE requires an even head width")
    half = x.shape[-1] // 2
    if cos.shape[-1] != half or sin.shape[-1] != half:
        raise ValueError("cos/sin width must be half the attention head width")
    left, right = x[..., :half], x[..., half:]
    return torch.cat(
        (left * cos - right * sin, right * cos + left * sin), dim=-1
    )


def _boolean_mask(mask: torch.Tensor) -> torch.Tensor:
    return mask if mask.dtype == torch.bool else mask != 0


def _avoid_empty_rows(mask: torch.Tensor) -> torch.Tensor:
    """Give padding-only query rows a dummy key so attention cannot produce NaNs."""
    empty = ~mask.any(dim=-1, keepdim=True)
    if not torch.any(empty):
        return mask
    mask = mask.clone()
    mask[..., :1] = mask[..., :1] | empty
    return mask


def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must be rank-4 BHTD tensors")
    batch, heads, query_len, score_dim = q.shape
    if query_len <= 0:
        raise ValueError("Attention requires at least one query position")
    if k.shape[0] != batch or v.shape[0] != batch:
        raise ValueError("q, k and v must have the same batch size")
    if k.shape[1] != v.shape[1] or heads % k.shape[1]:
        raise ValueError("Query heads must be divisible by the shared KV-head count")
    if k.shape[2] != v.shape[2] or k.shape[2] <= 0:
        raise ValueError("K and V must have the same non-zero sequence length")
    if k.shape[3] != score_dim:
        raise ValueError("Q and K must have the same score width")


def _prepare_attention_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_len: int,
    is_causal: bool,
):
    query_len, key_len = q.shape[-2], k.shape[-2]
    if past_len < 0 or key_len != past_len + query_len:
        raise ValueError(
            f"Expected key length past_len + query_len = {past_len + query_len}, "
            f"got {key_len}"
        )
    batch, heads = q.shape[:2]
    device, dtype = q.device, q.dtype
    boolean_mask = None
    additive_mask = None

    if attention_mask is not None:
        if attention_mask.ndim == 2:
            if attention_mask.shape != (batch, key_len):
                raise ValueError(
                    f"2D attention_mask must be [B,L]={(batch, key_len)}, "
                    f"got {tuple(attention_mask.shape)}"
                )
            boolean_mask = _boolean_mask(attention_mask)[:, None, None, :]
        elif attention_mask.ndim == 4:
            if attention_mask.shape[0] not in (1, batch):
                raise ValueError("4D attention_mask has an incompatible batch dimension")
            if attention_mask.shape[1] not in (1, heads):
                raise ValueError("4D attention_mask has an incompatible head dimension")
            if attention_mask.shape[-2:] != (query_len, key_len):
                raise ValueError(
                    f"4D attention_mask must end in [T,L]={(query_len, key_len)}"
                )
            if attention_mask.dtype == torch.bool:
                boolean_mask = attention_mask
            else:
                additive_mask = attention_mask.to(device=device, dtype=dtype)
        else:
            raise ValueError("attention_mask must be 2D key padding or 4D boolean/additive")
        if boolean_mask is not None:
            boolean_mask = boolean_mask.to(device=device)

    needs_explicit_causal = is_causal and query_len > 1 and past_len > 0
    if is_causal and (attention_mask is not None or needs_explicit_causal):
        query_positions = past_len + torch.arange(query_len, device=device)
        key_positions = torch.arange(key_len, device=device)
        causal = key_positions[None, :] <= query_positions[:, None]
        causal = causal[None, None, :, :]
        boolean_mask = causal if boolean_mask is None else boolean_mask & causal

    if boolean_mask is not None:
        boolean_mask = _avoid_empty_rows(boolean_mask)

    mask = boolean_mask
    if additive_mask is not None:
        if boolean_mask is not None:
            additive_mask = additive_mask.masked_fill(~boolean_mask, float("-inf"))
        empty = torch.isneginf(additive_mask).all(dim=-1, keepdim=True)
        if torch.any(empty):
            additive_mask = additive_mask.clone()
            additive_mask[..., :1] = torch.where(
                empty,
                torch.zeros((), device=device, dtype=dtype),
                additive_mask[..., :1],
            )
        mask = additive_mask

    use_builtin_causal = bool(
        is_causal and query_len > 1 and past_len == 0 and mask is None
    )
    return mask, use_builtin_causal


def _load_fa4():
    """Import FA4 only when its backend is explicitly requested."""
    try:
        cute = importlib.import_module("flash_attn.cute")
    except (ImportError, OSError) as exc:
        raise ImportError(
            "backend='fa4' requires FlashAttention-4. Install it with "
            "`python -m pip install flash-attn-4` (or `flash-attn-4[cu13]` "
            "for the recommended CUDA 13 dependency set) on a supported GPU."
        ) from exc
    try:
        return cute.flash_attn_func
    except AttributeError as exc:
        raise ImportError(
            "The installed flash_attn.cute package does not expose "
            "flash_attn_func; install a current flash-attn-4 release."
        ) from exc


def fa4_backend_info():
    """Return the package version after verifying the FA4 entry point."""
    _load_fa4()
    try:
        version = importlib.metadata.version("flash-attn-4")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return {"package": "flash-attn-4", "version": version}


def _fa4_attention(q, k, v, *, is_causal, past_len, dropout_p):
    """Run dense FA4 on BHTD inputs without an SDPA fallback."""
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("backend='fa4' requires Q, K, and V on CUDA devices")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("backend='fa4' currently requires FP16 or BF16 Q/K/V")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("backend='fa4' requires Q, K, and V to have the same dtype")
    if dropout_p:
        raise ValueError(
            "backend='fa4' does not support attention dropout through this API; "
            "use dropout=0 or the SDPA backend"
        )
    if past_len < 0 or k.shape[2] != past_len + q.shape[2]:
        raise ValueError(
            "FA4 attention requires key length = past_len + query length"
        )
    flash_attn_func = _load_fa4()
    # FA4 consumes BTHD and natively supports fewer K/V heads than Q heads.
    # Its bottom-right causal alignment matches our cached-query convention.
    output = flash_attn_func(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        causal=is_causal,
        softmax_scale=q.shape[-1] ** -0.5,
    )
    if isinstance(output, tuple):
        output = output[0]
    return output.transpose(1, 2)


def grouped_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    attention_mask: torch.Tensor | None = None,
    past_len: int = 0,
    dropout_p: float = 0.0,
    backend: str = "sdpa",
):
    """Grouped attention over BHTD tensors with offset-aware masks.

    ``backend='sdpa'`` uses PyTorch SDPA. ``backend='reference'`` materializes
    attention probabilities and is intended for portability and debugging.
    ``backend='fa4'`` explicitly calls FlashAttention-4 without fallback.
    """
    _validate_qkv(q, k, v)
    if backend not in {"sdpa", "reference", "fa4"}:
        raise ValueError("backend must be 'sdpa', 'reference', or 'fa4'")
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must be in [0,1)")
    if backend == "fa4":
        if attention_mask is not None:
            raise ValueError(
                "backend='fa4' currently supports only unmasked dense or causal "
                "attention; use SDPA/reference for padding or custom masks"
            )
        return _fa4_attention(
            q, k, v, is_causal=is_causal, past_len=past_len, dropout_p=dropout_p
        )
    mask, use_builtin_causal = _prepare_attention_mask(
        q, k, attention_mask, past_len, is_causal
    )
    group_size = q.shape[1] // k.shape[1]

    if backend == "reference":
        if use_builtin_causal:
            length = q.shape[-2]
            allowed = torch.tril(
                torch.ones(length, length, device=q.device, dtype=torch.bool)
            )
            mask = allowed[None, None]
        k = k.repeat_interleave(group_size, dim=1)
        v = v.repeat_interleave(group_size, dim=1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, float("-inf"))
            else:
                scores = scores + mask
        probabilities = torch.softmax(scores, dim=-1)
        probabilities = F.dropout(probabilities, p=dropout_p, training=dropout_p > 0)
        return probabilities @ v

    kwargs = {
        "attn_mask": mask,
        "dropout_p": dropout_p,
        "is_causal": use_builtin_causal,
    }
    if group_size != 1:
        try:
            return F.scaled_dot_product_attention(q, k, v, enable_gqa=True, **kwargs)
        except TypeError:
            k = k.repeat_interleave(group_size, dim=1)
            v = v.repeat_interleave(group_size, dim=1)
    return F.scaled_dot_product_attention(q, k, v, **kwargs)


def grouped_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    attention_mask: torch.Tensor | None = None,
    past_len: int = 0,
    dropout_p: float = 0.0,
):
    """Backward-compatible SDPA-only wrapper around :func:`grouped_attention`."""
    return grouped_attention(
        q,
        k,
        v,
        is_causal=is_causal,
        attention_mask=attention_mask,
        past_len=past_len,
        dropout_p=dropout_p,
        backend="sdpa",
    )


def project_bmb_uv(
    z: torch.Tensor,
    q_factor: torch.Tensor,
    k_factor: torch.Tensor,
):
    """Project shared latent Z into explicit Q/K, both in [B,T,H,s] layout."""
    if z.ndim != 3:
        raise ValueError("z must have shape [B,T,r]")
    if q_factor.ndim != 3 or k_factor.ndim != 3:
        raise ValueError("q_factor and k_factor must have shape [H,r,s]")
    if q_factor.shape[1:] != k_factor.shape[1:]:
        raise ValueError("Q and K factors must share latent rank and factor rank")
    if z.shape[-1] != q_factor.shape[1]:
        raise ValueError("The latent width of z does not match the UV factors")
    if q_factor.shape[0] <= 0 or k_factor.shape[0] <= 0:
        raise ValueError("Q and K factors must contain at least one head")
    if q_factor.shape[0] % k_factor.shape[0]:
        raise ValueError("Query heads must be divisible by KV heads")
    for name, factor in (("q_factor", q_factor), ("k_factor", k_factor)):
        if factor.device != z.device or factor.dtype != z.dtype:
            raise ValueError(f"{name} and z must share device and dtype")
    q = torch.einsum("btr,hrs->bths", z, q_factor)
    k = torch.einsum("btr,hrs->bths", z, k_factor)
    return q, k


def _cache_length(past_key_value) -> int:
    if past_key_value is None:
        return 0
    if isinstance(past_key_value, dict):
        if not {"k", "v", "length"}.issubset(past_key_value):
            raise ValueError("Static cache must contain 'k', 'v', and 'length'")
        length = past_key_value["length"]
        if not isinstance(length, int) or length < 0:
            raise ValueError("Static cache 'length' must be a non-negative int")
        return length
    if isinstance(past_key_value, (tuple, list)) and len(past_key_value) == 2:
        if not isinstance(past_key_value[0], torch.Tensor) or past_key_value[0].ndim != 4:
            raise ValueError("Dynamic K cache must be a rank-4 BHTD tensor")
        return past_key_value[0].shape[2]
    raise TypeError("past_key_value must be a (K,V) tuple or a static-cache dict")


def _append_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    past_key_value,
    use_cache: bool,
):
    """Append BHTD K/V and return (K,V,past_length,present)."""
    batch, kv_heads, query_len, score_dim = k.shape
    past_len = _cache_length(past_key_value)
    if isinstance(past_key_value, dict):
        past_k, past_v = past_key_value["k"], past_key_value["v"]
        expected_k = (batch, kv_heads, score_dim)
        expected_v = (batch, kv_heads, v.shape[-1])
        if past_k.ndim != 4 or (
            past_k.shape[0], past_k.shape[1], past_k.shape[3]
        ) != expected_k:
            raise ValueError("Static K cache has an incompatible shape")
        if past_v.ndim != 4 or (
            past_v.shape[0], past_v.shape[1], past_v.shape[3]
        ) != expected_v:
            raise ValueError("Static V cache has an incompatible shape")
        if past_k.shape[2] != past_v.shape[2]:
            raise ValueError("Static K and V caches must have equal capacity")
        if past_len + query_len > past_k.shape[2]:
            raise ValueError(
                f"Static cache capacity {past_k.shape[2]} is insufficient for "
                f"{past_len + query_len} positions"
            )
        for name, tensor in (("K", past_k), ("V", past_v)):
            if tensor.device != k.device or tensor.dtype != k.dtype:
                raise ValueError(f"Static {name} cache must match input device and dtype")
        past_key_value["k"][:, :, past_len:past_len + query_len] = k
        past_key_value["v"][:, :, past_len:past_len + query_len] = v
        k = past_key_value["k"][:, :, :past_len + query_len]
        v = past_key_value["v"][:, :, :past_len + query_len]
        past_key_value["length"] = past_len + query_len
        present = past_key_value if use_cache else None
    else:
        if past_key_value is not None:
            past_k, past_v = past_key_value
            if past_k.ndim != 4 or past_v.ndim != 4:
                raise ValueError("Dynamic K/V caches must be rank-4 BHTD tensors")
            if past_k.shape[:2] != (batch, kv_heads) or past_k.shape[3] != score_dim:
                raise ValueError("Dynamic K cache has an incompatible shape")
            if past_v.shape != (*past_k.shape[:3], v.shape[-1]):
                raise ValueError("Dynamic V cache has an incompatible shape")
            if past_k.device != k.device or past_v.device != k.device:
                raise ValueError("Dynamic cache and current inputs must share a device")
            if past_k.dtype != k.dtype or past_v.dtype != v.dtype:
                raise ValueError("Dynamic cache and current inputs must share dtypes")
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        present = (k, v) if use_cache else None
    return k, v, past_len, present


def bmb_uv_attention(
    z: torch.Tensor,
    q_factor: torch.Tensor,
    k_factor: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    past_key_value=None,
    use_cache: bool = False,
    cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
    is_causal: bool = True,
    dropout_p: float = 0.0,
    backend: str = "sdpa",
):
    """Functional BMB-UV attention, analogous to a low-level attention operator.

    Args:
        z: Shared latent [B,T,r], normally computed as ``z = x @ basis.T``.
        q_factor: Per-query-head factors [H,r,s].
        k_factor: Per-KV-head factors [H_kv,r,s].
        value: Values in [B,T,H_kv,d_v] layout.
        cos_sin: Optional external RoPE tensors broadcastable to Q/K in BTHD
            layout. If omitted, Q/K are used without position-dependent rotation.
        past_key_value: Optional dynamic (K,V) tuple or static-cache dictionary.
        backend: ``'sdpa'`` (default) or ``'reference'``.

    Returns:
        ``(output, present_key_value)`` where output is [B,T,H,d_v]. The output
        projection is intentionally left to the caller in this functional API.
    """
    batch, query_len, _ = z.shape
    q, k = project_bmb_uv(z, q_factor, k_factor)
    expected_value = (batch, query_len, k_factor.shape[0])
    if value.ndim != 4 or value.shape[:3] != expected_value:
        raise ValueError("value must have shape [B,T,H_kv,d_v]")
    if value.device != z.device or value.dtype != z.dtype:
        raise ValueError("value and z must share device and dtype")
    if cos_sin is not None:
        if not isinstance(cos_sin, (tuple, list)) or len(cos_sin) != 2:
            raise ValueError("cos_sin must be a (cos,sin) tuple")
        q = apply_rope(q, cos_sin[0], cos_sin[1])
        k = apply_rope(k, cos_sin[0], cos_sin[1])

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = value.transpose(1, 2)
    k, v, past_len, present = _append_kv_cache(
        k, v, past_key_value, use_cache
    )
    y = grouped_attention(
        q,
        k,
        v,
        is_causal=is_causal,
        attention_mask=attention_mask,
        past_len=past_len,
        dropout_p=dropout_p,
        backend=backend,
    )
    return y.transpose(1, 2), present


class BMBAttention(nn.Module):
    """Trainable BMB-UV attention sublayer on [B,T,d_model] hidden states.

    It includes basis/UV projections, V and output projections, optional built-in
    RoPE, grouped attention, and explicit dynamic or static K/V caching. Set
    ``n_kv_heads=n_heads`` for MHA-style sharing, ``1`` for MQA-style sharing,
    or an intermediate divisor for GQA-style sharing.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        latent_rank: int,
        factor_rank: int,
        n_kv_heads: int | None = None,
        value_dim: int = 64,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        rope: bool = True,
        backend: str = "sdpa",
    ):
        super().__init__()
        n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        if d_model <= 0 or n_heads <= 0 or n_kv_heads <= 0:
            raise ValueError("d_model and head counts must be positive")
        if n_heads % n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if latent_rank <= 0 or factor_rank <= 0 or value_dim <= 0:
            raise ValueError("latent_rank, factor_rank and value_dim must be positive")
        if rope and factor_rank % 2:
            raise ValueError("factor_rank must be even when using RoPE")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if backend not in {"sdpa", "reference", "fa4"}:
            raise ValueError("backend must be 'sdpa', 'reference', or 'fa4'")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_head = n_heads
        self.n_kv_head = n_kv_heads
        self.group_size = n_heads // n_kv_heads
        self.latent_rank = latent_rank
        self.factor_rank = factor_rank
        self.value_dim = value_dim
        self.dropout = dropout
        self.rope_theta = rope_theta
        self.use_rope = rope
        self.backend = backend

        self.basis = nn.Linear(d_model, latent_rank, bias=False)
        self.q_factor = nn.Parameter(torch.empty(n_heads, latent_rank, factor_rank))
        self.k_factor = nn.Parameter(torch.empty(n_kv_heads, latent_rank, factor_rank))
        self.v_proj = nn.Linear(d_model, n_kv_heads * value_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * value_dim, d_model, bias=False)
        self.rotary = RotaryEmbedding(factor_rank, rope_theta) if rope else None
        self.reset_bmb_parameters()

    @property
    def score_dim(self) -> int:
        return self.factor_rank

    def get_config(self) -> dict:
        """Return constructor arguments for configuration serialization."""
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "latent_rank": self.latent_rank,
            "factor_rank": self.factor_rank,
            "value_dim": self.value_dim,
            "rope_theta": self.rope_theta,
            "dropout": self.dropout,
            "rope": self.use_rope,
            "backend": self.backend,
        }

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)

    def reset_bmb_parameters(self):
        nn.init.xavier_uniform_(self.basis.weight)
        for factors in (self.q_factor, self.k_factor):
            for head_factor in factors:
                nn.init.xavier_uniform_(head_factor)

    def project_qk(self, x: torch.Tensor):
        """Return unrotated Q/K in BTHD layout for integration and inspection."""
        return project_bmb_uv(self.basis(x), self.q_factor, self.k_factor)

    def create_static_cache(
        self,
        batch_size: int,
        max_length: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor | int]:
        """Allocate one layer's static cache; create one per model layer."""
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("batch_size and max_length must be positive")
        device = self.basis.weight.device if device is None else device
        dtype = self.basis.weight.dtype if dtype is None else dtype
        return {
            "k": torch.zeros(
                batch_size,
                self.n_kv_heads,
                max_length,
                self.factor_rank,
                device=device,
                dtype=dtype,
            ),
            "v": torch.zeros(
                batch_size,
                self.n_kv_heads,
                max_length,
                self.value_dim,
                device=device,
                dtype=dtype,
            ),
            "length": 0,
        }

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value=None,
        use_cache: bool = False,
        position_ids: torch.Tensor | None = None,
        is_causal: bool = True,
        cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        """Return ``(output, present_key_value)`` with output shaped like ``x``.

        Dynamic caches support autograd. Static-cache writes are intended for
        inference. ``cos_sin`` allows a host model to supply its own RoPE and
        takes precedence over the built-in rotary module.
        """
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"x must have shape [B,T,{self.d_model}]")
        batch, query_len, _ = x.shape
        if query_len <= 0:
            raise ValueError("Attention requires a non-empty sequence")
        past_len = _cache_length(past_key_value)
        if cos_sin is not None and position_ids is not None:
            raise ValueError("Pass either cos_sin or position_ids, not both")
        if cos_sin is None and self.use_rope:
            if position_ids is not None:
                expected = (batch, query_len) if position_ids.ndim == 2 else (query_len,)
                if tuple(position_ids.shape) != expected:
                    raise ValueError(
                        f"position_ids must have shape [T] or [B,T] for T={query_len}"
                    )
            cos_sin = self.rotary(
                x,
                position_offset=past_len,
                length=query_len,
                position_ids=position_ids,
            )
        elif position_ids is not None:
            raise ValueError("position_ids requires rope=True or explicit cos_sin")

        z = self.basis(x)
        value = self.v_proj(x).view(
            batch, query_len, self.n_kv_heads, self.value_dim
        )
        y, present = bmb_uv_attention(
            z,
            self.q_factor,
            self.k_factor,
            value,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cos_sin=cos_sin,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
            backend=self.backend,
        )
        y = y.reshape(batch, query_len, self.n_heads * self.value_dim)
        return self.o_proj(y), present

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"n_kv_heads={self.n_kv_heads}, latent_rank={self.latent_rank}, "
            f"factor_rank={self.factor_rank}, value_dim={self.value_dim}, "
            f"rope={self.use_rope}, backend={self.backend!r}"
        )
