from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import ViTConfig, ViTForImageClassification


class LayerLowRankSelfAttention(nn.Module):
    """Training-time low-rank Q/K baseline.

    Each head's query and key projections are factorized as
        W^Q_{l,h} = A^Q_{l,h} B^Q_{l,h}
        W^K_{l,h} = A^K_{l,h} B^K_{l,h}
    with A in R^{d x r} and B in R^{r x d_k}.

    This is a head-wise low-rank parameterization trained end-to-end,
    distinct from post-training SVD compression.
    """

    def __init__(self, config: ViTConfig, low_rank: int) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        num_heads = int(config.num_attention_heads)
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_heads})"
            )
        if low_rank <= 0:
            raise ValueError("low_rank must be positive")

        self.num_attention_heads = num_heads
        self.attention_head_size = hidden_size // num_heads
        self.all_head_size = hidden_size
        self.hidden_size = hidden_size
        self.low_rank = int(low_rank)

        # Head-wise low-rank factors: A in R^{d x r}, B in R^{r x d_k}.
        self.q_a = nn.Parameter(torch.empty(num_heads, hidden_size, self.low_rank))
        self.q_b = nn.Parameter(torch.empty(num_heads, self.low_rank, self.attention_head_size))
        self.k_a = nn.Parameter(torch.empty(num_heads, hidden_size, self.low_rank))
        self.k_b = nn.Parameter(torch.empty(num_heads, self.low_rank, self.attention_head_size))

        self.value = nn.Linear(hidden_size, self.all_head_size, bias=bool(config.qkv_bias))
        self.dropout = nn.Dropout(float(config.attention_probs_dropout_prob))

        self.reset_parameters(float(getattr(config, "initializer_range", 0.02)))

    def reset_parameters(self, initializer_range: float) -> None:
        for p in (self.q_a, self.q_b, self.k_a, self.k_b):
            nn.init.normal_(p, mean=0.0, std=initializer_range)
        nn.init.normal_(self.value.weight, mean=0.0, std=initializer_range)
        if self.value.bias is not None:
            nn.init.zeros_(self.value.bias)

    def _project(self, hidden_states: torch.Tensor, a: nn.Parameter, b: nn.Parameter) -> torch.Tensor:
        """Low-rank projection: X @ (A @ B) via einsum.

        hidden_states: [batch, seq, d]
        a: [heads, d, r]
        b: [heads, r, d_k]
        returns: [batch, heads, seq, d_k]
        """
        mid = torch.einsum("bnd,hdr->bhnr", hidden_states, a)
        out = torch.einsum("bhnr,hrd->bhnd", mid, b)
        return out

    def head_matrices(self) -> torch.Tensor:
        """Reconstruct full [H, d, d_k] projection matrices for diagnostics."""
        w_q = torch.matmul(self.q_a, self.q_b)  # [H, d, d_k]
        w_k = torch.matmul(self.k_a, self.k_b)  # [H, d, d_k]
        return torch.cat([w_q, w_k], dim=0)

    def effective_layer_kernel(self) -> torch.Tensor:
        w = self.head_matrices()
        h = self.num_attention_heads
        w_q = w[:h]
        w_k = w[h:]
        kernels = torch.matmul(w_q.transpose(-1, -2), w_k)  # [H, d, d]
        return kernels.mean(dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query_layer = self._project(hidden_states, self.q_a, self.q_b)
        key_layer = self._project(hidden_states, self.k_a, self.k_b)

        batch_size, seq_len, _ = hidden_states.shape
        value_layer = self.value(hidden_states)
        value_layer = value_layer.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        value_layer = value_layer.permute(0, 2, 1, 3)

        if output_attentions or head_mask is not None:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(float(self.attention_head_size))
            attention_probs = torch.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)
            if head_mask is not None:
                attention_probs = attention_probs * head_mask
            context_layer = torch.matmul(attention_probs, value_layer)
        else:
            context_layer = F.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                dropout_p=float(self.dropout.p) if self.training else 0.0,
                is_causal=False,
            )
            attention_probs = None

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return context_layer, attention_probs


def apply_layer_lowrank_attention(
    model: ViTForImageClassification,
    model_config: dict[str, Any],
) -> ViTForImageClassification:
    low_rank = int(model_config["low_rank"])
    config = model.config
    for layer in model.vit.encoder.layer:
        layer.attention.attention = LayerLowRankSelfAttention(config, low_rank=low_rank)
    return model


def summarize_layer_lowrank_attention(model_config: dict[str, Any]) -> dict[str, float | int]:
    hidden_size = int(model_config["hidden_size"])
    num_heads = int(model_config["num_attention_heads"])
    image_size = int(model_config.get("image_size", 32))
    patch_size = int(model_config.get("patch_size", 4))
    low_rank = int(model_config["low_rank"])
    head_dim = hidden_size // num_heads
    num_patches = (image_size // patch_size) ** 2
    tokens = num_patches + 1

    per_layer_qk_params_baseline = 2 * hidden_size * hidden_size
    # Each head, each side: d*r + r*d_k
    per_layer_qk_params_variant = 2 * num_heads * (hidden_size * low_rank + low_rank * head_dim)

    qk_flops_baseline = (
        4 * tokens * hidden_size * hidden_size
        + 2 * tokens * tokens * hidden_size
    )
    # Low-rank: 2 projections per side (A and B), plus score matmul.
    # A: tokens * d * r; B: tokens * r * d_k
    qk_flops_variant = (
        4 * num_heads * tokens * hidden_size * low_rank
        + 4 * num_heads * tokens * low_rank * head_dim
        + 2 * tokens * tokens * hidden_size
    )

    per_layer_attn_params_baseline = 4 * hidden_size * hidden_size
    per_layer_attn_params_variant = per_layer_qk_params_variant + 2 * hidden_size * hidden_size

    full_attn_flops_baseline = (
        8 * tokens * hidden_size * hidden_size
        + 4 * tokens * tokens * hidden_size
    )
    full_attn_flops_variant = (
        qk_flops_variant
        + 4 * tokens * hidden_size * hidden_size
        + 2 * tokens * tokens * hidden_size
    )

    return {
        "attention_variant": "layer_lowrank",
        "low_rank": low_rank,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "tokens_per_example": tokens,
        "per_layer_qk_params_baseline": per_layer_qk_params_baseline,
        "per_layer_qk_params_variant": per_layer_qk_params_variant,
        "per_layer_qk_params_saved": per_layer_qk_params_baseline - per_layer_qk_params_variant,
        "per_layer_qk_param_reduction_pct": round(
            100.0 * (per_layer_qk_params_baseline - per_layer_qk_params_variant) / per_layer_qk_params_baseline,
            4,
        ),
        "per_layer_qk_flops_baseline": qk_flops_baseline,
        "per_layer_qk_flops_variant": qk_flops_variant,
        "per_layer_qk_flops_reduction_pct": round(
            100.0 * (qk_flops_baseline - qk_flops_variant) / qk_flops_baseline,
            4,
        ),
        "per_layer_attention_params_baseline": per_layer_attn_params_baseline,
        "per_layer_attention_params_variant": per_layer_attn_params_variant,
        "per_layer_attention_param_reduction_pct": round(
            100.0
            * (per_layer_attn_params_baseline - per_layer_attn_params_variant)
            / per_layer_attn_params_baseline,
            4,
        ),
        "per_layer_attention_flops_baseline": full_attn_flops_baseline,
        "per_layer_attention_flops_variant": full_attn_flops_variant,
        "per_layer_attention_flops_reduction_pct": round(
            100.0 * (full_attn_flops_baseline - full_attn_flops_variant) / full_attn_flops_baseline,
            4,
        ),
    }
