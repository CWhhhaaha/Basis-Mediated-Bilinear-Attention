from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import ViTConfig, ViTForImageClassification


class LayerPartialQKSharedSelfAttention(nn.Module):
    """Layer-level partial sharing between Q and K.

    This follows the core idea of Yang et al. (2024): construct full-layer
    query/key projections

        W^Q = [W_share | W^Q_priv],   W^K = [W_share | W^K_priv]

    where the shared block is tied between Q and K before the result is split
    into attention heads.
    """

    def __init__(self, config: ViTConfig, shared_qk_dim: int) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        num_heads = int(config.num_attention_heads)
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_heads})"
            )

        self.num_attention_heads = num_heads
        self.attention_head_size = hidden_size // num_heads
        self.all_head_size = hidden_size
        self.hidden_size = hidden_size
        self.shared_qk_dim = int(shared_qk_dim)
        self.private_qk_dim = self.hidden_size - self.shared_qk_dim
        if self.private_qk_dim < 0:
            raise ValueError(
                f"shared_qk_dim ({shared_qk_dim}) cannot exceed hidden_size ({self.hidden_size})"
            )

        self.share = nn.Linear(hidden_size, self.shared_qk_dim, bias=bool(config.qkv_bias))
        if self.private_qk_dim > 0:
            self.query_priv = nn.Linear(hidden_size, self.private_qk_dim, bias=bool(config.qkv_bias))
            self.key_priv = nn.Linear(hidden_size, self.private_qk_dim, bias=bool(config.qkv_bias))
        else:
            self.query_priv = None
            self.key_priv = None
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=bool(config.qkv_bias))
        self.dropout = nn.Dropout(float(config.attention_probs_dropout_prob))

        self.reset_parameters(float(getattr(config, "initializer_range", 0.02)))

    def reset_parameters(self, initializer_range: float) -> None:
        modules = [self.share, self.value]
        if self.query_priv is not None:
            modules.extend([self.query_priv, self.key_priv])
        for module in modules:
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def _full_qk_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        w_share = self.share.weight  # [r_s, d]
        if self.private_qk_dim > 0:
            w_q = torch.cat([w_share, self.query_priv.weight], dim=0)
            w_k = torch.cat([w_share, self.key_priv.weight], dim=0)
        else:
            w_q = w_share
            w_k = w_share
        return w_q, w_k  # [d, d]

    def head_matrices(self) -> torch.Tensor:
        w_q, w_k = self._full_qk_weights()
        w_q = w_q.view(self.num_attention_heads, self.attention_head_size, self.hidden_size).transpose(-1, -2)
        w_k = w_k.view(self.num_attention_heads, self.attention_head_size, self.hidden_size).transpose(-1, -2)
        return torch.cat([w_q, w_k], dim=0)

    def effective_layer_kernel(self) -> torch.Tensor:
        w = self.head_matrices()
        h = self.num_attention_heads
        w_q = w[:h]
        w_k = w[h:]
        kernels = torch.matmul(w_q, w_k.transpose(-1, -2))
        return kernels.mean(dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        share = self.share(hidden_states)
        if self.private_qk_dim > 0:
            query_layer = torch.cat([share, self.query_priv(hidden_states)], dim=-1)
            key_layer = torch.cat([share, self.key_priv(hidden_states)], dim=-1)
        else:
            query_layer = share
            key_layer = share

        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(self.value(hidden_states))

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


def apply_layer_partial_qk_shared_attention(
    model: ViTForImageClassification,
    model_config: dict[str, Any],
) -> ViTForImageClassification:
    shared_qk_dim = int(model_config["shared_qk_dim"])
    config = model.config
    for layer in model.vit.encoder.layer:
        layer.attention.attention = LayerPartialQKSharedSelfAttention(config, shared_qk_dim=shared_qk_dim)
    return model


def summarize_layer_partial_qk_shared_attention(model_config: dict[str, Any]) -> dict[str, float | int]:
    hidden_size = int(model_config["hidden_size"])
    num_heads = int(model_config["num_attention_heads"])
    image_size = int(model_config.get("image_size", 32))
    patch_size = int(model_config.get("patch_size", 4))
    shared_qk_dim = int(model_config["shared_qk_dim"])
    num_patches = (image_size // patch_size) ** 2
    tokens = num_patches + 1

    per_layer_qk_params_baseline = 2 * hidden_size * hidden_size
    per_layer_qk_params_variant = 2 * hidden_size * hidden_size - hidden_size * shared_qk_dim

    qk_flops_baseline = (
        4 * tokens * hidden_size * hidden_size
        + 2 * tokens * tokens * hidden_size
    )
    qk_flops_variant = (
        2 * tokens * hidden_size * shared_qk_dim
        + 4 * tokens * hidden_size * (hidden_size - shared_qk_dim)
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
        "attention_variant": "layer_partial_qk_shared",
        "shared_qk_dim": shared_qk_dim,
        "private_qk_dim": hidden_size - shared_qk_dim,
        "num_heads": num_heads,
        "head_dim": hidden_size // num_heads,
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
