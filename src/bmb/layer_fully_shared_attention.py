from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import ViTConfig, ViTForImageClassification


class LayerFullySharedSelfAttention(nn.Module):
    """Reformer-style fully shared QK: Q=K=XW for each head.

    Each head uses a single projection matrix W_{l,h} for both queries and keys,
    so the induced kernel is symmetric positive semidefinite.
    """

    def __init__(self, config: ViTConfig) -> None:
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

        # One projection per head, shared by Q and K.
        self.query_key = nn.Linear(hidden_size, self.all_head_size, bias=bool(config.qkv_bias))
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=bool(config.qkv_bias))
        self.dropout = nn.Dropout(float(config.attention_probs_dropout_prob))

        self.reset_parameters(float(getattr(config, "initializer_range", 0.02)))

    def reset_parameters(self, initializer_range: float) -> None:
        nn.init.normal_(self.query_key.weight, mean=0.0, std=initializer_range)
        nn.init.normal_(self.value.weight, mean=0.0, std=initializer_range)
        if self.query_key.bias is not None:
            nn.init.zeros_(self.query_key.bias)
        if self.value.bias is not None:
            nn.init.zeros_(self.value.bias)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def head_matrices(self) -> torch.Tensor:
        # [H, d_k, d] for the shared QK projection.
        w = self.query_key.weight.view(self.num_attention_heads, self.attention_head_size, self.hidden_size)
        return w

    def effective_layer_kernel(self) -> torch.Tensor:
        # Average head kernel for diagnostics.
        w = self.head_matrices()  # [H, d_k, d]
        kernels = torch.matmul(w.transpose(-1, -2), w)  # [H, d, d]
        return kernels.mean(dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query_layer = self.transpose_for_scores(self.query_key(hidden_states))
        key_layer = self.transpose_for_scores(self.query_key(hidden_states))
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


def apply_layer_fully_shared_attention(
    model: ViTForImageClassification,
    model_config: dict[str, Any],
) -> ViTForImageClassification:
    config = model.config
    for layer in model.vit.encoder.layer:
        layer.attention.attention = LayerFullySharedSelfAttention(config)
    return model


def summarize_layer_fully_shared_attention(model_config: dict[str, Any]) -> dict[str, float | int]:
    hidden_size = int(model_config["hidden_size"])
    num_heads = int(model_config["num_attention_heads"])
    image_size = int(model_config.get("image_size", 32))
    patch_size = int(model_config.get("patch_size", 4))
    num_patches = (image_size // patch_size) ** 2
    tokens = num_patches + 1

    per_layer_qk_params_baseline = 2 * hidden_size * hidden_size
    per_layer_qk_params_shared = hidden_size * hidden_size

    qk_flops_baseline = (
        4 * tokens * hidden_size * hidden_size
        + 2 * tokens * tokens * hidden_size
    )
    qk_flops_shared = qk_flops_baseline

    per_layer_attn_params_baseline = 4 * hidden_size * hidden_size
    per_layer_attn_params_shared = per_layer_qk_params_shared + 2 * hidden_size * hidden_size

    full_attn_flops_baseline = (
        8 * tokens * hidden_size * hidden_size
        + 4 * tokens * tokens * hidden_size
    )
    full_attn_flops_shared = (
        qk_flops_shared
        + 4 * tokens * hidden_size * hidden_size
        + 2 * tokens * tokens * hidden_size
    )

    return {
        "attention_variant": "layer_fully_shared",
        "num_heads": num_heads,
        "head_dim": hidden_size // num_heads,
        "tokens_per_example": tokens,
        "per_layer_qk_params_baseline": per_layer_qk_params_baseline,
        "per_layer_qk_params_variant": per_layer_qk_params_shared,
        "per_layer_qk_params_saved": per_layer_qk_params_baseline - per_layer_qk_params_shared,
        "per_layer_qk_param_reduction_pct": round(
            100.0 * (per_layer_qk_params_baseline - per_layer_qk_params_shared) / per_layer_qk_params_baseline,
            4,
        ),
        "per_layer_qk_flops_baseline": qk_flops_baseline,
        "per_layer_qk_flops_variant": qk_flops_shared,
        "per_layer_qk_flops_reduction_pct": round(
            100.0 * (qk_flops_baseline - qk_flops_shared) / qk_flops_baseline,
            4,
        ),
        "per_layer_attention_params_baseline": per_layer_attn_params_baseline,
        "per_layer_attention_params_variant": per_layer_attn_params_shared,
        "per_layer_attention_param_reduction_pct": round(
            100.0
            * (per_layer_attn_params_baseline - per_layer_attn_params_shared)
            / per_layer_attn_params_baseline,
            4,
        ),
        "per_layer_attention_flops_baseline": full_attn_flops_baseline,
        "per_layer_attention_flops_variant": full_attn_flops_shared,
        "per_layer_attention_flops_reduction_pct": round(
            100.0 * (full_attn_flops_baseline - full_attn_flops_shared) / full_attn_flops_baseline,
            4,
        ),
    }
