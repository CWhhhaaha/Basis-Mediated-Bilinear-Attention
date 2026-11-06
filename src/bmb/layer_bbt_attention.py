from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from transformers import ViTConfig, ViTForImageClassification


class LayerBBTSelfAttention(nn.Module):
    def __init__(self, config: ViTConfig, latent_rank: int) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        num_heads = int(config.num_attention_heads)
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_heads})"
            )
        if latent_rank <= 0:
            raise ValueError("latent_rank must be positive")

        self.num_attention_heads = num_heads
        self.attention_head_size = hidden_size // num_heads
        self.all_head_size = hidden_size
        self.hidden_size = hidden_size
        self.latent_rank = int(latent_rank)

        self.basis = nn.Linear(hidden_size, self.latent_rank, bias=False)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=bool(config.qkv_bias))
        self.dropout = nn.Dropout(float(config.attention_probs_dropout_prob))

        self.reset_parameters(float(getattr(config, "initializer_range", 0.02)))

    def reset_parameters(self, initializer_range: float) -> None:
        nn.init.normal_(self.basis.weight, mean=0.0, std=initializer_range)
        nn.init.normal_(self.value.weight, mean=0.0, std=initializer_range)
        if self.value.bias is not None:
            nn.init.zeros_(self.value.bias)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def head_matrices(self) -> torch.Tensor:
        eye = torch.eye(self.latent_rank, device=self.basis.weight.device, dtype=self.basis.weight.dtype)
        return eye.unsqueeze(0).repeat(self.num_attention_heads, 1, 1)

    def effective_layer_kernel(self) -> torch.Tensor:
        basis = self.basis.weight.transpose(0, 1)
        return basis @ basis.transpose(0, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        latent = self.basis(hidden_states)
        attention_scores = latent @ latent.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(float(self.latent_rank))
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        attention_probs = attention_probs.unsqueeze(1)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        context_layer = attention_probs @ value_layer
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(context_layer.size()[:-2] + (self.all_head_size,))
        returned_probs = attention_probs if output_attentions else None
        return context_layer, returned_probs


def apply_layer_bbt_attention(
    model: ViTForImageClassification,
    model_config: dict[str, Any],
) -> ViTForImageClassification:
    latent_rank = int(model_config["latent_rank"])
    config = model.config
    for layer in model.vit.encoder.layer:
        layer.attention.attention = LayerBBTSelfAttention(config, latent_rank=latent_rank)
    return model


def summarize_layer_bbt_attention(model_config: dict[str, Any]) -> dict[str, float | int]:
    hidden_size = int(model_config["hidden_size"])
    num_heads = int(model_config["num_attention_heads"])
    image_size = int(model_config.get("image_size", 32))
    patch_size = int(model_config.get("patch_size", 4))
    latent_rank = int(model_config["latent_rank"])
    num_patches = (image_size // patch_size) ** 2
    tokens = num_patches + 1

    per_layer_qk_params_baseline = 2 * hidden_size * hidden_size
    per_layer_qk_params_latent = hidden_size * latent_rank

    qk_flops_baseline = (
        4 * tokens * hidden_size * hidden_size
        + 2 * tokens * tokens * hidden_size
    )
    qk_flops_latent = (
        2 * tokens * hidden_size * latent_rank
        + 2 * tokens * tokens * latent_rank
    )

    per_layer_attn_params_baseline = 4 * hidden_size * hidden_size
    per_layer_attn_params_latent = per_layer_qk_params_latent + 2 * hidden_size * hidden_size

    full_attn_flops_baseline = (
        8 * tokens * hidden_size * hidden_size
        + 4 * tokens * tokens * hidden_size
    )
    full_attn_flops_latent = (
        qk_flops_latent
        + 4 * tokens * hidden_size * hidden_size
        + 2 * tokens * tokens * hidden_size
    )

    return {
        "attention_variant": "layer_bbt",
        "latent_rank": latent_rank,
        "num_heads": num_heads,
        "head_dim": hidden_size // num_heads,
        "qk_head_dim": latent_rank,
        "tokens_per_example": tokens,
        "per_layer_qk_params_baseline": per_layer_qk_params_baseline,
        "per_layer_qk_params_latent": per_layer_qk_params_latent,
        "per_layer_qk_params_saved": per_layer_qk_params_baseline - per_layer_qk_params_latent,
        "per_layer_qk_param_reduction_pct": round(
            100.0 * (per_layer_qk_params_baseline - per_layer_qk_params_latent) / per_layer_qk_params_baseline,
            4,
        ),
        "per_layer_qk_flops_baseline": qk_flops_baseline,
        "per_layer_qk_flops_latent": qk_flops_latent,
        "per_layer_qk_flops_reduction_pct": round(
            100.0 * (qk_flops_baseline - qk_flops_latent) / qk_flops_baseline,
            4,
        ),
        "per_layer_attention_params_baseline": per_layer_attn_params_baseline,
        "per_layer_attention_params_latent": per_layer_attn_params_latent,
        "per_layer_attention_param_reduction_pct": round(
            100.0
            * (per_layer_attn_params_baseline - per_layer_attn_params_latent)
            / per_layer_attn_params_baseline,
            4,
        ),
        "per_layer_attention_flops_baseline": full_attn_flops_baseline,
        "per_layer_attention_flops_latent": full_attn_flops_latent,
        "per_layer_attention_flops_reduction_pct": round(
            100.0 * (full_attn_flops_baseline - full_attn_flops_latent) / full_attn_flops_baseline,
            4,
        ),
    }
