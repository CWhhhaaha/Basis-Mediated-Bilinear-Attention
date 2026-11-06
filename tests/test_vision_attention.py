from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch
from transformers import ViTConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmb.layer_bbt_attention import LayerBBTSelfAttention, summarize_layer_bbt_attention
from bmb.layer_fully_shared_attention import (
    LayerFullySharedSelfAttention,
    summarize_layer_fully_shared_attention,
)
from bmb.layer_lowrank_attention import LayerLowRankSelfAttention
from bmb.layer_partial_qk_shared_attention import LayerPartialQKSharedSelfAttention
from bmb.layer_symmetric_latent_attention import LayerSymmetricLatentSelfAttention
from bmb.layer_uv_latent_attention import LayerUVLatentSelfAttention
from bmb.vision_training_recipe import theoretical_attention_summary


def vision_config() -> ViTConfig:
    return ViTConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        image_size=32,
        patch_size=4,
        num_channels=3,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        qkv_bias=True,
    )


class VisionAttentionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.config = vision_config()
        self.x = torch.randn(2, 9, 64)

    def test_bbt_matches_shared_score_reference(self) -> None:
        attention = LayerBBTSelfAttention(self.config, latent_rank=8).eval()
        with torch.no_grad():
            actual, _ = attention(self.x)
            latent = attention.basis(self.x)
            probabilities = torch.softmax(latent @ latent.transpose(-2, -1) / math.sqrt(8), dim=-1)
            values = attention.value(self.x).view(2, 9, 4, 16).permute(0, 2, 1, 3)
            expected = (probabilities.unsqueeze(1) @ values).permute(0, 2, 1, 3).reshape(2, 9, 64)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_all_custom_variants_forward_backward(self) -> None:
        modules = [
            LayerFullySharedSelfAttention(self.config),
            LayerPartialQKSharedSelfAttention(self.config, shared_qk_dim=56),
            LayerLowRankSelfAttention(self.config, low_rank=4),
            LayerBBTSelfAttention(self.config, latent_rank=8),
            LayerSymmetricLatentSelfAttention(self.config, latent_rank=8),
            LayerUVLatentSelfAttention(self.config, latent_rank=8, latent_factor_rank=4),
            LayerUVLatentSelfAttention(self.config, latent_rank=8, latent_factor_rank=8),
        ]
        for attention in modules:
            with self.subTest(variant=type(attention).__name__):
                output, _ = attention(self.x)
                self.assertEqual(output.shape, self.x.shape)
                output.square().mean().backward()
                gradients = [p.grad for p in attention.parameters() if p.requires_grad]
                self.assertTrue(all(gradient is not None for gradient in gradients))
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_bmb_initialization(self) -> None:
        native = LayerSymmetricLatentSelfAttention(self.config, latent_rank=8)
        uv = LayerUVLatentSelfAttention(self.config, latent_rank=8, latent_factor_rank=4)
        matrix_bound = math.sqrt(6.0 / 16.0)
        factor_bound = math.sqrt(6.0 / 12.0)
        self.assertLessEqual(float(native.core.detach().abs().max()), matrix_bound)
        self.assertLessEqual(float(native.head_residual.detach().abs().max()), matrix_bound)
        self.assertLessEqual(float(uv.u_factor.detach().abs().max()), factor_bound)
        self.assertLessEqual(float(uv.v_factor.detach().abs().max()), factor_bound)

    def test_full_shared_and_bbt_flops(self) -> None:
        config = {
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "image_size": 224,
            "patch_size": 16,
            "shared_qk_dim": 728,
            "latent_rank": 64,
        }
        full = summarize_layer_fully_shared_attention(config)
        self.assertEqual(full["per_layer_qk_flops_variant"], full["per_layer_qk_flops_baseline"])
        bbt = summarize_layer_bbt_attention(config)
        expected = 2 * 197 * 768 * 64 + 2 * 197 * 197 * 64
        self.assertEqual(bbt["per_layer_qk_flops_latent"], expected)

    def test_theory_supports_all_eight_methods(self) -> None:
        base = {
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "image_size": 224,
            "patch_size": 16,
            "latent_rank": 64,
            "latent_factor_rank": 32,
            "shared_qk_dim": 728,
            "low_rank": 32,
        }
        variants = [
            "standard",
            "layer_fully_shared",
            "layer_partial_qk_shared",
            "layer_lowrank",
            "layer_bbt",
            "layer_uv_latent",
            "layer_symmetric_latent",
        ]
        for variant in variants:
            with self.subTest(variant=variant):
                result = theoretical_attention_summary({**base, "attention_variant": variant})
                self.assertGreater(result["qk_weight_params_total"], 0)
                self.assertGreater(result["attention_flops_per_example_total"], 0)
        uv64 = theoretical_attention_summary({**base, "attention_variant": "layer_uv_latent", "latent_factor_rank": 64})
        self.assertEqual(uv64["qk_weight_params_total"], 1769472)


if __name__ == "__main__":
    unittest.main()
