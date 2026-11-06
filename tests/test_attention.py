from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmb.attention_accounting import attention_forward_accounting
from bmb.gpt2_attention_variants import (
    GPT2BBTAttention,
    GPT2FullySharedAttention,
    GPT2LowRankAttention,
    GPT2PartialSharedAttention,
    GPT2SymmetricLatentAttention,
    GPT2UVLatentAttention,
)


def gpt_config() -> SimpleNamespace:
    return SimpleNamespace(
        n_head=4,
        n_embd=64,
        n_layer=2,
        n_positions=16,
        initializer_range=0.02,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
    )


class AccountingTests(unittest.TestCase):
    def test_gpt2_forward_counts(self) -> None:
        expected = {
            "baseline": (14155776, 28311552, 48318382080, 96636764160),
            "fullyshared": (7077888, 21233664, 48318382080, 96636764160),
            "partialshared": (7446528, 21602304, 34577842176, 82896224256),
            "lowrank": (7667712, 21823488, 35030827008, 83349209088),
            "bbt": (589824, 14745600, 2818572288, 51136954368),
            "bmbuv": (589824, 14745600, 10871635968, 59190018048),
            "bmb": (1228800, 15384576, 21743271936, 70061654016),
        }
        for variant, values in expected.items():
            with self.subTest(variant=variant):
                kwargs = {"rank": 32} if variant == "lowrank" else {"rank": 64}
                if variant == "bmbuv":
                    kwargs = {"rank": 32, "factor_rank": 32}
                if variant == "partialshared":
                    kwargs = {"shared_dim": 728}
                if variant in {"baseline", "fullyshared"}:
                    kwargs = {}
                result = attention_forward_accounting(
                    variant,
                    model_dim=768,
                    num_heads=12,
                    num_layers=12,
                    tokens=1024,
                    **kwargs,
                )
                actual = (
                    result["qk_params_total"],
                    result["attention_params_total"],
                    result["qk_flops_total"],
                    result["attention_flops_total"],
                )
                self.assertEqual(actual, values)

    def test_bbt_score_is_not_replicated_per_head(self) -> None:
        result = attention_forward_accounting(
            "bbt", model_dim=768, num_heads=12, num_layers=12, tokens=1024, rank=64
        )
        expected = 12 * 2 * (1024 * 768 * 64 + 1024 * 1024 * 64)
        self.assertEqual(result["qk_flops_total"], expected)


class GPT2AttentionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = gpt_config()
        self.x = torch.randn(2, 6, 64)

    def test_partshr_uses_layer_level_dimensions(self) -> None:
        attention = GPT2PartialSharedAttention(self.config, shared_qk_dim=56)
        self.assertEqual(attention.shared_qk_dim, 56)
        self.assertEqual(attention.private_qk_dim, 8)
        self.assertEqual(attention.share.out_features, 56)
        self.assertEqual(attention.query_priv.out_features, 8)
        y, _, weights = attention(self.x, output_attentions=True)
        self.assertEqual(y.shape, self.x.shape)
        self.assertEqual(weights.shape, (2, 4, 6, 6))

    def test_partshr_728_construction(self) -> None:
        config = gpt_config()
        config.n_embd = 768
        config.n_head = 12
        attention = GPT2PartialSharedAttention(config, shared_qk_dim=728)
        self.assertEqual(attention.private_qk_dim, 40)
        qk_params = sum(
            parameter.numel()
            for module in (attention.share, attention.query_priv, attention.key_priv)
            for parameter in module.parameters()
            if parameter.ndim == 2
        )
        self.assertEqual(qk_params, 620544)

    def test_bbt_matches_shared_score_reference(self) -> None:
        attention = GPT2BBTAttention(self.config, latent_rank=8).eval()
        with torch.no_grad():
            actual = attention(self.x)[0]
            latent = attention.basis(self.x)
            scores = latent @ latent.transpose(-2, -1) / math.sqrt(8)
            mask = torch.triu(torch.full((6, 6), float("-inf")), diagonal=1)
            probabilities = torch.softmax(scores + mask, dim=-1)
            values = attention.v_proj(self.x).view(2, 6, 4, 16).transpose(1, 2)
            expected = (probabilities.unsqueeze(1) @ values).transpose(1, 2).reshape(2, 6, 64)
            expected = attention.o_proj(expected)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_bmb_xavier_initialization(self) -> None:
        attention = GPT2SymmetricLatentAttention(self.config, latent_rank=8)
        basis_bound = math.sqrt(6.0 / (64 + 8))
        matrix_bound = math.sqrt(6.0 / (8 + 8))
        self.assertLessEqual(float(attention.basis.weight.detach().abs().max()), basis_bound)
        self.assertLessEqual(float(attention.core.detach().abs().max()), matrix_bound)
        self.assertLessEqual(float(attention.head_residual.detach().abs().max()), matrix_bound)
        self.assertGreater(float(attention.head_residual.detach().std()), 0.1)

    def test_bmbuv_xavier_initialization(self) -> None:
        attention = GPT2UVLatentAttention(self.config, latent_rank=8, latent_factor_rank=4)
        factor_bound = math.sqrt(6.0 / (8 + 4))
        self.assertLessEqual(float(attention.u_factor.detach().abs().max()), factor_bound)
        self.assertLessEqual(float(attention.v_factor.detach().abs().max()), factor_bound)

    def test_bmbuv_paper_rank_configurations(self) -> None:
        config = gpt_config()
        config.n_embd = 768
        config.n_head = 12
        x = torch.randn(1, 4, 768)
        for rank in (32, 64):
            with self.subTest(rank=rank):
                attention = GPT2UVLatentAttention(
                    config, latent_rank=rank, latent_factor_rank=rank
                )
                output = attention(x)[0]
                self.assertEqual(output.shape, x.shape)
                output.square().mean().backward()
                self.assertTrue(
                    all(
                        parameter.grad is not None and torch.isfinite(parameter.grad).all()
                        for parameter in attention.parameters()
                        if parameter.requires_grad
                    )
                )

    def test_all_variants_forward_backward_and_causality(self) -> None:
        modules = {
            "lowrank": GPT2LowRankAttention(self.config, low_rank=4),
            "fullyshared": GPT2FullySharedAttention(self.config),
            "bbt": GPT2BBTAttention(self.config, latent_rank=8),
            "bmb": GPT2SymmetricLatentAttention(self.config, latent_rank=8),
            "bmbuv_narrow_factors": GPT2UVLatentAttention(self.config, latent_rank=8, latent_factor_rank=4),
            "bmbuv_equal_ranks": GPT2UVLatentAttention(self.config, latent_rank=8, latent_factor_rank=8),
            "partialshared": GPT2PartialSharedAttention(self.config, shared_qk_dim=56),
        }
        changed = self.x.clone()
        changed[:, 4:] = torch.randn_like(changed[:, 4:])
        for name, attention in modules.items():
            with self.subTest(variant=name):
                attention.eval()
                with torch.no_grad():
                    first = attention(self.x)[0]
                    second = attention(changed)[0]
                torch.testing.assert_close(first[:, :4], second[:, :4], rtol=1e-5, atol=1e-6)
                attention.train()
                output = attention(self.x)[0]
                output.square().mean().backward()
                gradients = [p.grad for p in attention.parameters() if p.requires_grad]
                self.assertTrue(all(gradient is not None for gradient in gradients))
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
