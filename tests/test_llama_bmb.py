import json
import math
import unittest
from pathlib import Path

import torch

from bmb.attention import BMBAttention
from bmb.llama_bmb import BMBConfig, BMBForCausalLM, BMBUVAttention


ROOT = Path(__file__).resolve().parents[1]


class LlamaBMBTests(unittest.TestCase):
    def test_small_parameter_count_matches_paper(self):
        raw = json.loads((ROOT / "configs/fineweb_small.json").read_text())
        model = BMBForCausalLM(BMBConfig(**raw["model"]))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 110_592_000)

    def test_xl_parameter_count_matches_paper_on_meta_device(self):
        raw = json.loads((ROOT / "configs/fineweb_xl.json").read_text())
        with torch.device("meta"):
            model = BMBForCausalLM(BMBConfig(**raw["model"]))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1_328_793_600)

    def test_gqa_parameter_counts_on_meta_device(self):
        for scale, expected in [("small", 123_568_128), ("xl", 1_554_892_800)]:
            raw = json.loads((ROOT / f"configs/fineweb_gqa_{scale}.json").read_text())
            with torch.device("meta"):
                model = BMBForCausalLM(BMBConfig(**raw["model"]))
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected)

    def test_token_budgets(self):
        for scale, gpus, expected in [("small", 4, 49_152_000_000), ("xl", 24, 49_152_983_040)]:
            raw = json.loads((ROOT / f"configs/fineweb_{scale}.json").read_text())
            train = raw["training"]
            tokens = (
                gpus * train["micro_batch_size"] * train["gradient_accumulation_steps"]
                * raw["data"]["sequence_length"] * train["max_steps"]
            )
            self.assertEqual(tokens, expected)

    def test_forward_backward_and_explicit_cache(self):
        config = BMBConfig(
            vocab_size=97, n_layer=2, n_head=4, n_kv_head=2, n_embd=32,
            intermediate_size=48, latent_rank=16, factor_rank=8, block_size=32,
        )
        model = BMBForCausalLM(config)
        inputs = torch.randint(0, 97, (2, 7))
        output = model(inputs, labels=inputs, use_cache=True)
        self.assertEqual(output.logits.shape, (2, 7, 97))
        self.assertEqual(output.past_key_values[0][0].shape, (2, 2, 7, 8))
        self.assertEqual(output.past_key_values[0][1].shape, (2, 2, 7, 64))
        output.loss.backward()
        self.assertIsNotNone(model.layers[0].attn.basis.weight.grad)
        token = torch.randint(0, 97, (2, 1))
        decoded = model(token, past_key_values=output.past_key_values, use_cache=True)
        self.assertEqual(decoded.past_key_values[0][0].shape[2], 8)

    def test_cached_decode_matches_full_forward(self):
        config = BMBConfig(
            vocab_size=97, n_layer=1, n_head=4, n_kv_head=2, n_embd=32,
            intermediate_size=48, latent_rank=16, factor_rank=8, block_size=32,
        )
        model = BMBForCausalLM(config).eval()
        inputs = torch.randint(0, 97, (1, 8))
        with torch.no_grad():
            full = model(inputs).logits
            prefix = model(inputs[:, :-1], use_cache=True)
            decoded = model(
                inputs[:, -1:], past_key_values=prefix.past_key_values, use_cache=True
            )
        torch.testing.assert_close(decoded.logits[:, 0], full[:, -1], rtol=1e-4, atol=1e-5)

    def test_chunked_prefill_matches_full_forward(self):
        config = BMBConfig(
            vocab_size=97, n_layer=2, n_head=4, n_kv_head=2, n_embd=32,
            intermediate_size=48, latent_rank=16, factor_rank=8, block_size=32,
        )
        model = BMBForCausalLM(config).eval()
        inputs = torch.randint(0, 97, (1, 8))
        with torch.no_grad():
            full = model(inputs).logits
            past = None
            chunks = []
            for tokens in (inputs[:, :3], inputs[:, 3:6], inputs[:, 6:]):
                output = model(tokens, past_key_values=past, use_cache=True)
                past = output.past_key_values
                chunks.append(output.logits)
        torch.testing.assert_close(
            torch.cat(chunks, dim=1), full, rtol=1e-4, atol=1e-5
        )

    def test_static_cache_append(self):
        config = BMBConfig(
            vocab_size=97, n_layer=1, n_head=4, n_kv_head=2, n_embd=32,
            intermediate_size=48, latent_rank=16, factor_rank=8, block_size=32,
        )
        model = BMBForCausalLM(config).eval()
        cache = [{
            "k": torch.zeros(1, 2, 12, 8),
            "v": torch.zeros(1, 2, 12, 64),
            "length": 0,
        }]
        with torch.no_grad():
            model(torch.randint(0, 97, (1, 7)), past_key_values=cache, use_cache=True)
            self.assertEqual(cache[0]["length"], 7)
            model(torch.randint(0, 97, (1, 1)), past_key_values=cache, use_cache=True)
        self.assertEqual(cache[0]["length"], 8)

    def test_scaled_residual_initialization(self):
        config = BMBConfig(
            vocab_size=97, n_layer=2, n_head=4, n_kv_head=2, n_embd=32,
            intermediate_size=48, latent_rank=16, factor_rank=8,
        )
        model = BMBForCausalLM(config)
        expected_std = 0.02 / math.sqrt(2 * config.n_layer)
        self.assertAlmostEqual(model.layers[0].attn.o_proj.weight.std().item(), expected_std, places=3)
        self.assertAlmostEqual(model.layers[0].mlp.down.weight.std().item(), expected_std, places=3)

    def test_left_padding_is_masked_with_explicit_positions(self):
        config = BMBConfig(
            vocab_size=97, n_layer=1, n_head=4, n_kv_head=2, n_embd=32,
            intermediate_size=48, latent_rank=16, factor_rank=8,
        )
        model = BMBForCausalLM(config).eval()
        inputs = torch.randint(0, 97, (2, 5))
        mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
        positions = torch.tensor([[0, 0, 0, 1, 2], [0, 1, 2, 3, 4]])
        with torch.no_grad():
            batched = model(
                inputs, attention_mask=mask, position_ids=positions
            ).logits
            expected = model(inputs[0:1, 2:]).logits
        self.assertTrue(torch.isfinite(batched).all())
        torch.testing.assert_close(
            batched[0:1, 2:], expected, rtol=1e-4, atol=1e-5
        )

    def test_model_adapter_and_standalone_share_parameters_and_outputs(self):
        config = BMBConfig(
            n_layer=1, n_head=4, n_kv_head=2, n_embd=32,
            latent_rank=16, factor_rank=8, value_dim=12,
        )
        adapter = BMBUVAttention(config).double().eval()
        standalone = BMBAttention(
            d_model=32, n_heads=4, n_kv_heads=2,
            latent_rank=16, factor_rank=8, value_dim=12,
        ).double().eval()
        standalone.load_state_dict(adapter.state_dict())
        x = torch.randn(2, 6, 32, dtype=torch.float64)
        with torch.no_grad():
            expected, _ = adapter(x)
            actual, _ = standalone(x)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_cache_narrowing_is_25_percent(self):
        gqa = 64 + 64
        bmbuv = 32 + 64
        self.assertEqual(1 - bmbuv / gqa, 0.25)


if __name__ == "__main__":
    unittest.main()
