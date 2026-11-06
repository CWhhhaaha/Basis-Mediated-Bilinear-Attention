import unittest

import torch
import torch.nn.functional as F

from bmb.attention import BMBAttention, grouped_sdpa


class PlugAndPlayAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.attention = BMBAttention(
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            latent_rank=16,
            factor_rank=8,
            value_dim=12,
        ).double().eval()
        self.x = torch.randn(2, 7, 32, dtype=torch.float64)

    def assertClose(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)

    def test_forward_backward_and_cache_shapes(self):
        attention = BMBAttention(
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            latent_rank=16,
            factor_rank=8,
            value_dim=12,
        ).double().train()
        x = self.x.clone().requires_grad_(True)
        output, cache = attention(x, use_cache=True)
        self.assertEqual(output.shape, self.x.shape)
        self.assertEqual(cache[0].shape, (2, 2, 7, 8))
        self.assertEqual(cache[1].shape, (2, 2, 7, 12))
        output.square().mean().backward()
        for parameter in (
            attention.basis.weight,
            attention.q_factor,
            attention.k_factor,
            attention.v_proj.weight,
            attention.o_proj.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertIsNotNone(x.grad)

    def test_full_forward_matches_token_by_token_decode(self):
        with torch.no_grad():
            expected, _ = self.attention(self.x)
            cache = None
            outputs = []
            for index in range(self.x.shape[1]):
                output, cache = self.attention(
                    self.x[:, index:index + 1],
                    past_key_value=cache,
                    use_cache=True,
                )
                outputs.append(output)
        self.assertClose(torch.cat(outputs, dim=1), expected)
        self.assertEqual(cache[0].shape[2], 7)

    def test_chunked_prefill_uses_offset_causal_mask(self):
        with torch.no_grad():
            expected, _ = self.attention(self.x)
            cache = None
            outputs = []
            for chunk in (self.x[:, :3], self.x[:, 3:5], self.x[:, 5:]):
                output, cache = self.attention(
                    chunk, past_key_value=cache, use_cache=True
                )
                outputs.append(output)
        self.assertClose(torch.cat(outputs, dim=1), expected)

    def test_static_cache_matches_dynamic_cache(self):
        static = self.attention.create_static_cache(
            batch_size=2, max_length=8, dtype=torch.float64
        )
        with torch.no_grad():
            expected, _ = self.attention(self.x)
            first, static = self.attention(
                self.x[:, :4], past_key_value=static, use_cache=True
            )
            second, static = self.attention(
                self.x[:, 4:], past_key_value=static, use_cache=True
            )
        self.assertClose(torch.cat((first, second), dim=1), expected)
        self.assertEqual(static["length"], 7)
        self.assertEqual(static["k"].shape, (2, 2, 8, 8))
        self.assertEqual(static["v"].shape, (2, 2, 8, 12))

    def test_left_padding_and_decode_match_unpadded_sequences(self):
        lengths = [3, 5]
        x = self.x[:, :5].clone()
        mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
        positions = torch.tensor([[0, 0, 0, 1, 2], [0, 1, 2, 3, 4]])
        with torch.no_grad():
            batched, cache = self.attention(
                x, attention_mask=mask, position_ids=positions, use_cache=True
            )
            for row, length in enumerate(lengths):
                start = 5 - length
                expected, _ = self.attention(x[row:row + 1, start:])
                self.assertClose(batched[row:row + 1, start:], expected)
            self.assertTrue(torch.isfinite(batched).all())

            next_token = torch.randn(2, 1, 32, dtype=torch.float64)
            next_mask = torch.cat((mask, torch.ones(2, 1, dtype=torch.long)), dim=1)
            next_positions = torch.tensor([[3], [5]])
            decoded, _ = self.attention(
                next_token,
                attention_mask=next_mask,
                position_ids=next_positions,
                past_key_value=cache,
                use_cache=True,
            )
            extended = torch.cat((x, next_token), dim=1)
            extended_positions = torch.cat((positions, next_positions), dim=1)
            full, _ = self.attention(
                extended,
                attention_mask=next_mask,
                position_ids=extended_positions,
            )
        self.assertClose(decoded, full[:, -1:])

    def test_additive_and_boolean_4d_masks_agree(self):
        length = self.x.shape[1]
        allowed = torch.ones(1, 1, length, length, dtype=torch.bool)
        allowed[..., -1] = False
        additive = torch.zeros(allowed.shape, dtype=torch.float64).masked_fill(
            ~allowed, float("-inf")
        )
        with torch.no_grad():
            from_bool, _ = self.attention(self.x, attention_mask=allowed)
            from_float, _ = self.attention(self.x, attention_mask=additive)
        self.assertClose(from_bool, from_float)

    def test_grouped_attention_matches_explicit_head_repetition(self):
        q = torch.randn(2, 4, 5, 8, dtype=torch.float64)
        k = torch.randn(2, 2, 5, 8, dtype=torch.float64)
        v = torch.randn(2, 2, 5, 12, dtype=torch.float64)
        actual = grouped_sdpa(q, k, v, is_causal=True)
        expected = F.scaled_dot_product_attention(
            q,
            k.repeat_interleave(2, dim=1),
            v.repeat_interleave(2, dim=1),
            is_causal=True,
        )
        self.assertClose(actual, expected)

    def test_static_cache_overflow_is_reported(self):
        cache = self.attention.create_static_cache(2, 3, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "capacity"):
            self.attention(self.x[:, :4], past_key_value=cache, use_cache=True)

    def test_mask_shape_is_validated(self):
        with self.assertRaisesRegex(ValueError, "attention_mask"):
            self.attention(self.x, attention_mask=torch.ones(2, 6))


if __name__ == "__main__":
    unittest.main()
